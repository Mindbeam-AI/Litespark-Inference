"""
Minimal torchless safetensors reader with two access modes:

- `get(name)` -- pread into a short-lived bytes buffer, appropriate for
  tensors we consume then discard (e.g. weights we immediately quantize).
- `view(name)` -- zero-copy numpy view over the underlying mmap. The
  SafetensorsFile object must outlive any view returned this way; when
  it's closed, the views become invalid.

bf16 is returned as uint16 (numpy has no native bfloat16); callers that
need the fp32 value must reinterpret the bits with
((uint16.astype(uint32) << 16).view(float32)).
"""

from __future__ import annotations

import json
import math
import mmap
import os
from dataclasses import dataclass

import numpy as np


_MAX_HEADER_SIZE = 100 * 1024 * 1024  # 100 MB, matching safetensors spec


_NP_STORE = {
    "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "F64": np.float64,
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "BOOL": np.bool_,
}


@dataclass
class SafetensorsFile:
    fd: int
    mm: "mmap.mmap | None"
    header: dict
    data_start: int
    path: str = ""
    file_size: int = 0

    def close_mmap(self) -> bool:
        """Drop the mmap only, keep the fd (so pread-based get() still works).

        Useful for int4/int8 loaders that finish touching the embedding as
        an mmap view early, then do the rest of the load via pread: closing
        the mmap lets the kernel reclaim 626 MB of faulted embedding pages
        that would otherwise stay resident for the whole layer loop.

        Returns True if the mmap was released, False if a live view was
        blocking it (caller must drop the view and retry).
        """
        if self.mm is None:
            return True
        try:
            self.mm.close()
        except BufferError:
            return False
        self.mm = None
        return True

    def close(self) -> None:
        # Note: closing while live `view()` arrays exist invalidates them.
        # The torchless loader keeps the SafetensorsFile alive via
        # PackedBitNetModel._safetensors for as long as any zero-copy view
        # is held.
        self.close_mmap()
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def _meta(self, name: str) -> tuple[list, str, int, int]:
        if name == "__metadata__":
            raise KeyError(name)
        meta = self.header[name]
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid metadata entry for {name!r}")
        shape = meta.get("shape")
        dtype = meta.get("dtype")
        offsets = meta.get("data_offsets")
        if dtype not in _NP_STORE:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {name!r}")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise ValueError(f"Invalid shape {shape!r} for {name!r}")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            raise ValueError(f"Invalid data_offsets {offsets!r} for {name!r}")
        off0, off1 = offsets
        if (
            not isinstance(off0, int) or isinstance(off0, bool)
            or not isinstance(off1, int) or isinstance(off1, bool)
            or off0 < 0 or off1 < off0
        ):
            raise ValueError(f"Invalid data_offsets {offsets!r} for {name!r}")
        if self.file_size > 0 and self.data_start + off1 > self.file_size:
            raise ValueError(
                f"Tensor {name!r} data_offsets [{off0}, {off1}] exceed file size {self.file_size}"
            )
        expected_nbytes = math.prod(shape) * np.dtype(_NP_STORE[dtype]).itemsize
        if (off1 - off0) != expected_nbytes:
            raise ValueError(
                f"Tensor {name!r} byte length {off1 - off0} does not match "
                f"shape {shape} * dtype {dtype} ({expected_nbytes} bytes)"
            )
        return shape, dtype, off0, off1

    def get(self, name: str) -> tuple[np.ndarray, str]:
        """
        Copy-read into a fresh bytes object and return a numpy view over it.
        Use for tensors that get consumed and dropped quickly so the 626 MB+
        of embedding weights don't stay resident twice.

        Uses os.pread on POSIX for atomicity; falls back to lseek + read on
        Windows (which lacks pread). The loader is single-threaded, so the
        seek-position side-effect of the fallback is safe.
        """
        shape, dtype, off0, off1 = self._meta(name)
        nbytes = off1 - off0
        np_store = _NP_STORE[dtype]
        offset = self.data_start + off0
        if hasattr(os, "pread"):
            buf = os.pread(self.fd, nbytes, offset)
        elif self.mm is not None:
            # Windows + still-open mmap: slice it directly. Same profile
            # as pread (caller-owned bytes, no fd-state mutation).
            buf = bytes(self.mm[offset:offset + nbytes])
        else:
            # Windows + close_mmap() was already called: re-open a tiny
            # mmap for this read. Stays inexpensive because Windows
            # mmap is mapped lazily by page.
            with open(self.path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm2:
                    buf = bytes(mm2[offset:offset + nbytes])
        return np.frombuffer(buf, dtype=np_store).reshape(shape), dtype

    def view(self, name: str) -> tuple[np.ndarray, str]:
        """
        Zero-copy numpy view into the mmap. The returned array must not
        outlive this SafetensorsFile object.
        """
        if self.mm is None:
            raise RuntimeError("SafetensorsFile has been closed")
        shape, dtype, off0, off1 = self._meta(name)
        nbytes = off1 - off0
        np_store = _NP_STORE[dtype]
        count = nbytes // np_store().itemsize
        return (
            np.frombuffer(
                self.mm, dtype=np_store,
                count=count, offset=self.data_start + off0,
            ).reshape(shape),
            dtype,
        )


def open_safetensors(path: str) -> SafetensorsFile:
    fd = os.open(path, os.O_RDONLY)
    mm = None
    try:
        size = os.fstat(fd).st_size
        if size < 8:
            raise ValueError(f"Safetensors file too small ({size} bytes): {path}")
        # POSIX uses prot=PROT_READ; Windows uses access=ACCESS_READ. Pick the
        # right keyword so the mmap is read-only on both platforms.
        if hasattr(mmap, "PROT_READ"):
            mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        else:
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        header_len = int.from_bytes(mm[:8], "little")
        if header_len <= 0 or header_len > _MAX_HEADER_SIZE or 8 + header_len > size:
            raise ValueError(
                f"Invalid safetensors header length {header_len} for file size {size}"
            )
        header = json.loads(bytes(mm[8:8 + header_len]).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be a JSON object")
        return SafetensorsFile(
            fd=fd, mm=mm, header=header, data_start=8 + header_len,
            path=path, file_size=size,
        )
    except Exception:
        if mm is not None:
            mm.close()
        os.close(fd)
        raise
