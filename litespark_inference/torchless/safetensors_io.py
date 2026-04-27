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
import mmap
import os
from dataclasses import dataclass

import numpy as np


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
        meta = self.header[name]
        off0, off1 = meta["data_offsets"]
        return meta["shape"], meta["dtype"], off0, off1

    def get(self, name: str) -> tuple[np.ndarray, str]:
        """
        Copy-read: pread into a fresh bytes object and return a numpy view
        over it. Use for tensors that get consumed and dropped quickly so
        the 626 MB+ of embedding weights don't stay resident twice.
        """
        shape, dtype, off0, off1 = self._meta(name)
        nbytes = off1 - off0
        np_store = _NP_STORE[dtype]
        buf = os.pread(self.fd, nbytes, self.data_start + off0)
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
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    header_len = int.from_bytes(mm[:8], "little")
    header = json.loads(bytes(mm[8:8 + header_len]).decode())
    return SafetensorsFile(fd=fd, mm=mm, header=header, data_start=8 + header_len)
