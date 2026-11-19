import os
import stat

def private_dir(path):
    #create a private cache and refuse if unsafe existing directories exists
    os.makedirs(path, mode=0o700, exist_ok=True)
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) & 0o077:
            raise PermissionError(
                f"Unsafe cache directory: {path}; use a new private cache directory"
            )
    finally:
        os.close(fd)