"""
fix_encoding.py
---------------
One-off utility: if any .py file in this folder got saved with UTF-16
encoding (common after certain Windows/OneDrive sync round-trips), Python
throws "SyntaxError: source code string cannot contain null bytes" when
importing it. This script detects that and re-saves the file as plain
UTF-8, backing up the original first.

Run once from the backend folder:
    python fix_encoding.py
"""

import os
import glob
import time

TARGET_FILES = glob.glob("*.py")
TARGET_FILES = [f for f in TARGET_FILES if f != os.path.basename(__file__)]


def try_fix(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()

    if b"\x00" not in raw:
        return "ok (already UTF-8, no change needed)"

    # Back up the original before touching anything.
    backup_path = path + ".bak"
    with open(backup_path, "wb") as f:
        f.write(raw)

    fixed_encoding = None
    for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(encoding)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            fixed_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if fixed_encoding is None:
        return (
            "COULD NOT AUTO-FIX — encoding not recognised. "
            f"Original backed up as {backup_path}. Please fix this file "
            "manually."
        )

    # VERIFY the fix actually stuck — a sync tool (OneDrive, Dropbox, etc.)
    # watching this folder can silently overwrite the file with its own
    # cloud copy a moment after we write it. Check right away, and again
    # after a short delay, to catch that.
    for delay in (0, 1.5):
        time.sleep(delay)
        with open(path, "rb") as f:
            check = f.read()
        if b"\x00" in check:
            return (
                f"fixed (was {fixed_encoding}) but REVERTED itself within {delay}s — "
                "something else is overwriting this file after we save it. "
                "This is almost always a sync tool (OneDrive/Dropbox) restoring "
                "its own cloud copy. Pause sync or move this project outside "
                "the synced folder, then re-run this script. "
                f"Backup of the broken version is at {backup_path}."
            )

    return f"fixed (was {fixed_encoding}) and verified stable — backup saved as {backup_path}"


if __name__ == "__main__":
    if not TARGET_FILES:
        print("No .py files found in this folder — are you in the backend/ directory?")
    for filename in sorted(TARGET_FILES):
        result = try_fix(filename)
        print(f"{filename}: {result}")