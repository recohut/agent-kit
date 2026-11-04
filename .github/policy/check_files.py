"""Reject tracked files that commonly contain local or private data."""
from pathlib import Path
from subprocess import check_output

for raw_path in check_output(["git", "ls-files", "-z"]).split(b"\0"):
    if not raw_path:
        continue
    path = Path(raw_path.decode())
    if path.is_symlink() or path.name.startswith((".env", "service-account")) or (
        path.name == "credentials.json"
        or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".db", ".sqlite", ".sqlite3", ".log"}
    ):
        raise SystemExit(f"Blocked tracked file: {path}")
