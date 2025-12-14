
from __future__ import annotations
import shutil
from pathlib import Path


def pack_artifacts(src: Path | str = Path("artifacts/v1"), dest: Path | str = Path("artifacts_v1.zip")) -> Path:
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Artifacts folder not found: {src_path}")
    dest_path = Path(dest).expanduser().resolve()
    base = dest_path.with_suffix('')
    archive = shutil.make_archive(str(base), 'zip', root_dir=src_path)
    return Path(archive)


if __name__ == "__main__":
    out = pack_artifacts()
    print(f"Created archive: {out}")
