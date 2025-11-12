import subprocess
from pathlib import Path
from typing import Dict, Any, List

def _parse_with_hwp5txt(path: Path) -> List[str]:
    res = subprocess.run(["hwp5txt", str(path)], capture_output=True, text=True, check=True)
    text = res.stdout
    chunks = text.split("\f")
    return [c.strip() for c in chunks if c.strip()]

def parse_hwp(path: Path) -> Dict[str, Any]:
    try:
        pages = _parse_with_hwp5txt(path)
    except Exception as e:
        raise RuntimeError(f"HWP parsing failed: {e}")

    return {
        "file_path": str(path),
        "n_pages": len(pages),
        "pages": [{"page": i + 1, "text": t} for i, t in enumerate(pages)],
    }
