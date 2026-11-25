# src/tools/sandbox.py
import subprocess, sys, tempfile
from pathlib import Path
from src.utils.helpers import PROJECT_ROOT


def run_code(code: str, timeout: int = 120) -> dict:
    """Run generated Python code in a subprocess with a time limit.
    Not a security boundary: only run code on your own data and machine."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True,
                timeout=timeout, cwd=PROJECT_ROOT)
            return {"success": p.returncode == 0,
                    "output": p.stdout[-5000:],
                    "error": p.stderr[-5000:]}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "",
                    "error": f"Timeout after {timeout} seconds"}