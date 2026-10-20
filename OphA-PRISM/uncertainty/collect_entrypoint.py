"""Directory entry point; forwards to the root CLI with the documented command."""
from pathlib import Path
import subprocess, sys
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.exit(subprocess.run([sys.executable, str(root/"run_pipeline.py"), 'collect', *sys.argv[1:]], check=False).returncode)
