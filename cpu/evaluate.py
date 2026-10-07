"""Run the challenge on CPU: python cpu/evaluate.py [--episodes N] [--view] [--videos N] [--level training|dev]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vinci_cpu.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
