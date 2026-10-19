"""Entry point: python run.py <command> ...  (see adkit/render.py for commands)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from adkit.render import main
    main()
