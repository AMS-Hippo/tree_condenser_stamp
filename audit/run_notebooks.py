"""Execute the current schema-1 example notebooks without modifying them."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks"
    with tempfile.TemporaryDirectory(prefix="tree-coarsening-notebooks-") as directory:
        output_dir = Path(directory)
        for path in sorted(notebook_dir.glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(root)}},
            ).execute()
            nbformat.write(notebook, output_dir / path.name)
            print(f"PASS {path.relative_to(root)}")


if __name__ == "__main__":
    main()
