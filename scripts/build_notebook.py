"""Build a Jupyter notebook from a plain Python file and execute it in place.

Cells in the source file are separated by lines starting with "# %%"; a cell whose first
line is "# %% [markdown]" becomes a markdown cell (its lines have the leading "# " removed).

Usage:
    python scripts/build_notebook.py notebooks/src/01_data_exploration.py notebooks/01_data_exploration.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent


def parse_cells(source: str) -> list:
    """Split "# %%"-delimited source into notebook cells."""
    cells, kind, lines = [], None, []

    def flush() -> None:
        text = "\n".join(lines).strip("\n")
        if kind is not None and text:
            if kind == "markdown":
                text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
                cells.append(nbformat.v4.new_markdown_cell(text))
            else:
                cells.append(nbformat.v4.new_code_cell(text))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()
    return cells


def main() -> None:
    """Build and execute the notebook (working directory = project root)."""
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    nb = nbformat.v4.new_notebook()
    nb.cells = parse_cells(src.read_text(encoding="utf-8"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dst)
    print(f"Executed and saved {dst}")


if __name__ == "__main__":
    main()
