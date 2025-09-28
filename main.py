from gui import RunGUI

"""
Simplex Solver GUI with Table Input (PySide6)

Features:
- Select number of variables and constraints.
- Enter objective function coefficients in a table row.
- Enter constraints in a table (coefficients and RHS, with comparator).
- Solve using the Simplex Method.

How to run (with uv):
uv venv
uv sync
uv run main.py

How to run (without uv):

Set up virtual environment (might be optional)
1. python -m venv .venv
2. source .venv/Scripts/activate.bat

Install PySide 6
3. pip install pyside6 tabulate
4. python main.py
"""

if __name__ == "__main__":
    RunGUI()
