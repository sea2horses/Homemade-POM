
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QSpinBox,
                               QTableWidget, QTableWidgetItem, QComboBox, QPushButton,
                               QVBoxLayout, QHBoxLayout, QMessageBox, QPlainTextEdit, QInputDialog)
from PySide6.QtCore import Qt
import sys
from fractions import Fraction
from models import MO_Model, ConstraintType, Simplex_Solver, ModelObjective
import traceback


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Homemade POM")
        self.resize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        self.widget_layout = QVBoxLayout()

        # Problem size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Variables:"))
        self.var_spin = QSpinBox()
        self.var_spin.setValue(3)
        self.var_spin.setMinimum(1)
        size_layout.addWidget(self.var_spin)

        size_layout.addWidget(QLabel("Constraints:"))
        self.con_spin = QSpinBox()
        self.con_spin.setValue(3)
        self.con_spin.setMinimum(1)
        size_layout.addWidget(self.con_spin)

        self.update_btn = QPushButton("Update Tables")
        self.update_btn.clicked.connect(self.build_tables)
        size_layout.addWidget(self.update_btn)

        self.widget_layout.addLayout(size_layout)

        # Objective function table
        self.widget_layout.addWidget(QLabel("Objective Function:"))
        self.obj_table = QTableWidget(1, 3)
        self.widget_layout.addWidget(self.obj_table)

        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["Maximize", "Minimize"])
        self.widget_layout.addWidget(self.opt_combo)

        # Constraints table
        self.widget_layout.addWidget(
            QLabel("Constraints (non-negativity is assumed):"))
        self.cons_table = QTableWidget(3, 5)
        self.widget_layout.addWidget(self.cons_table)

        # Buttons
        btn_layout = QHBoxLayout()
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self.on_solve)
        btn_layout.addWidget(self.solve_btn)
        self.widget_layout.addLayout(btn_layout)

        # Output
        self.widget_layout.addWidget(QLabel("Output:"))
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.widget_layout.addWidget(self.output)

        central.setLayout(self.widget_layout)
        self.build_tables()

    def build_tables(self):
        n = self.var_spin.value()
        m = self.con_spin.value()

        self.obj_table.setColumnCount(n)
        self.obj_table.setRowCount(1)
        self.obj_table.setHorizontalHeaderLabels([f"x{i+1}" for i in range(n)])

        self.cons_table.clear()
        self.cons_table.setColumnCount(n+2)  # coeffs + comparator + RHS
        self.cons_table.setRowCount(m)
        headers = [f"x{i+1}" for i in range(n)] + ["Type", "RHS"]
        self.cons_table.setHorizontalHeaderLabels(headers)
        for i in range(m):
            comp = QComboBox()
            comp.addItems(["<=", "=", ">="])
            self.cons_table.setCellWidget(i, n, comp)

    def get_obj_coeffs(self):
        n = self.obj_table.columnCount()
        coeffs = []
        for j in range(n):
            item = self.obj_table.item(0, j)
            val = Fraction(item.text()) if item else Fraction(0)
            coeffs.append(val)
        # Not necessary, MO does it itself
        # if self.opt_combo.currentText() == "Minimize":
        #     coeffs = [-c for c in coeffs]
        return coeffs

    def get_constraints(self):
        m = self.cons_table.rowCount()
        n = self.var_spin.value()
        cons = []
        for i in range(m):
            coeffs = []
            for j in range(n):
                item = self.cons_table.item(i, j)
                val = Fraction(item.text()) if item else Fraction(0)
                coeffs.append(val)
            comp_widget = self.cons_table.cellWidget(i, n)
            comp = comp_widget.currentText() if comp_widget else "<="
            rhs_item = self.cons_table.item(i, n+1)
            rhs = Fraction(rhs_item.text()) if rhs_item else Fraction(0)
            cons.append((coeffs, comp, rhs))
        return cons

    def input_method(self, prompt) -> str:
        text = ""
        ok = False
        while not ok:
            text, ok = QInputDialog.getText(self, "Input", prompt)
        return text

    def on_solve(self):
        try:
            obj = self.get_obj_coeffs()
            cons = self.get_constraints()
            model = MO_Model(len(obj), len(cons))
            model.set_objective_function(obj, objective=(
                ModelObjective.MAXIMIZE if self.opt_combo.currentText() == "Maximize" else ModelObjective.MINIMIZE))
            for coeffs, comp, rhs in cons:
                ctype = ConstraintType.str_to_constrainttype(comp)
                model.add_constraint(coeffs, ctype, rhs)

            solver = Simplex_Solver(
                model, print_function=lambda msg: self.output.appendPlainText(msg), input_function=self.input_method)

            self.output.clear()
            solution, objval = solver.solve()
            if self.opt_combo.currentText() == "Minimize":
                objval = -objval

            self.output.appendPlainText("\nFinal Solution:")
            for i, v in enumerate(solution, start=1):
                self.output.appendPlainText(
                    f"x{i} = {float(v):.6g} ({str(v)})")
            self.output.appendPlainText(
                f"Objective value = {float(objval):.6g} ({str(objval)})")
        except Exception as e:
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", str(e))


def RunGUI():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
