# Represents a Linear Programming Model
from fractions import Fraction
from tabulate import tabulate
from copy import deepcopy
from enum import Enum


class ModelObjective(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


class ConstraintType(Enum):
    LIMITATION = 0  # <=
    EXACTITUDE = 1  # =
    REQUIREMENT = 2  # >=

    @staticmethod
    def str_to_constrainttype(string: str):
        match string:
            case "<=":
                return ConstraintType.LIMITATION
            case "=":
                return ConstraintType.EXACTITUDE
            case ">=":
                return ConstraintType.REQUIREMENT
            case _:
                raise Exception(f"{string} is not a valid constraint type.")


class MO_Model:

    def __init__(self, variable_count: int, constraint_count: int):
        self.varcount = variable_count
        self.concount = constraint_count
        self.varcount: int
        self.concount: int
        self.objective_function: tuple[list[Fraction],
                                       ModelObjective] | None = None
        self.constraints: list[tuple[list[Fraction],
                                     ConstraintType, Fraction]] = []

    def set_objective_function(self, coefficients: list[Fraction], objective: ModelObjective):
        if len(coefficients) != self.varcount:
            raise ValueError(
                f"Quantity of coefficients {len(coefficients)} exceed set variable count for the model {self.varcount} in objective function")

        for coefficient in coefficients:
            if coefficient <= 0:
                raise ValueError(
                    "All coefficients in objective function MUST be positive")
        self.objective_function = (coefficients, objective)

    def add_constraint(self, coefficients: list[Fraction], constraint: ConstraintType, result: Fraction):
        if len(coefficients) != self.varcount:
            raise ValueError(
                f"Quantity of coefficients {len(coefficients)} exceed set variable count for the model {self.varcount} in added constraint")

        # Add the tuple
        self.constraints.append((coefficients, constraint, result))


SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def var_index(string: str, num: int) -> str:
    return f"{string}{num}".translate(SUB)

# Return x1 when given 1, x2 when given 2 etc...


def x_index(num: int) -> str:
    return var_index("x", num)


def a_index(num: int) -> str:
    return var_index("a", num)


def s_index(num: int) -> str:
    return var_index("s", num)


class Simplex_Solver:
    # Constructor
    def __init__(self, model: MO_Model, print_function=print, input_function=input, debug=False):
        self.model = model
        self.solution = None
        self.log = print_function
        self.input = input_function
        self.debug = debug

    def invert_pivot(self, valor):
        valor = (valor)
        return 1 / valor

    def get_column_value(self, valor, pivote):
        valor = (valor)
        pivote = (pivote)
        return - (valor / pivote)

    def get_row_value(self, valor, pivote):
        valor = (valor)
        pivote = (pivote)
        return valor / pivote

    def get_new_value(self, valor_viejo, pivote, valor_columna, valor_fila):
        valor_viejo = (valor_viejo)
        pivote = (pivote)
        valor_columna = (valor_columna)
        valor_fila = (valor_fila)
        # print(
        #     f"Se llamo a la formula del nuevo valor y se aplicara con la siguiente formula: ( ({valor_viejo} * {pivote}) - ({valor_columna} * {valor_fila}) ) / {pivote}")
        return ((valor_viejo * pivote) - (valor_columna * valor_fila)) / pivote

    def _simplex_solver(self, non_basic_vars: list[str], basic_vars: list[str], value_grid: list[list[Fraction]], z_row: list[Fraction], results_column: list[Fraction], variable_value_list: dict[str, Fraction]):

        def print_table():
            print_string: str = ""
            print_string += ("cj\tB\t")
            for v in non_basic_vars:
                print_string += (f"{v}\t")
            print_string += (f"bi\t\n")

            # Imprimir cada una de las filas
            for i in range(0, len(basic_vars)):
                print_string += (f"{variable_value_list[basic_vars[i]]}\t")
                print_string += (f"{basic_vars[i]}\t")
                for elem in value_grid[i]:
                    print_string += (f"{elem}\t")
                print_string += (f"{results_column[i]}\n")

            # Imprimir fila z
            print_string += (f"\tZ\t")
            for z in z_row:
                print_string += (f"{z}\t")

            suma_final = 0
            for i in range(0, len(results_column)):
                suma_final += (results_column[i]
                               * (variable_value_list[basic_vars[i]]))
            print_string += (f"{suma_final}\t")
            self.log(print_string)

        self.log("Tabla Inicial:")
        print_table()

        changed: bool = False
        # Iterate through the basic variable list to check changes in the Z row
        for z_row_index in range(len(z_row)):
            # The sum of the current column's values times the basic variables
            column_sum: Fraction = Fraction(0)

            for column_value in range(len(basic_vars)):
                column_sum += value_grid[column_value][z_row_index] * \
                    variable_value_list[basic_vars[column_value]]

            if column_sum != 0:
                z_row[z_row_index] -= column_sum
                changed = True

        if changed:
            self.log("Tabla Ajustada: ")
            print_table()

        iteration = 1

        while True:

            pivot_column = -1
            for i in range(0, len(z_row)):
                if z_row[i] <= 0:
                    continue

                if pivot_column == -1:
                    pivot_column = i
                    continue

                if z_row[i] > z_row[pivot_column]:
                    pivot_column = i

            if pivot_column == -1:
                self.log("\nLa solucion ha sido encontrada!")
                # imprimir_tabla()
                break

            self.log(f"\n--- Iteración {iteration} ---")

            self.log(f"Columna Pivote: {pivot_column}")

            pivot_row = -1

            # Construir lista de cocientes válidos (divisor > 0)
            valid_candidates = []  # lista de (fila_index, cociente)
            for i in range(0, len(results_column)):
                ratio_divisor = value_grid[i][pivot_column]
                # saltar divisores no positivos (<= 0)
                try:
                    if (ratio_divisor) <= 0:
                        continue
                    divison_result = (results_column[i]) / (ratio_divisor)
                    valid_candidates.append((i, divison_result))
                    self.log(
                        f"Resultado Fila #{i}: {results_column[i]} / {ratio_divisor} = {divison_result}")
                except ZeroDivisionError:
                    # por seguridad (aunque ya evitamos divisor == 0 con la comprobación anterior)
                    continue

            if len(valid_candidates) == 0:
                # Si no hay filas con divisor > 0 entonces el problema es ilimitado
                raise Exception(
                    "Error, no hay mas filas pivote viables (problema ilimitado).")

            # Encontrar el/los mínimos
            min_val = min(c[1] for c in valid_candidates)
            min_indexes = [idx for idx,
                           val in valid_candidates if val == min_val]

            if len(min_indexes) == 1:
                pivot_row = min_indexes[0]
            else:
                # Empate -> pedir al usuario que elija
                self.log("Empate detectado entre las filas pivote")
                self.log(
                    f"Indices candidatos: {min_indexes} con cociente = {min_val}")
                while True:
                    try:
                        choice = int(
                            self.input(f"Empate detectado, elige una fila pivote {min_indexes}: "))
                        if choice in min_indexes:
                            pivot_row = choice
                            break
                        else:
                            self.log("Fila no válida, intentá de nuevo.")
                    except ValueError:
                        self.log("Por favor, ingresá un número válido.")

            self.log(f"Fila Pivote: {pivot_row}")

            self.log(f"Variable entrante: {non_basic_vars[pivot_column]}")
            self.log(f"Variable saliente: {basic_vars[pivot_row]}")

            non_basic_vars[pivot_column], basic_vars[pivot_row] = basic_vars[pivot_row], non_basic_vars[pivot_column]

            self.log(f"Calculando...")

            # Copy to prevent contamination while operating
            old_value_grid = deepcopy(value_grid)
            old_z_row = z_row.copy()
            old_result_column = results_column.copy()

            pivot = (value_grid[pivot_row][pivot_column])
            self.log(f"Pivote: {pivot}")

            for nrow in range(0, len(value_grid)):
                for ncolumn in range(0, len(value_grid[nrow])):
                    if nrow == pivot_row and ncolumn == pivot_column:
                        value_grid[nrow][ncolumn] = self.invert_pivot(
                            old_value_grid[nrow][ncolumn])
                    elif nrow == pivot_row:
                        value_grid[nrow][ncolumn] = self.get_row_value(
                            old_value_grid[nrow][ncolumn], pivot)
                    elif ncolumn == pivot_column:
                        value_grid[nrow][ncolumn] = self.get_column_value(
                            old_value_grid[nrow][ncolumn], pivot)
                    else:
                        # self.log(
                        #     f"En {nfila}, {ncolumna} = {valores_viejos[nfila][ncolumna]}")
                        # self.log(
                        #     f"Se aplicara valor nuevo con {nfila},{columna_pivote} = {valores_viejos[nfila][columna_pivote]}")
                        # self.log(
                        #     f"y tambien: {fila_pivote},{ncolumna} = {valores_viejos[fila_pivote][ncolumna]}")

                        value_grid[nrow][ncolumn] = self.get_new_value(
                            old_value_grid[nrow][ncolumn], pivot, old_value_grid[nrow][pivot_column], old_value_grid[pivot_row][ncolumn])
                        # self.log(f"Resultado: {valores_basicas[nfila][ncolumna]}")

                        # self.log()
                        # self.log()

            for ncolumn in range(0, len(z_row)):
                if ncolumn == pivot_column:
                    z_row[ncolumn] = self.get_column_value(
                        old_z_row[ncolumn], pivot)
                else:
                    z_row[ncolumn] = self.get_new_value(
                        old_z_row[ncolumn], pivot, old_z_row[pivot_column], old_value_grid[pivot_row][ncolumn])

            for nrow in range(0, len(results_column)):
                if nrow == pivot_row:
                    results_column[nrow] = self.get_row_value(
                        old_result_column[nrow], pivot)
                else:
                    results_column[nrow] = self.get_new_value(
                        old_result_column[nrow], pivot, old_result_column[pivot_row], old_value_grid[nrow][pivot_column])

            self.log(f"Calculos, hechos.")
            self.log(f"Iteracion #{iteration}:")
            print_table()
            iteration += 1

        # Determine solution
        sol: dict[str, Fraction] = {}
        for i in range(len(results_column)):
            sol.update({basic_vars[i]: results_column[i]})

        final_sum = Fraction(0)
        for i in range(0, len(results_column)):
            final_sum += (results_column[i]
                          * (variable_value_list[basic_vars[i]]))

        return sol, final_sum

    def solve(self):
        if self.model.objective_function is None:
            raise ValueError(
                "An objective function is not yet set, model is not solvable.")

        # We need to get some information
        basic_variables: list[str] = []
        non_basic_variables: list[str] = []

        results = []
        zrow = []

        variables: dict[str, Fraction] = {}

        minimize: bool = self.model.objective_function[1] == ModelObjective.MINIMIZE
        obj_coefficient_list: list[Fraction] = self.model.objective_function[0]

        # Find the max coefficient for artificial variables
        max_coefficient: Fraction = obj_coefficient_list[0]
        # Get every variable from the objective function
        for i in range(0, len(obj_coefficient_list)):
            coefficient: Fraction = obj_coefficient_list[i]
            if abs(coefficient) > abs(max_coefficient):
                max_coefficient = abs(coefficient)

            # Adjusted coefficient for minimization
            adjusted_coefficient = coefficient * -1 if minimize else coefficient

            variables.update({x_index(i + 1): adjusted_coefficient})
            # Every variable starts as a non-basic
            non_basic_variables.append(x_index(i + 1))

        # Adjust max coefficient
        max_coefficient = max_coefficient * -1 if minimize else max_coefficient
        # Now figure out the variables we'll have to add as our basic ones (and non-basic)
        artificial_count = 0
        clearance_count = 0

        non_basic_clearances: list[int] = []

        value_grid: list[list[Fraction]] = []
        # S grid
        s_grid_cache: list[list[Fraction]] = []

        # Basic clearances must always be added after basic artificial variables
        basic_clearances = []

        for i, constraint in enumerate(self.model.constraints):
            # Add the coefficient list to the value grid
            value_grid.append(constraint[0])
            results.append(constraint[2])

            if constraint[1] == ConstraintType.LIMITATION:  # <=
                # Add a clearance
                clearance_count += 1
                # Clearances must always be added AFTER artificial variables
                # variables.update({s_index(clearance_count): Fraction(0)})
                # basic_variables.append(s_index(clearance_count))
                basic_clearances.append(clearance_count)
            elif constraint[1] == ConstraintType.EXACTITUDE:  # =
                # Add ONLY an artificial variable
                artificial_count += 1
                variables.update(
                    {a_index(artificial_count): max_coefficient * 10})
                basic_variables.append(a_index(artificial_count))
            elif constraint[1] == ConstraintType.REQUIREMENT:  # >=
                # Add a clearance (as non-basic)
                clearance_count += 1
                variables.update({s_index(clearance_count): Fraction(0)})
                non_basic_variables.append(s_index(clearance_count))
                non_basic_clearances.append(clearance_count)

                # Add an artificial variable
                artificial_count += 1
                variables.update(
                    {a_index(artificial_count): max_coefficient * 10})
                basic_variables.append(a_index(artificial_count))

        # Add non basic clearance columns
        for c_index in non_basic_clearances:
            for row in range(len(value_grid)):
                if row == c_index - 1:
                    value_grid[row].append(Fraction(-1))
                else:
                    value_grid[row].append(Fraction(0))

        for clearance in basic_clearances:
            variables.update({s_index(clearance): Fraction(0)})
            basic_variables.append(s_index(clearance))
        # Set up the zrow
        for var in non_basic_variables:
            zrow.append(variables[var])

        self.log("Basic variables: [" + ", ".join(basic_variables) + "]")
        self.log("Non-Basic variables: [" +
                 ", ".join(non_basic_variables) + "]")

        var_string = ""
        for varrow in value_grid:
            var_string += list((str(var) for var in varrow)).__str__() + "\n"
        self.log("Value grid: \n" + var_string)

        val_dict: dict[str, Fraction]
        opt_sol: Fraction

        val_dict, opt_sol = self._simplex_solver(
            non_basic_variables, basic_variables, value_grid, zrow, results, variables)

        var_val_list: list[Fraction] = []
        for i in range(len(obj_coefficient_list)):
            if x_index(i + 1) not in val_dict:
                var_val_list.append(Fraction(0))
            else:
                var_val_list.append(val_dict[x_index(i+1)])

        return var_val_list, opt_sol
