from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import List, Tuple, Set, Any, Iterator

import numpy as np


class Cell(Enum):
    """
    Cell values domain
    """
    Empty = '.'
    One = '1'
    Two = '2'
    Three = '3'
    Four = '4'
    Five = '5'
    Six = '6'
    Seven = '7'
    Eight = '8'
    Nine = '9'

    def __str__(self):
        return self.value


class Sudoku:
    # SQUARE SUDOKU DIMENSION

    SQUARE = 9
    CELL = 3

    # CONSTRUCTOR

    def __init__(self, in_file: str):
        """

        param in_file: file containing sudoku to solve a text of nine lines of nine digits (spaces are stripped)
            - each number is a filled cell
            - any other value is an empty cell
            raises an error if the configuration is not consistent
        """

        # initializing empty board
        fixed = np.array([[False] * self.SQUARE] * self.SQUARE)
        board = np.array([[Cell.Empty] * self.SQUARE] * self.SQUARE)

        self.fixed: np.array = fixed
        self.board: np.array = board  # board with values, digit or empty
        self.in_file = in_file

        init_state = open(in_file).read()  # read input file
        init_state = init_state.replace(' ', '')  # removing spaces
        rows = init_state.split('\n')  # splitting rows

        # rows number check
        if len(rows) != self.SQUARE:
            raise Exception(f"Got {len(rows)} rows, expected {self.SQUARE}")

        for row_ind, row in enumerate(rows):
            # columns number check
            if len(row) != self.SQUARE:
                raise Exception(f"Got {len(row)} columns in row {row_ind}, expected {self.SQUARE}")
            for col_ind, val in enumerate(row):
                # digit if given, empty otherwise
                if val in [str(i) for i in range(1, self.SQUARE + 1)]:  # digit
                    self.board[row_ind, col_ind] = Cell(val)
                    self.fixed[row_ind, col_ind] = True

        # configuration check
        check = self.global_check()
        if check != "":
            print(check)
            raise Exception("No valid configuration")

    # CLASS METHODS

    def __copy__(self) -> Sudoku:
        """
        Return a new instance of the sudoku
        :return: new sudoku
        """

        new_s = Sudoku(self.in_file)

        # list of non-fixed cells that are filled
        to_fill = list(set(self.empty_cells).difference(new_s.empty_cells))

        for row, col in to_fill:
            new_s.set_cell(row=row, col=col, cell=self.get_cell(row=row, col=col))

        return new_s

    def __str__(self) -> str:
        """
        TODO change grid length in function of board length
        :return: board representation in ASCII-art
        """
        str_out: str = ""
        str_out += "╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗\n"
        for i in range(self.SQUARE):
            str_out += "║"
            for j in range(self.SQUARE):
                if j != 0 and j % self.CELL == 0:
                    str_out = str_out[:-1] + "║"
                str_out += f" {self.get_cell(row=i, col=j)} |"
            str_out = str_out[:-1] + f"║\n"
            if i + 1 != self.SQUARE and (i + 1) % self.CELL == 0:
                str_out += '╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣\n'
            elif i + 1 != self.SQUARE:
                str_out += "╟───┼───┼───╫───┼───┼───╫───┼───┼───╢\n"
        str_out += "╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝\n"
        return str_out

    # GETTER

    @property
    def empty_cells(self) -> List[Tuple[int, int]]:
        """
        Return all empty cells of the board
        :return: list of couples row-column of empty cells
        """
        return [(i, j)
                for i in range(self.SQUARE)
                for j in range(self.SQUARE)
                if self.get_cell(row=i, col=j) == Cell.Empty]

    @property
    def non_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Return all non-empty cells of the board
        :return: list of couples row-column of non-empty cells
        """
        return [(i, j)
                for i in range(self.SQUARE)
                for j in range(self.SQUARE)
                if self.get_cell(row=i, col=j) != Cell.Empty]

    def get_cell(self, row: int, col: int) -> Cell:
        """
        Return cell value
        :param row: row index
        :param col: column index
        :return: cell value in given row and column
        """
        self._valid_row(row=row)
        self._valid_col(col=col)
        return self.board[row][col]

    def get_row(self, row: int) -> List[Cell]:
        """
        Returns all non-empty values in a given rows (including duplicates)
        :param row: row index
        :return: values in given row
        """
        self._valid_row(row=row)
        return [c for c in self.board[row, :] if c != Cell.Empty]

    def get_col(self, col: int) -> List[Cell]:
        """
        Returns all non-empty values in a given column (including duplicates)
        :param col: column index
        :return: values in given column
        """
        self._valid_col(col=col)
        return [c for c in self.board[:, col] if c != Cell.Empty]

    def _block_iterator(self, block_row: int, block_col: int) -> Iterator[Tuple[int, int]]:
        """
        Returns indexes iterator to iterate over a block
        :param block_row: block row index
        :param block_col: column row index
        :return: block indexes iterator
        """
        self._valid_block_row(block_row=block_row)
        self._valid_col_block(block_col=block_col)
        starting_row = block_row * self.CELL
        starting_col = block_col * self.CELL
        indexes = []
        for row in range(self.CELL):
            for col in range(self.CELL):
                indexes.append((starting_row + row, starting_col + col))
        return iter(indexes)

    def get_block(self, block_row: int, block_col: int) -> List[Cell]:
        """
        Returns all non-empty values in a given block (including duplicates)
        :param block_row: block row index
        :param block_col: column row index
        :return: values in given block
        """
        cells: List[Cell] = []
        indexes = self._block_iterator(block_row=block_row, block_col=block_col)
        for index in indexes:
            row, col = index
            cells.append(self.get_cell(row=row, col=col))
        return [b for b in cells if b != Cell.Empty]

    def _get_belonging_block(self, row: int, col: int) -> Tuple[int, int]:
        """
        Given row and column of a cell returns belonging block
        :param row: row index
        :param col: column index
        :return: block-row and bock-column indexes
        """
        self._valid_row(row=row)
        self._valid_col(col=col)
        return row // self.CELL, col // self.CELL

    def _block_missing_values(self, block_row: int, block_col: int) -> List[Cell]:
        """
        Return missing values in a block
        :param block_row: block row index
        :param block_col: block column index
        :return: list of missing values
        """

        return list(self._all_values.difference(self.get_block(block_row=block_row, block_col=block_col)))

    # CELL MANIPULATION

    def reset(self):
        """ Reset sudoku to starting state """
        for row in range(self.SQUARE):
            for col in range(self.SQUARE):
                if not self.fixed[row][col] and \
                        self.get_cell(row=row, col=col) != Cell.Empty:
                    self.undo(row=row, col=col)

    def set_cell(self, row: int, col: int, cell: Cell):
        """
        Set a non-empty value to a cell, raise an error:
            if value is empty one
            if cell is no-empty
        :param row: row index
        :param col: column index
        :param cell: non-empty cell value
        """
        value = self.get_cell(row=row, col=col)
        err_str = f"Trying to set {cell} to {row}, {col} "
        if self.fixed[row][col]:
            raise Exception(err_str + "but the cell is fixed")
        if value != Cell.Empty:
            raise Exception(err_str + f"but it already filled with {value}")
        if cell is Cell.Empty:
            raise Exception(err_str + f"which is already empty")
        self.board[row][col] = cell

    def undo(self, row: int, col: int):
        """
        Set a non-empty cell to empty if not fixed value, raise an error
            cell is empty
            cell is fixed
        :param row: row index
        :param col: column index
        """
        value = self.get_cell(row=row, col=col)
        if value == Cell.Empty:
            raise Exception(f"Trying to undo {row}, {col} which is already empty")
        if self.fixed[row][col]:
            raise Exception(f"Cannot undo fixed value {value} in {row}, {col}")

        self.board[row][col] = Cell.Empty

    # VALUES HANDLING

    @property
    def _all_values(self) -> Set[Cell]:
        """
        Util function returning all possible cell values excluding empty one
        :return: set of values for non-empty cells
        """
        return set([Cell(str(cell + 1)) for cell in range(self.SQUARE)])

    def _feasible_values(self, row: int, col: int) -> List[Cell]:
        """
        Returns feasible value for a cell
        :param row: row index
        :param col: column index
        :return: feasible values for the given cell
        """

        if self.get_cell(row=row, col=col) != Cell.Empty:
            raise Exception(f"Cell ({row}, {col}) is not empty ")

        block_row, block_col = self._get_belonging_block(row=row, col=col)

        # constraint
        rows_values = self.get_row(row=row)
        cols_values = self.get_col(col=col)
        blocks_values = self.get_block(block_row=block_row, block_col=block_col)

        # filtering
        values = set(rows_values).union(cols_values).union(blocks_values)
        feasible = list(self._all_values.difference(values))

        return feasible

    @property
    def best_assignment(self) -> Tuple[Tuple[int, int], List[Cell]]:
        """
        Return best next assignment (with least possible feasible values)
        :return: row and column index and the list of feasible values assignment
        """
        return next(iter(self.feasible_assignments))

    @property
    def feasible_assignments(self) -> List[Tuple[Tuple[int, int], List[Cell]]]:
        """
        Foreach empty cell return all feasible assignments
        :return: list of cells as row and column index and their feasible values
        """
        empties = self.empty_cells
        assignments = [((r, c), self._feasible_values(row=r, col=c)) for r, c in empties]
        assignments = sorted(assignments, key=lambda x: len(x[1]))
        return assignments

    # INTEGRITY CHECK

    def _rows_check(self) -> str:
        """
        Check if constraints are satisfied for each row
        :return: inconsistent values as string; empty if no errors
        """
        errors = ""
        for i in range(self.SQUARE):
            values = self.get_row(row=i)
            if not self._uniqueness(values):
                errors += f"Multiple value in row {i}: {[str(s) for s in self._repeated_values(values)]}\n"
        return errors

    def _cols_check(self) -> str:
        """
        Check if constraints are satisfied for each column
        :return: inconsistent values as string; empty if no errors
        """
        errors = ""
        for j in range(self.SQUARE):
            values = self.get_col(col=j)
            if not self._uniqueness(values):
                errors += f"Multiple value in column {j}: {[str(s) for s in self._repeated_values(values)]}\n"
        return errors

    def _blocks_check(self) -> str:
        """
        Check if constraints are satisfied for each block
        :return: inconsistent values as string; empty if no errors
        """
        errors = ""
        for i in range(self.CELL):
            for j in range(self.CELL):
                values = self.get_block(block_row=i, block_col=j)
                if not self._uniqueness(values):
                    errors += f"Multiple value in block {i}, {j}: {[str(s) for s in self._repeated_values(values)]}\n"
        return errors

    def global_check(self) -> str:
        """
        Check if constraints are satisfied for each row, column and block
        prints errors if present
        :return: true if no error, false otherwise
        """
        r_errs = self._rows_check()
        c_errs = self._cols_check()
        b_errs = self._blocks_check()
        tot_errs = r_errs + c_errs + b_errs

        return tot_errs

    # WIN

    @property
    def win(self) -> bool:
        """
        Check win-game conditions
        :return: true if game is finished, false otherwise
        """
        return len(self.empty_cells) == 0 and self.global_check() == ""

    # INDEX VALIDATION

    def _valid_row(self, row: int):
        """
        Check if row index is a feasible value
        :param row: row index
        :return: row is in valid range
        """
        if 0 <= row < self.SQUARE:
            return
        raise Exception(f"Row must be between {0} and {self.SQUARE - 1}, got {row} instead")

    def _valid_col(self, col: int):
        """
        Check if column index is a feasible value
        :param col: column index
        :return: column is in valid range
        """
        if 0 <= col < self.SQUARE:
            return
        raise Exception(f"Column must be between {0} and {self.SQUARE - 1}, got {col} instead")

    def _valid_block_row(self, block_row: int):
        """
        Check if block-row index is a feasible value
        :param block_row: block-row index
        :return: block-row is in valid range
        """
        if 0 <= block_row < self.CELL:
            return
        raise Exception(f"Block row must be between {0} and {self.CELL - 1}, got {block_row} instead")

    def _valid_col_block(self, block_col: int):
        """
        Check if block-column index is a feasible value
        :param block_col: block-row index
        :return: block-col is in valid range
        """
        if 0 <= block_col < self.CELL:
            return
        raise Exception(f"Block column must be between {0} and {self.CELL - 1}, got {block_col} instead")

    # OTHERS

    @staticmethod
    def _repeated_values(values: List[Any]) -> List[Any]:
        """
        Util function to get repeated values in a list
        :param values: list of values
        :return: list of duplicates, empty list if no one
        """
        seen = set()
        dupes = [x for x in values if x in seen or seen.add(x)]
        return dupes

    @staticmethod
    def _uniqueness(values: List[Any]) -> bool:
        """
        Util function to ensure there's no duplicate in a list
        :param values: list of values
        :return: true if no duplicates, false otherwise
        """
        return len(Sudoku._repeated_values(values=values)) == 0

    def save(self, file_out: str):
        """
        Store sudoku state to file
        :param file_out: path to file
        """
        open(file_out, 'w').write(str(self))

    # SIMULATED ANNEALING FUNCTIONS

    def _fill_block(self, block_row: int, block_col: int):
        """
        Fill empty cells in a block with its missing values (don't necessary respect constraints)
        :param block_row: block row index
        :param block_col: block column index
        """

        indexes = self._block_iterator(block_row=block_row, block_col=block_col)
        missing = iter(self._block_missing_values(block_row=block_row, block_col=block_col))

        for index in indexes:
            row, col = index
            cell = self.get_cell(row=row, col=col)
            if cell == Cell.Empty:
                self.set_cell(row=row, col=col, cell=next(missing))

    def fill_blocks(self):
        """
        Once no-initial position are set to empty,
            fill each block with missing values (don't necessary respect constraints)
        """

        self.reset()
        for block_row in range(self.CELL):
            for block_col in range(self.CELL):
                self._fill_block(block_row=block_row, block_col=block_col)

        self._blocks_check()

    def _get_block_swaps(self, block_row: int, block_col: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Return all possible no-fixed cell swaps in a block
        :param block_row: block row index
        :param block_col: block column index
        :return: list of swaps as a couple of indexes
        """
        indexes1 = self._block_iterator(block_row=block_row, block_col=block_col)
        swaps = []
        for index1 in indexes1:
            row1, col1 = index1
            indexes2 = self._block_iterator(block_row=block_row, block_col=block_col)
            for index2 in indexes2:
                row2, col2 = index2
                if not (self.fixed[row1][col1] or self.fixed[row2][col2]):
                    swaps.append(((row1, col1), (row2, col2)))
        swaps = [(i1, i2) for i1, i2 in swaps if i1[0] * self.SQUARE + i1[1] < i2[0] * self.SQUARE + i2[1]]
        return swaps

    def get_swaps(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Return all possible no-fixed cell swaps per block
        :return: list of swaps as a couple of indexes
        """
        block_swaps = [self._get_block_swaps(block_row=block_row, block_col=block_col)
                       for block_row in range(self.CELL) for block_col in range(self.CELL)]
        swaps = [item for sublist in block_swaps for item in sublist]
        return swaps

    def _row_repetitions(self, row: int) -> int:
        """
        Return number of repetitions over a given row
        :param row: row index
        :return: repeated values in a row
        """
        all_values = Counter(list(self._all_values))
        existing_values = Counter(self.get_row(row=row))
        diff = list((existing_values - all_values).elements())
        return len(diff)

    def _col_repetitions(self, col: int) -> int:
        """
        Return number of repetitions over a given column
        :param col: column index
        :return: repeated values in a column
        """
        all_values = Counter(list(self._all_values))
        existing_values = Counter(self.get_col(col=col))
        diff = list((existing_values - all_values).elements())
        return len(diff)

    @property
    def repetitions(self) -> int:
        """
        Objective function for simulated annealing
        Return total number of repetitions over all rows and columns
        :return:
        """
        col_rep = [self._row_repetitions(row=row) for row in range(self.SQUARE)]
        row_rep = [self._col_repetitions(col=col) for col in range(self.SQUARE)]
        return sum(col_rep) + sum(row_rep)

    def swap(self, index1: Tuple[int, int], index2: Tuple[int, int]):
        """
        Swap values of two cells if they are not fixed, and they belong to the same block
        :param index1: first cell row and column index
        :param index2: second cell row and column index
        """
        row1, col1 = index1
        row2, col2 = index2

        self._valid_row(row=row1)
        self._valid_row(row=row2)
        self._valid_col(col=col1)
        self._valid_col(col=col2)

        if self.fixed[row1][col1]:
            raise Exception(f"({row1}, {col1}) is fixed")

        if self.fixed[row2][col2]:
            raise Exception(f"({row2}, {col2}) is fixed")

        if self._get_belonging_block(row=row1, col=col1) != self._get_belonging_block(row=row2, col=col2):
            raise Exception(f"({row1}, {col1}) and ({row2}, {col2}) do not belong to same block")

        self.board[row1][col1], self.board[row2][col2] =\
            self.board[row2][col2], self.board[row1][col1]

    def neighborhoods(self) -> List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]]:
        """
        Produces a list of swaps and objective function they produce
        :return: list of swap associated with repetition number
        """
        neighbors = []
        swaps = self.get_swaps()

        for swap in swaps:
            index1, index2 = swap
            self.swap(index1=index1, index2=index2)
            rep = self.repetitions
            self.swap(index1=index1, index2=index2)
            neighbors.append((swap, rep))

        neighbors = sorted(neighbors, key=lambda x: x[1])

        return neighbors

    def best_neighbor(self) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]:
        """
        Returns the swap that with minimum value of the objective function
        :return: swap move and objective function value of best move
        """
        return next(iter(self.neighborhoods()))
