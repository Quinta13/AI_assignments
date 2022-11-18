from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from math import exp
from typing import Any, Dict, Tuple

from assigment_1.sudoku_solver.sudoku import Sudoku


class Solver(ABC):

    def __init__(self, s: Sudoku):
        """

        :param s: Sudoku to solve
        """

        self._s1 = s
        self._s1.reset()

        self._s2 = self._s1.__copy__()

    def __str__(self) -> str:
        """
        String representation comparing initial and solved
        :return: string representation of solution specifying unsatisfied conditions
        """

        arrow = " +----------|> "
        empty = " " * len(arrow)

        str1s = str(self._s1).split("\n")
        str2s = str(self._s2).split("\n")

        mid = len(str1s) // 2 - 1

        str_out = ""

        for i, strs in enumerate(zip(str1s, str2s)):
            str1, str2 = strs
            sep = arrow if i == mid else empty
            str_out += str1 + sep + str2 + "\n"

        # Result check
        str_out += "\n" + self._s2.global_check()

        return str_out

    @abstractmethod
    def solve(self) -> Any:
        """ Method to solve sudoku"""
        pass

    def save(self, file_out: str):
        """
        Store string comparison to a given file
        :param file_out: file to store comparison
        """

        open(file_out, 'w', encoding="utf-8").write(str(self))


class Backtracking(Solver, ABC):

    def solve(self) -> Dict:
        """
        Solve Sudoku using backtracking and constraint propagation
        Use a dictionary to save computation information
        :return: computation information as a dictionary
        """
        d = {'expanded': 0, 'backtracked': 0}
        st = time.time()
        self.solve_rec(d=d)
        et = time.time()
        d['elapsed'] = et - st
        return d

    def solve_rec(self, d: Dict) -> bool:
        """
        Recursive function to solve sudoku:
        - returns True if sudoku is complete
        - otherwise tries all possible values of the best assignment
            - return True if a value bring to a winning state
            - otherwise returns False
        :return: True if sudoku is complete, False otherwise
        """
        if self._s2.win:
            return True

        cell, values = self._s2.best_assignment
        row, col = cell
        for value in values:
            d['expanded'] += 1
            self._s2.set_cell(row=row, col=col, cell=value)
            if self.solve_rec(d=d):
                return True
            d['backtracked'] += 1
            self._s2.undo(row, col)
        return False


class HillClimbing(Solver, ABC):

    def solve(self) -> Tuple[int, int]:
        """
        Optimization process to minimize the number of repetitions over columns and rows
        :return: number of swaps, best value of objective function reached
        """

        self._s2.fill_blocks()

        num_swaps = 0

        while True:
            current = self._s2.repetitions
            best_move = self._s2.best_neighbor()
            swap, value = best_move
            if value == current or current == 0:
                return num_swaps, current
            index1, index2 = swap
            self._s2.swap(index1=index1, index2=index2)
            num_swaps += 1


class SimulatedAnnealing(Solver, ABC):

    def solve(self) -> Dict:
        """
        Optimization process to minimize the number of repetitions over columns and rows
            introducing a random decreasing temperature
        :return: best value of objective function reached
        """

        self._s2.fill_blocks()

        trend = {}  # dictionary to map the trend

        count = 0

        t: float = 0.5 + 0.001 * len(self._s2.get_swaps())  # starting temperature
        multiplier: float = 0.999995  # multiplier for temperature
        max_iteration: int = 300000  # maximum number of iteration admitted
        debug: int = 10  # index to print state

        while count < max_iteration:

            actual_score = self._s2.repetitions

            trend[count] = actual_score

            if count % debug == 0:
                print(f"[{count}] \tT = {t},  \tscore = {actual_score}")

            if actual_score == 0:
                return trend

            moves = self._s2.neighborhoods()
            random_move = random.choice(moves)

            swap, new_score = random_move
            index1, index2 = swap

            de = new_score - actual_score

            if de < 0:
                self._s2.swap(index1=index1, index2=index2)

            elif random.uniform(0, 1) < exp(- de / t):
                self._s2.swap(index1=index1, index2=index2)

            t *= multiplier
            count += 1

        return trend
