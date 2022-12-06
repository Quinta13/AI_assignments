"""
This script ...
"""

import json
import os

from matplotlib import pyplot as plt

from assignment_1.sudoku_solver.sudoku import Sudoku
from assignment_1.sudoku_solver.solvers import SimulatedAnnealing
from assignment_1.sudoku_solver.settings import IN, OUT


def save_as_plot(d: dict, name: str):

    x = list(d.keys())
    y = list(d.values())

    plt.clf()
    plt.plot(x, y, marker='.', linestyle='--', color='r')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('Simulate annealing')
    plt.savefig(name)


# I/O settings
OUT_DIR_NAME = "simulated_annealing"
OUT_FILE_NAME = "_simulated_annealing.txt"
OUT_IMG_FILE = "_simulated_annealing.png"
OUT_INFO_FILE = "execution_info.json"

OUT_DIR = os.path.join(OUT, OUT_DIR_NAME)


def simulated_annealing():

    try:
        print(f"Creating {OUT_DIR} already exists")
        os.makedirs(OUT_DIR)
    except FileExistsError:
        print(f"{OUT_DIR} already exists")

    files = os.listdir(IN)  # input sudoku in .txt file
    files.sort()

    info = {}  # dictionary for execution info to store as a .json

    for file in files:
        file_no_ext = ''.join(file.split(".")[:-1])  # removing extension to file

        print(f"Processing {file_no_ext}")

        file_in = os.path.join(IN, file)
        file_out = os.path.join(OUT_DIR, file_no_ext + OUT_FILE_NAME)
        img_out = os.path.join(OUT_DIR, file_no_ext + OUT_IMG_FILE)

        s = Sudoku(in_file=file_in)

        # stats
        empties = len(s.empty_cells)
        filled = len(s.non_empty_cells)
        swaps = len(s.get_swaps())
        start_t = 0.5 + 0.001 * len(s.get_swaps())

        # solve
        solver = SimulatedAnnealing(s=s)
        trend = solver.solve()

        tot_iter = list(trend.keys())[-1]
        final_repetitions = list(trend.values())[-1]
        convergence = final_repetitions == 0

        # saving stats
        info[file_no_ext] = {
            'empty_cells': empties, 'filled_cells': filled,
            "possible_swaps": swaps, 'total_iter': tot_iter,
            "starting_temperature": start_t,
            'converged': convergence, 'final_repetitions': final_repetitions
        }

        save_as_plot(d=trend, name=img_out)

        solver.save(file_out=file_out)


    OUT_INFO = os.path.join(OUT_DIR, OUT_INFO_FILE)
    json.dump(info, open(OUT_INFO, "w"))


if __name__ == "__main__":
    simulated_annealing()
