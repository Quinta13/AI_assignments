"""
This script ...
"""

import json
import os

from assigment_1.sudoku_solver.sudoku import Sudoku
from assigment_1.sudoku_solver.solvers import HillClimbing
from assigment_1.sudoku_solver.settings import IN, OUT

# I/O settings
OUT_DIR_NAME = "hill_climbing"
OUT_FILE_NAME = "_hill_climbing.txt"
OUT_INFO_FILE = "execution_info.json"

OUT_DIR = os.path.join(OUT, OUT_DIR_NAME)

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

    s = Sudoku(in_file=file_in)

    # stats
    empties = len(s.empty_cells)
    filled = len(s.non_empty_cells)
    swaps = len(s.get_swaps())

    # solve
    solver = HillClimbing(s=s)
    n_swap, rep = solver.solve()

    # saving stats
    info[file_no_ext] = {
        'empty_cells': empties, 'filled_cells': filled, 'n_swap': n_swap,
        "possible_swaps": swaps, 'final_repetitions': rep, 'solved': rep == 0
    }

    solver.save(file_out=file_out)

OUT_INFO = os.path.join(OUT_DIR, OUT_INFO_FILE)
json.dump(info, open(OUT_INFO, "w"))
