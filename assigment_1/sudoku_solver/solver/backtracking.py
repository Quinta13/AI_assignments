"""
This script ...
"""

import json
import os
from statistics import mean

from assigment_1.sudoku_solver.settings import OUT, IN
from assigment_1.sudoku_solver.solvers import Backtracking
from assigment_1.sudoku_solver.sudoku import Sudoku

# I/O settings
OUT_DIR_NAME = "backtracking"
OUT_FILE_NAME = "_backtracking.txt"
OUT_INFO_FILE = "execution_info.json"

OUT_DIR = os.path.join(OUT, OUT_DIR_NAME)

try:
    print(f"Creating {OUT_DIR}")
    os.makedirs(OUT_DIR)
except FileExistsError:
    print(f"{OUT_DIR} already exists")

files = os.listdir(IN)  # input sudoku in .txt file
files.sort()

info = {}  # dictionary for execution info to store as a .json

print("")
for file in files:
    file_no_ext = ''.join(file.split(".")[:-1])  # removing extension to file

    print(f"Processing {file_no_ext}")

    file_in = os.path.join(IN, file)
    file_out = os.path.join(OUT_DIR, file_no_ext + OUT_FILE_NAME)

    s = Sudoku(in_file=file_in)

    # stats
    empties = len(s.empty_cells)
    filled = len(s.non_empty_cells)
    mean_moves = round(mean([len(ass) for _, ass in s.feasible_assignments]), 3)

    # solve
    solver = Backtracking(s=s)
    d = solver.solve()

    # saving stats
    info[file_no_ext] = {
        'time': d['elapsed'], 'empty_cells': empties, 'filled_cells': filled,
        "mean_feasible_moves": mean_moves, 'expanded': d['expanded'], 'backtracked': d['backtracked']
    }

    solver.save(file_out=file_out)

OUT_INFO = os.path.join(OUT_DIR, OUT_INFO_FILE)
json.dump(info, open(OUT_INFO, "w"))
