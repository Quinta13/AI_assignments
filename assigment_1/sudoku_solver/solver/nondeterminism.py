"""
This script ...
"""

import json
import os

from assigment_1.sudoku_solver.sudoku import Sudoku
from assigment_1.sudoku_solver.solvers import Backtracking
from assigment_1.sudoku_solver.settings import IN, OUT

# I/O settings
OUT_DIR_NAME = "backtracking"
OUT_INFO_FILE = "non_determinism.json"

OUT_DIR = os.path.join(OUT, OUT_DIR_NAME)

N_ITER = 10

FILE = 'sudoku6.txt'

exec_time = {}

file_in = os.path.join(IN, FILE)
s = Sudoku(in_file=file_in)

for i in range(N_ITER):
    print(i)
    solver = Backtracking(s=s)
    d = solver.solve()
    exec_time[i] = d['elapsed']

OUT_INFO = os.path.join(OUT_DIR, OUT_INFO_FILE)
json.dump(exec_time, open(OUT_INFO, "w"))
