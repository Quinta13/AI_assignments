"""
This script exports all notebooks contained in a certain directory to a .pdf
"""

import os
import subprocess

from loguru import logger

from assignment_3.clustering.settings import get_root_dir

EXT = '.ipynb'
COMMAND = 'jupyter nbconvert --to webpdf'


def main():

    # https://stackoverflow.com/questions/18394147/how-to-do-a-recursive-sub-folder-search-and-return-files-in-a-list
    notebooks = [os.path.join(dp, f) for dp, dn, filenames in os.walk(get_root_dir())
                 for f in filenames if os.path.splitext(f)[1] == EXT]

    # subprocess.run("pip install nbconvert[webpdf]")  need to install the pacakge

    for notebook in notebooks:
        command = f"{COMMAND} {notebook}"
        logger.info(command)
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()