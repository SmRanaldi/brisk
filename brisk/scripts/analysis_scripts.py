import numpy as np
import argparse
from termcolor import colored

import os
import sys

from brisk.analysis import segmentation
from brisk.utils import path
from brisk import out_dir
from brisk.utils.cl import *

def run_subject():
    parser = argparse.ArgumentParser(description='Run the analysis on a subject')
    parser.add_argument('name')
    parser.add_argument('-u', '--update', default=False, help='Update the analysis on the subject', action='store_true')
    args = parser.parse_args()

    subject = args.name
    update = False
    if args.update: update=True

    trials_list = path.get_trials(subject)

    for t in trials_list:
        print(f'Trial {t}:\n')
        print_ongoing('Extracting indexes and average profiles...\n')
        segmentation.get_average_profiles(subject, t, update)
        print_success('Done.')

    print()
