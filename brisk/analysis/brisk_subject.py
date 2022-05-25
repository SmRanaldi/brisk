import pandas as pd

import os
import json

from brisk import config_dir, out_dir
from brisk.analysis import segmentation
from brisk.utils.cl import print_error, print_warning
from brisk.data_importer.imu import get_imu_config, load_raw_data
from brisk.utils import path

# Class for the brisk subject
class BriskSubject():

    '''
    Class representing one subject from the BRISK database. This class implements all the methods for loading and analysing single-subject recordings
    '''

    # --- Initialization
    def __init__(self, name) -> None:
        
        self.name = name.lower()
        self.archive_path = path.join_path([out_dir, '_archive', name])
        self.db_path = path.join_path([out_dir, name])
        self.trials = []
        self.raw_data = {}
        self.segmented_data = {}
        self.cycle_indexes = {}
        self.parameters = []
        self.age = None
        self.weight = None
        self.height = None

        self.imu_config = get_imu_config(self.archive_path)

        self.fs = 102.4
        self.samples_per_cycle = 200


    # --- Conversion to string
    def __str__(self) -> str:
        return self.name.title()
    
    
    # --- Import only data from the archive
    def import_from_archive(self):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the archive.')
            return

        self.raw_data = load_raw_data(base_dir=self.archive_path)


    # --- Import all available data
    def import_data(self):

        self.import_from_archive()

        if path.search_subject(self.name):
            self.trials = path.get_trials(self.name)

        self.segmented_data = {t:
            segmentation.get_filtered_data(self.name, t)
            for t in self.trials
            if path.search_trial(self.name, t)
        }
        self.cycle_indexes = {t:
            segmentation.load_indexes(self.name, t)
            for t in self.trials
            if path.search_trial(self.name, t)
        }
        self.age, self.weight, self.height = path.get_anthropometrics(self.name)
        if any([x==None for x in [self.age, self.weight, self.age]]):
            print_warning('Anthropometrics not found.')


    # --- Get average profiles
    def get_average_profiles(self):
        out_profiles = {t: 
            segmentation.get_average_profiles(
                subject=self.name,
                trial=t,
                update=False
            )
            for t in self.trials
        }

        self.samples_per_cycle = out_profiles[self.trials[0]].shape[0]

        return out_profiles