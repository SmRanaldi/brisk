import pandas as pd

import os
import json

from brisk import config_dir, out_dir, fs_marker, fs_imu
from brisk.analysis import segmentation
from brisk import analysis
from brisk.utils.cl import print_error, print_ongoing, print_warning
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
        self.average_data = {}
        self.cycle_events = {} # In seconds
        self.cycle_events_marker = {} # In seconds
        self.parameters = []
        self.age = None
        self.weight = None
        self.height = None

        self.imu_config = get_imu_config(self.archive_path)

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
        self.cycle_events = {t:
            segmentation.load_indexes(self.name, t)/fs_imu
            for t in self.trials
            if path.search_trial(self.name, t)
        }
        self.age, self.weight, self.height = path.get_anthropometrics(self.name)
        if any([x==None for x in [self.age, self.weight, self.age]]):
            print_warning('Anthropometrics not found.')


    # --- Get average profiles
    def get_average_profiles(self):
        if not self.average_data.keys():
            out_profiles = {t: 
                segmentation.get_average_profiles(
                    subject=self.name,
                    trial=t,
                    update=False
                )
                for t in self.trials
            }

            self.samples_per_cycle = out_profiles[self.trials[0]].shape[0]

            self.average_data = out_profiles

        else:

            out_profiles = self.average_data

        return out_profiles

    # --- Get cycle indexes from markers
    def get_marker_indexes(self):
        self.trials = path.get_trials(self.name)
        ctrl = True
        for t in self.trials:
            filename_evt = path.join_path([self.db_path, t, 'rawdata', 'events_marker.csv'])
            if os.path.exists(filename_evt):
                dd_evt = pd.read_csv(filename_evt)
                self.cycle_events_marker[t] = dd_evt.values[:,0]/fs_marker
            else:
                ctrl = False
        if not ctrl:
            print_error('Marker events not found.')


    # --- Dump absolute indexes to db/rawdata
    def dump_indexes(self):
        if not self.trials:
            self.import_data()

        for t in self.trials:
            with open(path.join_path([self.db_path, t, 'rawdata', 'events.json'])) as f:
                start_t = json.load(f)['events']['imu'][0]/fs_imu

            evt = pd.DataFrame(self.cycle_events[t] + start_t, columns=['events'])
            evt.to_csv(path.join_path([self.db_path, t, 'rawdata', 'events_absolute.csv']), index=None)

    # --- Update all data in the db
    def update(self):

        if path.search_subject(self.name):
            self.trials = path.get_trials(self.name)

        for t in self.trials:
            print_ongoing(f'\nUpdating subject {self.name}, trial {t}...')
            segmentation.update_indexes(self.name, t)
            segmentation._filter_data(self.name, t)
            segmentation._calculate_average(self.name, t)
        
        print_ongoing(f'\nUpdating subject {self.name}, parameters...')
        analysis.cycle_parameters(self.name, True)
        analysis.global_parameters(self.name, True)