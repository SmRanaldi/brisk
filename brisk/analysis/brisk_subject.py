import pandas as pd
import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt

import os
import json

from brisk import config_dir, out_dir, fs_marker, fs_imu, fs_emg
from brisk.analysis import segmentation, parameters, kinematics
from brisk.utils.cl import print_error, print_ongoing, print_warning
from brisk.data_importer.imu import get_imu_config, load_raw_imu
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
        self.raw_imu = {}
        self.raw_emg = {}
        self.raw_forces = {}
        self.raw_moments = {}
        self.raw_cop = {}
        self.phases = {}
        self.phases_limits = []
        self.phase_duration = {}
        self.segmented_data = {}
        self.average_data = {}
        self.cycle_events = {} # In seconds
        self.cycle_events_marker = {} # In seconds
        self.cycle_events_absolute = {} # In seconds
        self.parameters = []
        self.age = None
        self.weight = None
        self.height = None

        self.samples_per_cycle = 200
        self.segmentation_labels = ['trunk_acc_y', 'trunk_acc_z']

    # --- Conversion to string
    def __str__(self) -> str:
        return self.name.title()
    
    # *********** Raw data functions *************

    # --- Import trial names
    def get_trials(self):
        if not self.trials:
            self.trials = path.get_trials(self.name)
        return self.trials
    
    # --- Import IMU
    def get_raw_imu(self):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the archive.')
            return
        if not self.raw_imu.keys():
            self.raw_imu = {
                t: pd.read_csv(path.join_path([
                    self.db_path,
                    t,
                    'rawdata',
                    'imu.csv'
                ])).iloc[int(self.get_absolute_indexes()[t][0]*fs_imu):int(self.get_absolute_indexes()[t][-1]*fs_imu),:]
                for t in self.get_trials()
            }

        return self.raw_imu

    # --- Import EMG
    def get_raw_emg(self):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the archive.')
            return
        if not self.raw_emg.keys():
            self.raw_emg = {
                t: pd.read_csv(path.join_path([
                    self.db_path,
                    t,
                    'rawdata',
                    'emg.csv'
                ])).iloc[int(self.get_absolute_indexes()[t][0]*fs_emg):int(self.get_absolute_indexes()[t][-1]*fs_emg),:]
                for t in self.get_trials()
            }

        return self.raw_emg

    # --- Import forces
    def get_raw_forces(self):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the archive.')
            return
        if not self.raw_forces.keys():
            self.raw_forces = {
                t: pd.read_csv(path.join_path([
                    self.db_path,
                    t,
                    'rawdata',
                    'force.csv'
                ])).iloc[int(self.get_absolute_indexes()[t][0]*fs_marker):int(self.get_absolute_indexes()[t][-1]*fs_marker),:]
                for t in self.get_trials()
            }

        return self.raw_forces

    # --- Import moments
    def get_raw_moments(self):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the archive.')
            return
        if not self.raw_moments.keys():
            self.raw_moments = {
                t: pd.read_csv(path.join_path([
                    self.db_path,
                    t,
                    'rawdata',
                    'moment.csv'
                ])).iloc[int(self.get_absolute_indexes()[t][0]*fs_marker):int(self.get_absolute_indexes()[t][-1]*fs_marker),:]
                for t in self.get_trials()
            }

        return self.raw_moments

    # --- Import cop
    def get_raw_cop(self):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the archive.')
            return
        if not self.raw_cop.keys():
            self.raw_cop = {
                t: pd.read_csv(path.join_path([
                    self.db_path,
                    t,
                    'rawdata',
                    'cop.csv'
                ])).iloc[int(self.get_absolute_indexes()[t][0]*fs_marker):int(self.get_absolute_indexes()[t][-1]*fs_marker),:]
                for t in self.get_trials()
            }

        return self.raw_cop

    # --- Get absolute indexes
    def get_absolute_indexes(self):
        self.trials = path.get_trials(self.name)
        ctrl = True
        for t in self.trials:
            filename_evt = path.join_path([self.db_path, t, 'rawdata', 'events_absolute.csv'])
            if os.path.exists(filename_evt):
                dd_evt = pd.read_csv(filename_evt)
                self.cycle_events_absolute[t] = dd_evt.values[:,0]
            else:
                ctrl = False
        if not ctrl:
            print_error('Marker events not found.')

        return self.cycle_events_absolute
    
    # --- Get indexes
    def get_indexes(self):
        idx = self.get_absolute_indexes()
        indexes_out = {k: np.asarray(v-v[0], dtype='int') for k,v in idx.items()}
        return indexes_out

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

        return self.cycle_events_marker

    # *********** Import functions *************

    # --- Import all available data
    def import_data(self):

        self.get_raw_imu()

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

    # *********** DB functions *************

    # --- Dump absolute indexes to db/rawdata
    def dump_indexes(self):
        if not self.trials.keys():
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
        parameters.cycle_parameters(self.name, True)
        parameters.global_parameters(self.name, True)
        # self.import_data()
        self.dump_indexes()


    # *********** Analysis functions *************

    # --- Get zones from trunk data
    def get_limits(self):
        if not np.asarray(self.phases_limits).size:
            self.phases_limits = kinematics.get_zones(self.get_raw_imu(), labels=self.segmentation_labels)
        return self.phases_limits

    # --- Get phase indexes and durations
    def get_zones(self):
        self.phase_duration = {}
        self.phases = {}
        for k, v in self.get_raw_imu().items():
            pd_temp, p_temp = kinematics.phase_count(data_in_raw=v[self.segmentation_labels].values, limits=self.get_limits())
            self.phase_duration[k] = pd_temp
            self.phases[k] = p_temp
        return self.phases

    def get_phase_duration(self):
        if not self.phase_duration.keys():
            self.get_zones()
        return self.phase_duration
    
    # --- Fit to phases
    def fit_to_phases(self, data_in):
        if not self.phases.keys():
            self.get_zones()
        data = {k: sgn.resample(v, self.phases[k].size) for k, v in data_in.items()}
        out = {k: kinematics.average_by_phase(v, self.phases[k]) for k, v in data.items()}
        return out

    # --- Plot fitted data
    def plot_to_phases(self, data_in):
        if not self.phases.keys():
            self.get_zones()
        matrices_in = self.fit_to_phases(data_in)
        matrices_tot = np.asarray([v for v in matrices_in.values()])
        min_value, max_value = np.min(matrices_tot), np.max(matrices_tot)
        ax_titles = list(matrices_in.keys())
        fig, ax = plt.subplots(int(len(data_in.keys())/2), 2, figsize=(16,16), facecolor='w')
        for i, v in enumerate(matrices_in.values()):
            row = i%2
            col = int(i/2)
            kinematics.plot_phases(v, ax=ax[row,col], vmin=min_value, vmax=max_value)
            ax[row,col].set_title(
                ax_titles[i].replace('_', ' ').title(),
                fontsize=16
            )
        plt.show(fig)
        return matrices_in
