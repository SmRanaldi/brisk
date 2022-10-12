import pandas as pd
import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt

import os
import json
import itertools

from brisk import config_dir, out_dir, fs_marker, fs_imu, fs_emg
from brisk.analysis import segmentation, parameters, kinematics, emg, rqa
from brisk.analysis.synergies import VAF, extract_synergies, nnr, sort_W
from brisk.utils.cl import print_error, print_ongoing, print_success, print_warning
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
        self.W_tot = {}
        self.H_tot = {}
        self.W = {}
        self.H = {}
        self.VAF_extraction = {}
        self.W_template = []
        self.H_reconstruction = {}
        self.VAF_reconstruction = {}
        self.VAF_curve = {}
        self.VAF_muscles = {}
        self.n_syn = None
        self.age = None
        self.weight = None
        self.height = None

        self.samples_per_cycle = 200
        self.segmentation_labels = ['trunk_acc_y', 'trunk_acc_z']
        self.trials_order = ['full','half_robot_touch','half_robot','full_robot']

        self.set_muscles('all')

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
            b, a = sgn.butter(N=3, Wn=20/(fs_imu/2), btype='lowpass')
            self.raw_imu = {k: pd.DataFrame(data=sgn.filtfilt(b, a, v.values), columns=v.columns) for k, v in self.raw_imu.items()}

        return self.raw_imu
    
    # --- Get segmented IMU data
    def get_segmented_imu(self):
        if not self.segmented_data.keys():
            self.import_data()
        return self.segmented_data

    # --- Import EMG
    def get_raw_emg(self, remove_ecg=False, with_ecg=False):
        
        if not os.path.exists(self.db_path):
            print_error(f'Subject {self.name} not found in the archive.')
            return
        if remove_ecg or with_ecg:
            self.raw_emg = {}
        if not self.raw_emg.keys():
            for t in self.get_trials():
                emg_raw_filename = path.join_path([
                        self.db_path,
                        t,
                        'rawdata',
                        'emg.csv'
                    ])
                emg_noecg_filename = path.join_path([
                        self.db_path,
                        t,
                        'rawdata',
                        'emg_noecg.csv'
                    ])
                if (os.path.exists(emg_noecg_filename)) and not (remove_ecg or with_ecg):
                    print_ongoing(f'Using saved noECG data for trial {t}')
                    filename_ok = emg_noecg_filename
                else:
                    if with_ecg:
                        print_ongoing(f'Using raw EMG for trial {t}')
                    filename_ok = emg_raw_filename
                    remove_ecg = True
                self.raw_emg[t] = pd.read_csv(filename_ok)
                if remove_ecg and not with_ecg:
                    print_ongoing(f'Removing ECG from trial {t}')
                    emg_tmp = np.concatenate([
                        self.raw_emg[t].values[:,:-2],
                        emg.remove_ECG(self.raw_emg[t].values[:,-1])[0].reshape(-1,1),
                        emg.remove_ECG(self.raw_emg[t].values[:,-2])[0].reshape(-1,1)
                    ], axis=1)
                    colnames = list(self.raw_emg[t].columns)
                    self.raw_emg[t] = pd.DataFrame(emg_tmp, columns=colnames)
                    self.raw_emg[t].to_csv(emg_noecg_filename, index=None)
                self.raw_emg[t] = self.raw_emg[t].iloc[int(self.get_absolute_indexes()[t][0]*fs_emg):int(self.get_absolute_indexes()[t][-1]*fs_emg),:]
            for t in self.raw_emg.keys():
                self.raw_emg[t] = self.raw_emg[t].iloc[:,self.muscle_indexes]

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
        indexes_out = {k: np.asarray(v-v[0]) for k,v in idx.items()}
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

        if not self.cycle_events.keys():
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
            print_ongoing('Calculating phase limits...')
            self.phases_limits = kinematics.get_zones(self.get_raw_imu(), labels=self.segmentation_labels)
        return self.phases_limits

    # --- Set zones
    def set_limits(self, limits_in):
        if (limits_in.shape[0]==2) and (limits_in.shape[1]==2):
            self.phases_limits = limits_in
        else:
            print_error('Wrong limits as input, limits not set')

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
    def fit_to_phases(self, data_in, method='mean'):
        if not self.phases.keys():
            self.get_zones()
        data = {k: sgn.resample(v, self.phases[k].size) for k, v in data_in.items()}
        out = {k: kinematics.average_by_phase(v, self.phases[k], method) for k, v in data.items()}
        return out

    # --- Plot fitted data
    def plot_to_phases(self, data_in, method='mean'):
        if not self.phases.keys():
            self.get_zones()
        matrices_in = self.fit_to_phases(data_in, method)
        matrices_tot = np.asarray([v for v in matrices_in.values()])
        min_value, max_value = np.min(matrices_tot), np.max(matrices_tot)
        fig, ax = plt.subplots(int(len(data_in.keys())/2), 2, figsize=(16,16), facecolor='w')
        for i, k in enumerate(self.trials_order):
            v = matrices_in[k]
            col = i%2
            row = int(i/2)
            kinematics.plot_phases(v, ax=ax[row,col], vmin=min_value, vmax=max_value)
            ax[row,col].set_title(
                k.replace('_', ' ').title(),
                fontsize=16
            )
        plt.show(fig)
        return matrices_in

    # --- Get embedded recurrence matrix
    def get_recurrence_matrix(self, ds_step=5, samples_to_skip = 1500, tau=None, n_dim=5, n_cyc=10, direction='plane'):
        self.import_data()
        if direction == 'plane':
            col_idx = [0,1]
        elif direction == 'ml':
            col_idx = [0]
        elif direction == 'ap':
            col_idx = [1]
        else:
            col_idx = [0,1]
        if tau is None:
            l_cycle = {k: int(np.mean(np.diff(v))*102.4) for k, v in self.cycle_events.items()}
        data = {k: v[self.segmentation_labels].values[samples_to_skip:,col_idx] for k,v in self.segmented_data.items()}
        data = {k: rqa.normalize_trunk_data(v) for k,v in data.items()}
        rr = {
            k:
            rqa.embedded_recurrence_matrix(
                v, 
                ds_step, 
                tau=l_cycle[k], 
                m=n_dim, 
                n_tau=n_cyc)
            for k, v in data.items()
        }
        return rr

    # *********** Synergies functions *************

    # --- Get synergies
    def get_synergies(self):
        emg_tmp = self.get_raw_emg()
        events_tmp = self.get_indexes()
        if not all([self.H_tot.keys(), self.W_tot.keys(), self.VAF_curve.keys(), self.VAF_muscles.keys()]):
            print_ongoing('Extracting synergies...')
            for t in emg_tmp.keys():
                self.VAF_curve[t], self.W_tot[t], self.H_tot[t], self.VAF_muscles[t] = extract_synergies(emg_tmp[t].values, events_tmp[t], ds_factor=10)
        return self.VAF_curve, self.W_tot, self.H_tot, self.VAF_muscles

    # --- Plot VAF curves
    def plot_VAF(self, ths=None):
        VAF_tmp, _, _, VAFm_tmp = self.get_synergies()
        fig, ax = plt.subplots(2, len(VAF_tmp.keys()), facecolor='w', figsize=(20,10), sharex=True, sharey='row')
        for i, k in enumerate(VAF_tmp.keys()):
            t = np.linspace(1, VAF_tmp[k].size, VAF_tmp[k].size)
            ax[0,i].plot(t, VAF_tmp[k], 'k')
            if ths is not None:
                ax[0,i].plot(t, np.ones(t.shape)*ths, 'r')
            ax[1,i].plot(t, VAFm_tmp[k], 'k')
            if ths is not None:
                ax[1,i].plot(t, np.ones(t.shape)*ths, 'r')
            ax[0,i].set_title(k.replace('_',' ').title(), fontsize=16)
            ax[0,i].grid('on')
            ax[1,i].grid('on')
        plt.show()

    # --- Get VAF Curve
    def get_VAF_curve(self, mode='global'):
        VAF_tmp, _, _, VAFm_tmp = self.get_synergies()
        if mode=='global':
            return VAF_tmp
        elif mode=='muscles':
            return VAFm_tmp
        else:
            print_error('Invalid VAF mode, returning global.')
            return VAF_tmp

    # --- Set number of synergies
    def set_nsyn(self, n_in):
        if n_in<1:
            print_error('Invalid number of synergies')
        else:
            self.n_syn = int(np.round(n_in))
            print_success(f'Number of synergies set to {self.n_syn}')
        
    # --- Get synergy components
    def get_synergy_components(self):
        if self.n_syn is None:
            print_error('Define number of synergies before extracting components')
        else:
            if not all([self.H_tot.keys(), self.W_tot.keys(), self.VAF_curve.keys(), self.VAF_muscles.keys()]):
                self.get_synergies()
            for t in self.W_tot.keys():
                self.W[t] = np.asarray(self.W_tot[t][self.n_syn - 1])
                self.H[t] = np.asarray(self.H_tot[t][self.n_syn - 1])
                for i in range(self.n_syn):
                    self.H[t][i,:] *= np.linalg.norm(self.H[t][i,:])
                    self.W[t][:,i] /= np.linalg.norm(self.W[t][:,i])
            self.VAF_extraction = {k: v[self.n_syn-1] for k, v in self.VAF_curve.items()}
        return self.W, self.H, self.VAF_extraction

    # --- Order synergies
    def order_synergies(self, W_ref=None):
        W_all, H_all, _ = self.get_synergy_components()
        trials_ = self.get_trials()
        if W_ref is None:
            W_ref = W_all[trials_[0]]
        for t in trials_:
            sort_idx = sort_W(W_ref, W_all[t])
            W_all[t] = W_all[t][:,sort_idx]
            H_all[t] = H_all[t][sort_idx,:]
        W_tot = []
        for t in trials_:
            W_tot.append(W_all[t])
        W_template = np.median(np.asarray(W_tot), axis=0).squeeze()
        for t in trials_:
            sort_idx = sort_W(W_template, W_all[t])
            W_all[t] = W_all[t][:,sort_idx]
            H_all[t] = H_all[t][sort_idx,:] 
        self.W = W_all
        self.H = H_all
        W_tot = []
        for t in trials_:
            W_tot.append(W_all[t])
        W_template = np.median(np.asarray(W_tot), axis=0).squeeze()
        self.W_template = W_template

        return self.W, self.H

    # --- Get W template
    def get_W_template(self):
        if not list(self.W_template):
            self.order_synergies()
        return self.W_template

    # --- Reconstruction with template
    def reconstruct_H(self, W_template_=None):
        if W_template_ is not None:
            self.W_template = W_template_
        else:
            W_template_ = self.get_W_template()
        emg_tmp = self.get_raw_emg()
        events_tmp = self.get_indexes()
        print_ongoing('Reconstructing synergies')
        for t in emg_tmp.keys():
            self.H_reconstruction[t], self.VAF_reconstruction[t] = nnr(emg_tmp[t].values, W_template_, events_tmp[t])

        return self.H_reconstruction

    # --- Get reconstructed profile
    def get_H_rec(self, W_template_=None):
        if not self.H_reconstruction.keys():
            self.reconstruct_H(W_template_=W_template_)
        return self.H_reconstruction

    # --- Get reconstructed VAF
    def get_VAF_rec(self, W_template_=None):
        if not self.H_reconstruction.keys():
            self.reconstruct_H(W_template_=W_template_)
        return self.VAF_reconstruction

    # --- Set muscles for analysis
    def set_muscles(self, str_in):
        if str_in == 'all':
            print_success('EMG analysis set to all muscles')
            self.muscle_indexes = list(range(16)) # Change here for all muscles
        elif str_in == 'legs':
            print_success('EMG analysis set to both legs')
            self.muscle_indexes = [0,1,2,3,4,5,6,7,12,13]
        elif str_in == 'right':
            print_success('EMG analysis set to right leg')
            self.muscle_indexes = [0,2,4,6,12]
        elif str_in == 'left':
            print_success('EMG analysis set to left leg')
            self.muscle_indexes = [1,3,5,7,13]
        elif str_in == 'upper':
            print_success('EMG analysis set to upper body')
            self.muscle_indexes = [8,9,10,11,14,15]
        elif str_in == 'trunk':
            print_success('EMG analysis set to trunk')
            self.muscle_indexes = [10,11,14,15]
        else:
            print_error('Invalid muscle config. Setting to all muscles.')
            self.muscle_indexes = list(range(14)) # Change here for all muscles
        self.raw_emg = {}
    
    def filter_ECG(self):
        self.raw_emg = {}
        self.get_raw_emg(remove_ecg=True)