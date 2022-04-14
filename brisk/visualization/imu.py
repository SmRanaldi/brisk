from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

from brisk.utils import path
from brisk.analysis import segmentation

# ----- Constants -----
sensors = ['acc', 'gyr', 'mag']
sensors_labels = ['Accelerometer', 'Gyroscope', 'Magnetometer']
fc = 0.9 * np.asarray([1, 1, 1])
fs = 20
fs_title = 24
lw = 3
figsize = (20,15)
x_labels = ['0', '25', '50', '75', '100']

# ----- Auxiliary methods -----

def prepare_all_profiles(data_in, idx_in):
    data_out = []
    for i in range(len(idx_in)-1):
        data_out.append(data_in.values[idx_in[i]:idx_in[i+1],:])
    return np.asarray(data_out)

# ----- Functions -----

def plot_all_trials(subject, segment):

    trials_list = path.get_trials(subject)
    if segment not in ['trunk', 'arm', 'forearm']:
        print(f'Segment {segment} not in data')
        return

    data_avg={t: segmentation.get_average_profiles(subject, t) for t in trials_list}
    colors = ['k','r','b','g']
    labels = [t.replace('_',' ').title() for t in trials_list]
    data_col = [x for x in data_avg[trials_list[0]] if segment in x]

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=figsize, facecolor='w', sharex=True)

    for it, t in enumerate(trials_list):
        data_to_plot=data_avg[t][data_col]
        for i, s in enumerate(sensors):
            data_sens = data_to_plot[[x for x in data_to_plot.columns if s in x]].values
            for j in range(3):
                ax[j,i].plot(data_sens[:,j], color=colors[it], lw=lw, label=labels[it])
                ax[j,i].set_xticks([])

    for i in range(3):
        ax[0,i].set_title(sensors_labels[i], fontsize=fs_title) 
        ax[-1,i].set_xticks(np.linspace(0,data_sens.shape[0],5))
        ax[-1,i].set_xticklabels(x_labels)
        ax[-1,i].set_xlabel('% of gesture')

    ax[0,0].legend(labels)
    ax[0,0].set_ylabel('VT', fontsize=fs)
    ax[1,0].set_ylabel('AP', fontsize=fs)
    ax[2,0].set_ylabel('ML', fontsize=fs)
    for a in ax.flatten():
        a.grid('on')
        a.set_facecolor(fc)
    fig.show()
