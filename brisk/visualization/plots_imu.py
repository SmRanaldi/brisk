import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plot parameters for a subject
def plot_parameters(param, segment, sensor):

    plot_options = {
        'notch':True, 
        'boxprops':{
            'linewidth': 2, 
            'edgecolor': [0,0.2,0.7], 
            'facecolor': 'w'
        }, 
        'medianprops':{
            'linewidth': 4, 
            'color': 'g'
        }, 
        'showfliers':False
    }
    dimensions = np.unique(param['dimension'])
    parameter_name = np.unique([x.split('_')[0] for x in param.columns if '_' in x])
    trials_labels = [x.replace('_',' ').title() for x in np.unique(param['trial'])]
    directions_labels = ['Total', 'VT', 'AP', 'ML']

    fig, ax = plt.subplots(len(parameter_name), len(dimensions), figsize=(16,11), facecolor='w', sharex=True, sharey='row')

    for i,p in enumerate(parameter_name):
        for j,d in enumerate(dimensions):
            sns.boxplot(
                data=param[(param['segment'] == segment) & (param['dimension'] == d)],
                x = 'trial',
                y = p+'_'+sensor,
                ax = ax[i,j],
                **plot_options
            )
            ax[i,j].grid('on')
            ax[i,j].set_xticklabels(trials_labels, rotation=45, fontsize=14)
            ax[i,j].tick_params(axis='y', labelsize=14)
            ax[i,j].set_facecolor([0.9,0.9,0.9])
            ax[i,j].set_xlabel('')
            ax[i,j].set_ylabel(p.replace('_',' ').title(), fontsize=14)

    for a in ax[:,1:].flatten():
        a.set_ylabel('')
    for i, d in enumerate(dimensions):
        ax[0,i].set_title(directions_labels[i], fontsize=16)
    fig.align_ylabels(ax)
    plt.show()