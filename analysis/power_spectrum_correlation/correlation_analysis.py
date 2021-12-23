import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

ROOT_PATH = '/home/jhermiz/data/aer/OriginsOfECoG/'
EXP_FILES = ['R6_B10', 'R6_B16', 'R32_B7', 'R18_B12']
EXP_FILES = [x + '.h5' for x in EXP_FILES]
SIM_FILE = 'simulation_ecp_layers.nwb'

def open_file(path):
    fh = h5py.File(path, 'r')    
    return fh

def get_sim_spect(fh):
    dset = fh.get('scratch')
    freq = dset['freqs'].value
    pwr = dset['spectrum_ch_0']
    return pwr, freq

def get_exp_spect(fh):
    pwr = fh.get('power_spectra').value
    return pwr

def make_figure():
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    return fig, ax

def make_plot(freq, exp, sim):
    fig, ax = make_figure()
    ax.semilogx(freq, exp.T, color=0.6*np.ones(3))
    exp_mean = np.mean(exp, axis=0)
    exp_mean_norm = exp_mean/np.max(exp_mean)
    ax.semilogx(freq, exp_mean_norm, 'black', linewidth=2)
    ax.semilogx(freq, sim, 'red', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Z-score (norm)')
    ax.set_xlim([10, 1200])
    plt.show()

def filter_channels(data, thres=3):
    max_vals = np.max(data, axis=1)
    mask = max_vals > thres
    data_filt = data[mask, :]
    return data_filt

def norm_channels(data):
    max_vals = np.max(data, axis=1).reshape(-1, 1)
    data_norm = data/max_vals
    return data_norm

def run():
    # exp spectra
    data_list = []
    for file in EXP_FILES:
        path_file = os.path.join(ROOT_PATH, file)
        fh = open_file(path_file)
        pwr = get_exp_spect(fh)
        pwr_filt = filter_channels(pwr)
        pwr_norm = norm_channels(pwr_filt)
        #pwr_t = pwr.T #make freq by channels
        data_list.append(pwr_norm)    
        fh.close()
    exp = np.concatenate(data_list)
    
    #sim spectra
    fh = open_file(os.path.join(ROOT_PATH, SIM_FILE))
    pwr, freq = get_sim_spect(fh)
    sim = pwr/np.max(pwr)
    fh.close()
    
    make_plot(freq, exp, sim)
    
    
run()

## Test norm_channels
# data = np.random.rand(8, 6)*3
# data_norm = norm_channels(data)
# plt.plot(data[0, :])
# plt.plot(data_norm[0, :])
# plt.show()