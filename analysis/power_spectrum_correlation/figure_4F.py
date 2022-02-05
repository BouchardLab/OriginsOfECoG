"""This script reproduces figure 4F

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse


def open_file(path):
    fh = h5py.File(path, 'r')
    return fh


def get_sim_spect(fh, trial, channel):
    dset = fh.get('scratch')
    freq = dset['freqs'].value
    pwr = dset[f'spectrum_trial_{trial}_channel_{channel}']
    return pwr, freq


def get_exp_spect_nwb(fh, trial='avg'):
    dset = fh.get('scratch')
    nchan = len(dset) - 1
    freq = dset['freqs'].value
    pwr_list = []
    for ch in range(nchan):
        pwr_list.append(
            dset[f'spectrum_trial_{trial}_channel_{ch}'][:].reshape(-1, 1))
    pwr = np.concatenate(pwr_list, axis=1).transpose()
    return pwr, freq


def get_exp_spect(fh):
    pwr = fh.get('power_spectra').value
    return pwr


def make_figure():
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    return fig, ax


def make_plot_figure4F(freq, exp, sim_trials):
    fig, ax = make_figure()
    ax.semilogx(freq, exp.T, color=0.6*np.ones(3), linewidth=0.9, alpha=0.25)
    exp_mean = np.mean(exp, axis=0)
    exp_mean_norm = norm_channels(exp_mean.reshape((1, -1))).squeeze(axis=0)
    sim_avg = np.mean(sim_trials, axis=1)
    sim_avg_norm = norm_channels(sim_avg.reshape((1, -1))).squeeze(axis=0)
    ax.semilogx(freq, exp_mean_norm, 'black', linewidth=2)
    ax.semilogx(freq, sim_trials, color='#FF8A8A', linewidth=0.9, alpha=0.25)
    ax.semilogx(freq, sim_avg_norm, 'red', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim([10, 1200])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(1)
    return fig, ax


def filter_channels(data, thres=0):
    max_vals = np.max(data, axis=1)
    mask = max_vals > thres
    data_filt = data[mask, :]
    return data_filt


def norm_channels(data):
    min_vals = np.min(data, axis=1).reshape(-1, 1)
    data_norm = (data - min_vals)
    max_vals = np.max(data_norm, axis=1).reshape(-1, 1)
    data_norm = data_norm/max_vals
    return data_norm


def make_correlation_matrix_plot(x):
    plt.figure()
    plt.imshow(x, vmin=0, vmax=1, cmap='gray')
    plt.xlabel('Simulation trials')
    plt.ylabel('Experiment channels')
    plt.colorbar()
    return plt.gcf(), plt.gca()


def plot_stats(ax, x):
    mu = x.mean()
    median = np.median(x)
    sigma = x.std()
    first_quarter = np.percentile(x, 25)
    third_quarter = np.percentile(x, 75)
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu, ),
        r'$\sigma=%.2f$' % (sigma, ),
        r'$Q1=%.2f$' % (first_quarter, ),
        r'$\mathrm{median}=%.2f$' % (median, ),
        r'$Q3=%.2f$' % (third_quarter)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def make_histogram_plot(x):
    fig, ax = plt.subplots(1, 1)
    ax.hist(x, bins=50, density=True, color='black', edgecolor='gray')
    ax.set_xlabel('Correlation values')
    ax.set_ylabel('Count')
    plot_stats(ax, x)
    return fig, ax


def make_box_whisker(x):
    fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    ax.boxplot(x, showfliers=False, whis=[2.5, 97.5])
    ax.set_ylim([0, 1])
    return fig, ax


def run(plot, save, path, exp_ext, exp_files, sim_file):

    # append file extension
    exp_files = [x + exp_ext for x in exp_files]
    # exp spectra
    data_list = []
    for file in exp_files:
        path_file = path / file
        fh = open_file(path_file)
        if file.endswith('nwb'):
            pwr, _ = get_exp_spect_nwb(fh)  # get data from nwb
        else:
            pwr = get_exp_spect(fh)  # get data from h5
        pwr_filt = filter_channels(pwr)
        print(f'Number of good channels: {pwr_filt.shape[0]}')
        pwr_norm = norm_channels(pwr_filt)
        data_list.append(pwr_norm)
        fh.close()
    exp = np.concatenate(data_list)

    # sim spectra
    path_file = path / sim_file
    fh = open_file(path_file)
    sim_trials = []
    for idx in range(60):
        pwr, freq = get_sim_spect(fh, trial=idx, channel=0)
        pwr = np.reshape(pwr[:], (1, -1))
        pwr_norm = norm_channels(pwr).squeeze(axis=0)
        sim_trials.append(pwr_norm)
    sim_trials = np.stack(sim_trials, axis=1)
    fh.close()

    # correlation analysis
    exp = exp.T
    N, M = exp.shape[1], sim_trials.shape[1]
    corr_mat = np.zeros((N, M))
    for i in range(N):  # iterate over rows / exp channels
        for j in range(M):  # iterate over cols / sim trials
            tmp = np.corrcoef(exp[:, i], sim_trials[:, j])
            corr_mat[i, j] = tmp[0, 1]

    # correlation mean simulation and average experiment channels
    corr_avg_mat = np.zeros((N, 1))
    for i in range(N):  # iterate over rows / exp channels
        sim_avg = np.mean(sim_trials, axis=1)
        sim_avg_norm = norm_channels(sim_avg.reshape((1, -1))).squeeze(axis=0)
        tmp = np.corrcoef(sim_avg_norm, exp[:, i])
        corr_avg_mat[i] = tmp[0, 1]

    figs = []
    fig, _ = make_plot_figure4F(freq, exp.T, sim_trials)
    figs.append(fig)
    fig, _ = make_box_whisker(corr_avg_mat)
    figs.append(fig)
    fig, ax = make_histogram_plot(np.reshape(corr_avg_mat, (-1, 1)))
    ax.set_title('Simulation and experiment channel averages')
    figs.append(fig)
    # These were plots that Kris decided against
    # fig, _ = make_correlation_matrix_plot(corr_mat)
    # figs.append(fig)
    # fig, ax = make_histogram_plot(np.reshape(corr_mat, (-1, 1)))
    # ax.set_title('Simulation trials and experiment channel averages')
    # figs.append(fig)
    # fig, ax = make_histogram_plot(np.max(corr_mat, axis=0))
    # ax.set_title('Simulation trials and max experiment channel average')
    # figs.append(fig)

    if plot == 1:
        for idx, fig in enumerate(figs):
            fig.show()
            input('Press any key: ')
    if save == 1:
        for idx, fig in enumerate(figs):
            fig.savefig(f'figure4F_{idx}' + '.png')
            fig.savefig(f'figure4F_{idx}' + '.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script reproduces figure 4F')
    parser.add_argument('--path', type=str,
                        help='Path to data files')
    parser.add_argument('--exp_ext', default='nwb',
                        help='File type for experiment data (either .h5 .nwb')
    parser.add_argument('--exp_files', default=['R6_B10', 'R6_B16', 'R32_B7', 'R18_B12'],
                        help='List of experimental files')
    parser.add_argument('--sim_file', default='simulation_ecp_layers.nwb',
                        help='Simulation data file')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot, by default True')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save, by default False')
    args = parser.parse_args()
    run(plot=args.plot,
        save=args.save,
        path=Path(args.path),
        exp_ext=args.exp_ext,
        exp_files=args.exp_files,
        sim_file=args.sim_file)
