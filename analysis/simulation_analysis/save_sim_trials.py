"""This script saves the spectra for each trial given a set of simulation file(s)
"""
from pathlib import Path
from power_spectrum import PowerSpectrum, PowerSpectrumRatio
import argparse


def process_one_trial(nwbfile, trial, norm, dset):

    if norm == 'zscore':
        analysis = PowerSpectrum(str(nwbfile), ',', stim_i=trial,
                                 proc_dset_name=dset,
                                 nosave=True, write=True)
    elif norm == 'mean':
        analysis = PowerSpectrumRatio(str(nwbfile), ',', stim_i=trial,
                                      proc_dset_name=dset,
                                      nosave=True, write=True)
    else:
        raise Exception(f'Normalization {norm} not expected')
    analysis.run()


def process_file(nwbfile, num_trials, norm, dset):
    if num_trials == 'AVG':
        process_one_trial(nwbfile, 'avg')
    else:
        num_trials = int(num_trials)
        for trial in range(num_trials):
            print(trial)
            process_one_trial(nwbfile, trial, norm, dset)


def run(path, files, num_trials, norm, dset):
    for nwbfile in files:
        path_file = path / nwbfile
        process_file(path_file, num_trials, norm, dset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given an nwb file that is ' +
                                     'preprocessed (ie has spectral coefficients), ' +
                                     'saves the average spectrum and performs normalization')
    parser.add_argument('--path', type=str,
                        help='Path to data nwb files')
    parser.add_argument('--dset', type=str, default='Hilb_54bands',
                        help='Which preprocessing datatset to use. ' +
                        'Options include: Hilb_54bands and wvlt_amp_downsampled_ECoG')
    parser.add_argument('--files', default=['simulation_ecp_layers.nwb'],
                        help='List of simulation files')
    parser.add_argument('--num_trials', default=60,
                        help='Number of trials to save. Can also be string "AVG" ' +
                        'in which case the average spectrum is saved in scratch')
    parser.add_argument('--norm', default='mean',
                        help='Frequency normalization used including: ' +
                        'mean and z-score. Default is mean')
    args = parser.parse_args()
    run(path=Path(args.path),
        files=args.files,
        num_trials=args.num_trials,
        norm=args.norm,
        dset=args.dset)
