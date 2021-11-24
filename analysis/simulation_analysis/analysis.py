"""
Base classes for analysis objects
"""
import os
import h5py
from pynwb import NWBHDF5IO
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

class BasePlotter(object):
    def __init__(
            # TODO: remove outdir as an arg (saving should happen manually in the calling script)
            self, nwbfile, outdir, mode='r', device='ECoG', auxfile=None,
            raw_dset_name='Raw', proc_dset_name='Hilb_54bands', block=None,
            filetype='pdf', identifier='', no_baseline_stats=False,
            channel=None, stim_i=None, tstart=None, tstop=None,
            figsize=None, ax=None,
            color=None, linewidth=None, label=None, nosave=False, show=False,
            is_expt=False, nwb=None, write=False
    ):
        """
        identifier - appended to filename
        """
        self.nwbfile = nwbfile
        self.auxfile = auxfile
        self.device = device
        self.channel = channel
        self.block = block
        self.raw_dset_name = raw_dset_name
        self.proc_dset_name = proc_dset_name
        self.outdir = outdir
        self.filetype = filetype
        self.identifier = identifier
        self.stim_i = stim_i if stim_i is not None else 'avg'
        self.tstart = tstart
        self.tstop = tstop
        self.is_expt = is_expt
        self.write = write
        
        # TODO: remove?
        self.nosave = nosave
        self.show = show

        if figsize and not ax:
            plt.figure(figsize=figsize)

        self.linewidth = linewidth
        self.color = color
        self.label = label

        if nwb:
            self.nwb = nwb
        else:
            self.io = NWBHDF5IO(self.nwbfile, mode)
            self.nwb = self.io.read()

        
        if not no_baseline_stats:
            self.raw_bl_stats = self._compute_raw_baseline_stats()
            self.proc_bl_stats = self._compute_proc_baseline_stats()

    @property
    def proc_dset(self):
        try:
            return self.nwb.modules[self.proc_dset_name].data_interfaces[self.device]
        except KeyError:
            raise KeyError("Unable to read processed data from {} (probably needs to be preprocessed)".format(self.nwbfile))

    @property
    def raw_dset(self):
        return self.nwb.acquisition[self.raw_dset_name].electrical_series[self.device]

    @property
    def n_ch(self):
        return self.raw_dset.data.shape[1]
        
    def do_plots(self):
        """subclasses override"""
        dset = self.proc_dset
        n_timepts, n_ch = dset.data.shape[:2]
        for i in range(n_ch):
            self.plot_one(i)
        plt.clf()

    def fix_len_off_by_one(self, x, y):
        """
        In case a timeseries doesn't line up with its t-axis, fix by cutting off one timepoint
        from the end of whichever series is shorter
        """
        if len(x) == len(y) + 1:
            x = x[:-1]
        elif len(x) == len(y) - 1:
            y = y[:-1]
        return x, y

    def get_stim_periods(self, rate=None, pre_dur=0.1, post_dur=0.1):
        """
        Return stim-on times in seconds, unless rate is passed, in which case
        in samples at the given sampling rate
        """
        trials = self.nwb.trials
        idxs = trials['sb'][:] == 's'
        times = zip(trials['start_time'][idxs]-pre_dur, trials['stop_time'][idxs]+post_dur)

        if rate:
            return [(int(t[0]*rate), int(t[1]*rate)) for t in times]
        else:
            return times

    def get_baseline_periods(self, rate=None):
        """
        Return baseline period times in seconds, unless rate is passed, in which case
        in samples at the given sampling rate
        """
        trials = self.nwb.trials
        bl_idxs = trials['sb'][:] == 'b'
        times = zip(trials['start_time'][bl_idxs], trials['stop_time'][bl_idxs])
        
        if rate:
            return [(int(t[0]*rate), int(t[1]*rate)) for t in times]
        else:
            return times

    def _compute_raw_baseline_stats(self):
        # TODO
        return None, None

    def _compute_proc_baseline_stats(self):
        """
        Compute baseline stats per frequency band for the given channel data
        """
        if self.auxfile and os.path.exists(self.auxfile):
            print("Using saved baseline stats")
            with h5py.File(self.auxfile) as infile:
                return infile['/bl_mu'][:], infile['/bl_std'][:]
        else:
            print("Computing baseline stats")
            full_data = self.proc_dset.data
            rate = self.proc_dset.rate
            bl_periods = self.get_baseline_periods(rate=rate)
            idx = np.zeros(full_data.shape[0], dtype=bool)
            for t1, t2 in bl_periods:
                idx[t1:t2] = True
            bl_data = full_data[idx, ...]
            bl_mean = np.average(bl_data, axis=0)
            bl_std = np.std(bl_data, axis=0)
            return (bl_mean, bl_std)

    def get_bfs(self):
        with h5py.File(self.auxfile) as infile:
            return infile['/bf'][:]

        
    def run(self):
        self.do_plots()
        
        if self.write:            
            # write additions to nwb file


class PlotterArgParser(ArgumentParser):
    kwarg_fields = [
        'tstart', 'tstop', 'stim_i', 'identifier', 'filetype', 'nosave', 'show',
        'proc_dset_name', 'auxfile', 'write'
    ]
    def __init__(self, *args, **kwargs):
        super(PlotterArgParser, self).__init__(*args, **kwargs)
        self.add_argument('--nwbfile', '--nwb', type=str, required=True)
        self.add_argument('--auxfile', '--aux', type=str, required=False, default=None)
        self.add_argument('--outdir', type=str, required=False, default='.')
        self.add_argument('--proc-dset-name', '--proc-dset', '--proc', type=str, required=False,
                          default='Hilb_54bands')
        self.add_argument('--tstart', type=float, required=False, default=None)
        self.add_argument('--tstop', type=float, required=False, default=None)
        self.add_argument('--stim-i', type=int, required=False, default=None)
        self.add_argument('--channel', type=int, required=False, default=None)
        self.add_argument('--identifier', type=str, required=False, default='',
                          help='append this string to filename')
        self.add_argument('--filetype', '--extension', '--ext', required=False, default='pdf')
        self.add_argument('--nosave', default=False, action='store_true')
        self.add_argument('--show', default=False, action='store_true')
        self.add_argument('--write', default=False, action='write spectrum to nwb')

    @property
    def kwargs(self):
        args = self.parse_args()
        return {f: getattr(args, f) for f in PlotterArgParser.kwarg_fields}
