import os
import numpy as np
import matplotlib.pyplot as plt

from utils import bandpass
from analysis import BasePlotter, PlotterArgParser

class RawECP(BasePlotter):
    def __init__(self, nwbfile, outdir, **kwargs):
        self.bandpass = kwargs.pop('bandpass', None)

        super(RawECP, self).__init__(nwbfile, outdir, **kwargs)

    def add_stim_line(self, where='top'):
        tr = self.nwb.trials
        stim_idx = tr['sb'][:] == 's'
        start_times = tr['start_time'][stim_idx]
        stop_times = tr['stop_time'][stim_idx]
        top = plt.ylim()[1 if where=='top' else 0]

        for (start, stop) in zip(start_times, stop_times):
            if start > self.tstop or stop < self.tstart:
                continue
            plt.plot([start, stop], [top, top], color='blue')
        
    
    def plot_one(self, channel):
        if args.tstart is None or args.tstop is None:
            raise ValueError('Must specify --tstart and --tstop for RawECP')
        
        fig = plt.figure(figsize=(8, 2))
        rate = self.raw_dset.rate
        istart, istop = int(self.tstart*rate), int(self.tstop*rate)
        ch_data = self.raw_dset.data[istart:istop, channel]

        # TODO: axis labels
        t = np.arange(self.tstart, self.tstop, 1.0/rate)

        # fix off-by-one time round errors
        t, ch_data = self.fix_len_off_by_one(t, ch_data)

        if self.bandpass:
            print('bandpassing from {} to {}'.format(self.bandpass[0], self.bandpass[1]))
            ch_data = bandpass(ch_data, rate, lowcut=self.bandpass[0], highcut=self.bandpass[1])

        plt.plot(t, ch_data, linewidth=0.5, color='red')
        self.add_stim_line()
        plt.xlabel('Time (s)')
        plt.tight_layout()

        if not self.nosave:
            fn = 'rawECP_{}_ch{:02d}_{}.{}'.format(
                self.device, channel, self.identifier, self.filetype
            )
            full_fn = os.path.join(self.outdir, fn)
            plt.savefig(full_fn)

        if self.show:
            plt.show()

        fig.clear()
        
    def do_plots(self):
        dset = self.raw_dset
        n_timepts, n_ch = dset.data.shape[:2]
        for i in range(n_ch):
            self.plot_one(i)

    
if __name__ == '__main__':
    parser = PlotterArgParser()
    parser.add_argument('--bandpass', nargs=2, type=float, required=False, default=None,
                        help='low/high cutoffs to bandpass before running')
    args = parser.parse_args()

    analysis = RawECP(args.nwbfile, args.outdir, tstart=args.tstart, tstop=args.tstop, filetype=args.filetype, bandpass=args.bandpass)
    analysis.run()
