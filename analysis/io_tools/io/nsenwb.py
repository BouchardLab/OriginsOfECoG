"""
Write NSE Lab rodent electrophysiological response recordings to NWB
"""

__author__ = 'Max Dougherty <maxdougherty@lbl.gov'

# General Libraries
import numpy as np
from os import sep, path
from datetime import *; from dateutil.relativedelta import *
import itertools
import types
import h5py

# Import Python NWB Library
from pynwb.core import DynamicTable, NWBDataInterface, NWBData
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.misc import DecompositionSeries

from mars.configs.block_directory import bl


class NSENWB(object):

    def __init__(self, block_name, nwb, e_regions):
        """Initialize NSENWB object to write an NWB file for an NSE Lab experiment.
        You should probably use NSENWB.from_block_name() or NSENWB.from_existing_nwb()

        Parameters
        ----------
        block_name : string
            Name of the block to read/write data for

        nwb : pynwb.NWBFile object
            NWBFile object (new datasets added via this class will exist alongside
            the ones already in this file).

        e_regions: dict
            Dictionary returned by NSENWB._read_electrode_table or NSENWB._create_electrode_table
        """
        self.block_name = block_name
        self.block_params = bl[block_name] #Block Parameters
        self.exp = self.block_params['experiment']
        self.stim = self.block_params['stimulus']
        self.dev_configs = self.block_params['device']

        self.nwb = nwb
        self.e_regions = e_regions

    @classmethod
    def from_block_name(cls, block_name):
        nwb = NSENWB._instantiate_nwb(block_name)
        e_regions = NSENWB._create_electrode_table(nwb, block_name)
        return cls(block_name, nwb, e_regions)

    @classmethod
    def from_existing_nwb(cls, block_name, nwbfilename, mode='a'):
        io = NWBHDF5IO(nwbfilename, mode)
        nwb = io.read()
        e_regions = NSENWB._read_electrode_table(nwb)
        nsenwb = cls(block_name, nwb, e_regions)
        nsenwb.io = io
        nsenwb.nwb_directory = path.split(nwbfilename)[0]
        return nsenwb

    @classmethod
    def _instantiate_nwb(cls, block_name):
        exp = bl[block_name]['experiment']
        
        experiment_description = bl[block_name]['stimulus']['name'] + ' Stimulus Experiment'

        return NWBFile(
            session_description='foo',#exp['session_description'],
            identifier=block_name,
            session_start_time=datetime.now(), #TODO: GET ACCURATE START TIME.
            file_create_date=datetime.now(),
            experimenter=exp['experimenter'],
            experiment_description=experiment_description,
            session_id=block_name,
            institution=exp['institution'],
            lab=exp['lab'],
            pharmacology=exp['pharmacology'],
            notes=exp['notes'],
            surgery=exp['surgery']
        )

    @classmethod
    def _read_electrode_table(cls, nwb, acq_name='Raw'):
        e_regions = {}
        for device, series in nwb.acquisition[acq_name].electrical_series.items():
            e_regions[device] = series.electrodes
        return e_regions

    @classmethod
    def _create_electrode_table(cls, nwb, block_name):
        dev_configs = bl[block_name]['device']

        e_regions = {}

        e_id_gen = itertools.count() #Electrode ID, unique for channels across devices
        for device_name, dev_conf in dev_configs.items():
            if isinstance(dev_conf,str): #Skip mark and audio,
                continue
            ## Create the device
            device_source = dev_conf['manufacturer']
            device = nwb.create_device(name=device_name, 
                                            #source=device_source
                                            )
            ## Create the electrode group
            e_group = nwb.create_electrode_group(
                name=device_name,
                description='',
                location='',
                device=device)

            ## Add each electrode
            ch_ids = dev_conf['ch_ids']
            ch_pos = dev_conf['ch_pos']
            device_region_ids = []
            for i in ch_ids:
                e_id = next(e_id_gen)
                nwb.add_electrode(
                    e_id,
                    x=ch_pos[i]['x'],
                    y=ch_pos[i]['y'],
                    z=ch_pos[i]['z'],
                    imp=np.nan, #TODO: INCLUDE IMPEDENCE
                    location=str(i),
                    filtering='Low-Pass Filtered to Nyquist frequency',
                    group=e_group)
                device_region_ids.append(e_id) #Collect device channel IDs for electrode table region

            ## Create the electrode table region for this device
            table_region = nwb.create_electrode_table_region(
                                region=device_region_ids,
                                description='')
            e_regions[device_name]=table_region

        return e_regions


    #################
    ## ADD METHODS ##
    #################

    def add_raw(self, data, device_name, acq_name='Raw'):
        
        # Get the device configuration
        dev_conf = self.dev_configs[device_name]

        # Create the electrical series
        e_series = ElectricalSeries(name=device_name, #name
                                    data=data, #data
                                    electrodes=self.e_regions[device_name], #electrode table region
                                    starting_time=0.0,
                                    rate=dev_conf['sampling_rate'])

        # Create and add the LFP interface
        if acq_name not in self.nwb.acquisition:
            self.nwb.add_acquisition(LFP(name=acq_name))

        # Add the device dataset to the LFP interface
        self.nwb.acquisition[acq_name].add_electrical_series(e_series)

        

    def add_proc(self, data, device_name, dset_name, samp_rate, acq_name='Raw',
                 cfs=tuple(), sds=tuple(), band_limits=tuple()):
        """
        Add a processed dataset to the nwb file. If any of cfs, sds, or band_limits
        are specified, a DecompositionSeries is created, otherwise a TimeSeries

        cfs = a list of the center frequency of each band
        sds = a list of the stdev of each band
        band_limits = a list of 2-tuples specifying the lo and hi cutoffs of each band
        """
        # Create the electrical series
        source = self.nwb.acquisition[acq_name].electrical_series[device_name]

        if any([len(arr) for arr in (cfs, sds, band_limits)]):
            e_series = DecompositionSeries(
                name=device_name,
                description="processed data",
                data=data,
                metric='amplitude',
                starting_time=0.0,
                source_timeseries=source,
                rate=samp_rate
            )
        else:
            e_series = TimeSeries(
                name=device_name,
                description="processed data",
                data=data,
                starting_time=0.0,
                rate=samp_rate
            )

        # Add bands, if present
        for cf, sd, limits in itertools.zip_longest(cfs, sds, band_limits):
            if cf is None:
                cf = -1.
            if sd is None:
                sd = -1.
            if limits is None:
                limits = (-1., -1.)

            e_series.add_band(band_name=str(cf), band_limits=limits, band_mean=cf, band_stdev=sd)

        # Create the preprocessing module `dset_name` if it does not exist.
        if dset_name not in self.nwb.modules:
            proc_mod = self.nwb.create_processing_module(
                name=dset_name,
                description=''
            )
        else:
            proc_mod = self.nwb.modules[dset_name]

        # Remove the device from the processing module if it's already there
        # Doesn't work because ProcessingModule doesn't support deletion
        # if overwrite and device_name in proc_mod.data_interfaces:
        #     del proc_mod[dset_name]
        
        # Add the device dataset to the processing module
        proc_mod.add_data_interface(e_series)
    
    def add_mark(self, mark_track, rate, name='recorded_mark'):
        # Create the mark timeseries
        mrk_ts = TimeSeries(name=name,
                            data=mark_track,
                            unit='Volts',
                            starting_time=0.0,
                            rate=rate,
                            description='The neural recording aligned stimulus mark track.')
        # Add the mark track to the  NWBFile
        self.nwb.add_stimulus(mrk_ts)

    def add_stim(self, stim_track, rate, starting_time, name='raw_stimulus'):
        # Create the stimulus timeseries
        stim_ts = TimeSeries(name=name,
                            data=stim_track,
                            unit='Volts',
                            starting_time=starting_time,
                            rate=rate,
                            description='The neural recording aligned stimulus track.')
        # Add the mark track to the  NWBFile
        self.nwb.add_stimulus(stim_ts)

    def add_trial_column(self, *args, **kwargs):
        """ Pass through to self.nwb """
        self.nwb.add_trial_column(*args, **kwargs)

    def add_trial(self, *args, **kwargs):
        """ Pass through to self.nwb """
        self.nwb.add_trial(*args, **kwargs)

    def add_analysis_dataset(self,proc_name,device_name,dataset_name,data):
        # Write an NWB carriage dataset
        carr_path = path.join(self.nwb_directory,self.block_name+'.h5')
        device_path = proc_name + '/' + device_name
        dset_path = device_path + '/' + dataset_name
        with h5py.File(carr_path,'a') as f:
            if not proc_name in f:
                f.create_group(proc_name)
            if not device_path in f:
                f.create_group(device_path)
            if dset_path in f:
                del f[dset_path]
            f.create_dataset(dset_path,data=data)
            f.close()

        
    ##################
    ## READ METHODS ##
    ##################

    def read_raw(self, device_name, acq_name='Raw'):
        """
        Read a raw dataset from the currently open nwb file
        """
        return self.nwb.acquisition[acq_name].electrical_series[device_name]

    def read_proc(self, proc_name, device_name):
        proc_mod = self.nwb.modules[proc_name]
        return proc_mod.data_interfaces[device_name]
    
    def read_mark(self, name='recorded_mark'):
        return self.nwb.stimulus[name]

    def index_dset(self, dset,
                   device_channels=None,
                   dset_channels=None,
                   time_idx=None,
                   time_range=None,
                   trial_query=None,
                   pre_dur=0.0,
                   post_dur=0.0,
                   zscore=False,
    ):
        """
        Main function for indexing a dataset by channel and time/trial

        dset = nwb TimeSeries object

        # INDEX BY CHANNELS (choose at most 1):
        device_channels = channel numbers as specified in ch_ids in the block directory
        dset_channels = channel numbers in the dataset order

        # INDEX BY TIME (choose at most 1):
        time_idx = raw time indexes to return
        time_range = tuple of start, end time to return. Uses dset.rate to calculate indices
        trial_query = pandas dataframe query into the trials table. If specified, return value is an
                      iterator over stimulus presentation

        # Other options
        pre/post_dur = amount of time before/after trials to return
        zscore = if True, z-scores the returned data to total baseline stats
        """
        # INDEX BY TIME:
        if sum([bool(time_idx), bool(time_range), bool(trial_query)]) > 1:
            raise ValueError("Choose one time index method only")

        if time_idx:
            time_idx = time_idx
        elif time_range:
            time_idx = self._index_for_time_range(time_range, dset.rate, dset.starting_time)
        elif trial_query:
            time_idx = self._index_for_trials(dset.rate, trial_query, pre_dur, post_dur)
        else:
            time_idx = slice(None)


        # INDEX BY CHANNELS:
        if dset_channels and device_channels:
            raise ValueError("Choose one channel index method only")

        if dset_channels is not None:
            ch_idx = dset_channels
        elif device_channels is not None:
            ch_idx = self._index_for_device_channels(dset, device_channels)
        else:
            ch_idx = slice(None)

        if dset.data.ndim < 2:
            return dset.data[time_idx]

        # Prepare to zscore:
        if zscore:
            bl_data = np.concatenate(
                list(self.index_dset(dset, dset_channels=ch_idx, trial_query="sb == 'b'")),
                axis=0
            )
            bl_mean = np.mean(bl_data, axis=0)
            bl_std = np.std(bl_data, axis=0)
            maybe_zscore = lambda x: (x - bl_mean) / bl_std
        else:
            maybe_zscore = lambda x: x

        if isinstance(time_idx, types.GeneratorType):
            def _iter():
                for t_idx in time_idx:
                    yield maybe_zscore(np.atleast_2d(dset.data[t_idx, ch_idx, ...]))
            return _iter()
        else:
            return maybe_zscore(np.atleast_2d(dset.data[time_idx, ch_idx, ...]))

    @classmethod
    def _index_for_time_range(cls, time_range, rate, starting_time=0.0):
        # TODO: Allow for selecting multiple timeranges
        start = int(np.round((time_range[0]-starting_time) * rate))
        stop = int(np.round((time_range[1]-starting_time) * rate))
        return slice(start, stop)

    @classmethod
    def _index_for_device_channels(cls, dset, channels):
        device = dset.name # TODO: is dset.name always the device name?
        try:
            # processed dset
            electrodes_df = dset.source_timeseries.electrodes.table.to_dataframe().query('group_name == @device')
        except AttributeError:
            # raw dset
            electrodes_df = dset.electrodes.table.to_dataframe().query('group_name == @device')
        chs = {elec.location: i for i, elec in enumerate(electrodes_df.itertuples())}
        return [chs[str(chnum)] for chnum in channels]

    def _index_for_trials(self, rate, trial_query=None, pre_dur=0.0, post_dur=0.0):
        # Returns a generator
        table = self.nwb.trials.to_dataframe()
        if trial_query:
            table = table.query(trial_query)

        for s in table.itertuples():
            yield self._index_for_time_range((s.start_time-pre_dur, s.stop_time+post_dur), rate)

    # def electrode_order(self, device_name, device_channels, axis='z'):
    #     # Get the channel order
    #     device_raw = self.read_raw(device_name) #TODO: Can we read electrodes without needing to go through a dataset?
    #     channel_positions = []
    #     for ch in device_channels:
    #         query = 'group_name == "%s" & location == "%s"'%(device_name,ch)
    #         channel_positions.append(float(device_raw.electrodes.table.to_dataframe().query(query)[axis]))
    #     channel_positions = np.array(channel_positions)
    #     channel_order = np.arange(len(device_channels))[np.argsort(channel_positions)]
    #     return channel_order, np.sort(channel_positions)

    def has_analysis_dataset(self,device_path,device_name,dataset_name):
        # Check if NWB analysis dataset exists
        carr_path = path.join(self.nwb_directory,self.block_name+'.h5')
        dset_path = device_path + '/' + device_name + '/' + dataset_name
        if not path.exists(carr_path):
            return False
        with h5py.File(carr_path,'r') as f:
            if not dset_path in f:
                return False
        return True

    def read_analysis_dataset(self,device_path,device_name,dataset_name):
        # Read an NWB analysis dataset
        carr_path = path.join(self.nwb_directory,self.block_name+'.h5')
        dset_path = device_path + '/' + device_name + '/' + dataset_name
        if not path.exists(carr_path):
            return False
        with h5py.File(carr_path,'r') as f:
            if not dset_path in f:
                return False
            data = np.array(f[dset_path])
        return data

    # These functions are much less capable than index_dset() but are here for backwards compatibility
    def read_trials(self, dset, pre_dur=0.0, post_dur=0.0, trial_query=None):
        """
        Read data associated with a particular stimulus
        """
        return self.index_dset(dset, trial_query=trial_query, pre_dur=pre_dur, post_dur=post_dur)

    def index_by_device_channels(self, dset, channels, timerange=None):
        """
        dset - nwb Timeseries object
        channels - device-defined channel numbers
        """
        return self.index_dset(dset, device_channels=channels, time_range=timerange)

    ###################
    ## OTHER METHODS ##
    ###################
    def device_channels(self, device, remove_bad=False):
        """
        Return the device channel IDs.
        """
        elec = self.nwb.electrodes
        device_idx = elec['group_name'].data[:] == device
        device_chs = elec['location'].data[device_idx]
        if remove_bad:
            bad_chs = np.array(self.block_params['bad_chs'][device]).astype('str')
            device_chs = np.array([c for c in device_chs if not c in bad_chs])
        return device_chs

    def channel_positions(self, device, remove_bad=False):
        """
        Return an 3 column array containing the x,y,z positions of each electrode in device
        """
        elec = self.nwb.electrodes
        device_idx = elec['group_name'].data[:] == device
        device_chs = elec['location'].data[device_idx]
        return np.array([elec['x'].data[device_idx], 
                        elec['y'].data[device_idx],
                        elec['z'].data[device_idx]])

    def ordered_channels(self, device='Poly', reverse=False):
        """
        Return a list of device channel IDs (starting from 1) and dset indexes (starting from 0),
        sorted by z coordinate.
        Also return the corresponding z coordinates
        """
        elec = self.nwb.electrodes
        device_idx = elec['group_name'].data[:] == device
        z = elec['z'].data[device_idx]
        ch_ids = np.array([int(ch) for ch in elec['location'].data[device_idx]])
        
        sort_idx = np.argsort(z) # in mars, z coordinates are positive
        if reverse:
            return ch_ids[sort_idx][::-1], sort_idx[::-1], np.sort(z)[::-1]
        else:
            return ch_ids[sort_idx], sort_idx, np.sort(z)
        
    def write(self, save_path=None, time=False):
        tstart = datetime.now()
        self.io = NWBHDF5IO(save_path, 'w') if save_path else self.io
        self.io.write(self.nwb)
        if time:
            print('Write time for {}: {}s'.format(self.block_name,datetime.now()-tstart))

    def close(self):
        # check for self.io without throwing error
        if getattr(self, 'io'):
            self.io.close()


