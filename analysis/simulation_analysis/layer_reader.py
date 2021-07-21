from pynwb import NWBHDF5IO

class LayerReader(object):
    def __init__(self, nwb=None, nwbfile=None, device='ECoG',
                 raw_dset_name='Raw', proc_dset_name='Hilb_54bands'):
        """
        nwb = an ecp_layer_ei nwb file object
        nwbfile = pointer to an nwb file
        """

        if nwb:
            self.nwb = nwb
        elif nwbfile:
            self.nwbfile = nwbfile
            self.io = NWBHDF5IO(nwbfile, 'r')
            self.nwb = self.io.read()
        else:
            raise ValueError('Must specify either nwb or nwbfile')

        self.device = device
        self.raw_dset_name = raw_dset_name
        self.proc_dset_name = proc_dset_name

        self.raw_dset = self.nwb.acquisition[raw_dset_name].electrical_series[device]

    def raw_rate(self):
        return self.raw_dset.rate

    def raw_full(self):
        return self.raw_dset.data[:]

    def raw_contrib_dset_name(self, layer, ei):
        return '{}{}'.format(layer, ei)
    
    def raw_contrib(self, layer=None, ei=None):
        if layer and ei:
            return self.nwb.acquisition[self.raw_contrib_dset_name(layer, ei)] \
                           .electrical_series[self.device].data[:]
        elif layer and not ei:
            if layer == 1:
                return self.raw_contrib(1, 'i')
            return self.raw_contrib(layer, 'e') + self.raw_contrib(layer, 'i')
        elif ei and not layer:
            raise NotImplementedError("Total e/i across all layers not implemented yet")
        else:
            raise ValueError("Must specify layer or ei")

    def raw_lesion(self, layer=None, ei=None):
        return self.raw_full() - self.raw_contrib(layer, ei)

    def raw_slice(self, slice_i, thickness=100):
        if thickness == 100:
            return self.nwb.acquisition[str(slice_i)].electrical_series[self.device].data[:]
        elif thickness == 200:
            if slice_i == 10:
                return self.raw_slice(2*slice_i) # slice_i = 21 does not exist
            else:
                return self.raw_slice(2*slice_i) + self.raw_slice(2*slice_i + 1)
        else:
            raise ValueError("Can only take 100 or 200um slices")

    def raw_slice_lesion(self, slice_i, thickness=100):
        return self.raw_full() - self.raw_slice(slice_i, thickness=thickness)

    ###############################################################
    """Hilbert (processed) data, which must be preprocessed into
    contributions, not calculated on the fly. Most scripts that need
    this data will just grab it directly from the nwb, but here's a
    convenient way to do that"""
    ###############################################################

    @property
    def proc_dset(self):
        return self.nwb.modules[self.proc_dset_name].data_interfaces[self.device]
    
    def proc_rate(self):
        return self.proc_dset.rate

    def proc_contrib(self, layer=None, ei=None):
        if layer and ei:
            return self.get_proc_dset('{}_{}{}'.format(self.proc_dset_name, layer, ei))
        elif layer and not ei:
            return self.get_proc_dset('{}_{}'.format(self.proc_dset_name, layer))
        elif ei and not layer:
            raise NotImplementedError("Total e/i across all layers not implemented yet")
        else:
            raise ValueError("Must specify layer or ei")

    def proc_lesion(self, layer=None, ei=None):
        if layer and ei:
            return self.get_proc_dset('{}_l{}{}'.format(self.proc_dset_name, layer, ei))
        elif layer and not ei:
            return self.get_proc_dset('{}_l{}'.format(self.proc_dset_name, layer))
        elif ei and not layer:
            raise NotImplementedError("Total e/i across all layers not implemented yet")
        else:
            raise ValueError("Must specify layer or ei")
