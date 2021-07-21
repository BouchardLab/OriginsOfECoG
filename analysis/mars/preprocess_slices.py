"""Script to preprocess a 100um slice contributions file so it has
processed contributions and lesions """
import sys

from pynwb import NWBHDF5IO

from mars.io import NSENWB
from mars.wn import tokenize
from preprocess import _hilbert_transform, _get_cfs, _get_sds
from layer_reader import LayerReader

FIRST_RESAMPLE = 3200.0
FINAL_RESAMPLE = 400.0
BLOCK = 'Simulation_v1'
THICKNESS = 200
NSLICES = 11

if __name__ == '__main__':
    nwbfile = sys.argv[1]
    cfs = _get_cfs(None)
    sds = _get_sds(cfs, None)

    nsenwb = NSENWB.from_existing_nwb(BLOCK, nwbfile)
    reader = LayerReader(nsenwb.nwb)

    # full CSEP:
    X = reader.raw_full()
    X = _hilbert_transform(X, reader.raw_rate(), cfs, sds,
                           FIRST_RESAMPLE, FINAL_RESAMPLE)
    nsenwb.add_proc(X, 'ECoG', 'Hilb_54bands',
                    FINAL_RESAMPLE, cfs=cfs, sds=sds)

    for slice_i in range(NSLICES):
        # contrirbution:
        X = reader.raw_slice(slice_i, thickness=THICKNESS)
        X = _hilbert_transform(X, reader.raw_rate(), cfs, sds,
                               FIRST_RESAMPLE, FINAL_RESAMPLE)
        nsenwb.add_proc(X, 'ECoG', 'Hilb_54bands_{}'.format(slice_i),
                        FINAL_RESAMPLE, cfs=cfs, sds=sds)

        # lesion:
        X = reader.raw_slice_lesion(slice_i, thickness=THICKNESS)
        X = _hilbert_transform(X, reader.raw_rate(), cfs, sds,
                               FIRST_RESAMPLE, FINAL_RESAMPLE)
        nsenwb.add_proc(X, 'ECoG', 'Hilb_54bands_l{}'.format(slice_i),
                        FINAL_RESAMPLE, cfs=cfs, sds=sds)

    tokenize(nsenwb)
    nsenwb.write()
                
