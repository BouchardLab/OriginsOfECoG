"""Create an json file listing the number of segments from each layer
in each 100um disk.
"""
from __future__ import print_function
import os
import argparse
import json
from collections import defaultdict

import h5py

import utils

def get_layer_slice_outfile(jobnum):
    return os.path.join('runs', jobnum, 'output', 'layer_slice_counts.json')

def main(jobnum, outfile):
    outfile = outfile or get_layer_slice_outfile(jobnum)
    print("Counting layer segments in each slice")
    coords_dir = os.path.join(utils.get_output_dir(jobnum), 'seg_coords')
    layer_slice_counts = {layer: defaultdict(int) for layer in [1, 2, 3, 4, 5, 6]}
    # layer_slice_counts['JOBNUM'] = jobnum
    for layer, gids in utils.get_layer_gids(jobnum).items():
        for gid in gids:
            cellfile = os.path.join(coords_dir, '{}.h5'.format(gid))
            with h5py.File(cellfile, 'r') as coordsfile:
                depths = coordsfile['p05'][1, :]
                for slice_i in range(-2, 23):
                    num_in_slice = int(sum( ((2083-depths) // 100) == slice_i )) # cast to int b/c uses np.int64 on Cori, which is not json serializable
                    layer_slice_counts[layer][slice_i] += num_in_slice
        print("Done layer {}".format(layer))

    with open(outfile, 'w') as outf:
        print(json.dumps(layer_slice_counts, sort_keys=True, indent=4), file=outf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobnum', type=str, required=True, help='slurm number of simulation job')
    parser.add_argument('--outfile', type=str, required=False, default=None)

    args = parser.parse_args()

    main(args.jobnum, args.outfile)
