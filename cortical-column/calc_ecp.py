"""
Calculate ECP from im.h5, a file containing the membrane current for all sections at all timesteps
"""

import math
import os
import argparse
import logging

import pandas as pd
import numpy as np
import h5py

import utils
from stimulus import mark
from mars.io import NSENWB

log = logging.getLogger('calc_ecp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

if h5py.get_config().mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_threads = comm.Get_size()
    if n_threads > 1:
        mpi_loaded = True
        log.info("Using MPI")
    else:
        mpi_loaded = False
        log.info("Not using MPI, only 1 thread")
else:
    rank = 0
    comm = None
    n_threads = 1
    mpi_loaded = False
    log.info("Not using MPI")

log.info('Rank = {}'.format(rank))

def _transfer_resistance(p0, p05, p1, nsites, pos):
    sigma = 0.3  # mS/mm

    r05 = (p0 + p1) / 2
    dl = p1 - p0

    nseg = r05.shape[1]

    tr = np.zeros((nsites, nseg), dtype=np.float32)

    for j in range(nsites):  # calculate mapping for each site on the electrode
        rel = np.expand_dims(pos[:, j], axis=1)  # coordinates of a j-th site on the electrode
        rel_05 = rel - r05  # distance between electrode and segment centers

        # compute dot product column-wise, the resulting array has as many columns as original
        r2 = np.einsum('ij,ij->j', rel_05, rel_05)

        # compute dot product column-wise, the resulting array has as many columns as original
        rlldl = np.einsum('ij,ij->j', rel_05, dl)
        dlmag = np.linalg.norm(dl, axis=0)  # length of each segment
        rll = abs(rlldl / dlmag)  # component of r parallel to the segment axis it must be always positive
        rT2 = r2 - rll ** 2  # square of perpendicular component
        up = rll + dlmag / 2
        low = rll - dlmag / 2
        num = up + np.sqrt(up ** 2 + rT2)
        den = low + np.sqrt(low ** 2 + rT2)
        tr[j, :] = np.log(num / den) / dlmag  # units of (um) use with im_ (total seg current)
        np.copyto(tr[j, :], 0, where=(dlmag == 0)) # zero out stub segments

    tr *= 1 / (4 * math.pi * sigma)
    return tr

def _transfer_resistance_for(gid, coords_dir, pos, filt=None):
    """
    Return a matrix whose elements are the corresponding elements from the transfer resistance
    for cells whose gid's are in `filt`, zero for other cells

    filt = set of contributing gids
    """
    with h5py.File(os.path.join(coords_dir, '{}.h5'.format(gid)), 'r') as coordsfile:
        nsites = pos.shape[1]
        nseg = coordsfile['p0'].shape[1]
        if (filt is None) or (gid in filt):
            return _transfer_resistance(
                coordsfile['p0'][:], coordsfile['p05'][:], coordsfile['p1'][:],
                nsites=nsites,
                pos=pos
            ).T
        else:
            log.info('Found cell not in group, zeroing')
            return np.zeros(shape=(nseg, nsites), dtype=np.float32)

def get_pos(electrodes_file=None, new_electrodes_file=None):
    if new_electrodes_file:
        pos = None
    else:
        electrodes_file = electrodes_file or os.path.join(sim_network_dir, 'electrodes.csv')
        el_df = pd.read_csv(electrodes_file, sep=' ')
        pos = el_df[['x_pos', 'y_pos', 'z_pos']].T.values
        
    return pos

def get_transfer_resistances_and_gids(sim_jobnum, ordering, electrodes_file=None, new_electrodes_file=None):
    # Read electrode file
    sim_network_dir = utils.get_network_dir(sim_jobnum)
    pos = get_pos(electrodes_file, new_electrodes_file)
    coords_dir = os.path.join(utils.get_output_dir(sim_jobnum), 'seg_coords')

    if mpi_loaded:
        # Distributed load of cell coords/transfer matrix
        n_cells = len(os.listdir(coords_dir))
        n_each = int(n_cells / n_threads) + 1
        log.info("Using MPI for tr calc, with n_cells = {}, n_each = {}".format(n_cells, n_each))
        start = n_each * rank
        stop = min(n_each * (rank+1), len(ordering))
        cells = ordering[start:stop]
        
        log.info("This rank's {} cells: {}".format(len(cells), cells))
    else:
        cells = ordering
        log.info("Not using MPI for tr calc")

    # np array with all cells gid and transfer matrices
    tr_by_cell = [(gid, _transfer_resistance_for(gid, coords_dir, pos)) for gid in cells]
    gids = np.concatenate([[gid] * cell_tr.shape[0] for gid, cell_tr in tr_by_cell])
    this_tr = np.concatenate([cell_tr for gid, cell_tr in tr_by_cell])
    log.debug("calculated this rank's transfer matrix")

    # Replicate gids for each channel
    gids = np.tile(gids, (this_tr.shape[1], 1)).T.astype('intc')

    if mpi_loaded:
        log.debug("about to run AllGatherv")
        log.debug("this_tr.shape = {}".format(this_tr.shape))
        full_tr = utils.AllGatherv_unknown_shape(this_tr, comm, rank, n_threads, mpi_dtype=MPI.FLOAT, axis=1, log=log).astype(np.float32)
        log.debug("gathered tr")
        all_gids = utils.AllGatherv_unknown_shape(gids, comm, rank, n_threads, mpi_dtype=MPI.INT, axis=1)
        log.debug("gathered gids")
    else:
        full_tr = this_tr.astype(np.float32)
        all_gids = gids

    return full_tr, all_gids
    

def filter_transfer_matrix(tr, gids, filt):
    """ Zero out some cells in the transfer matrix """
    if filt is None:
        return tr
    if isinstance(filt, set):
        filt = list(filt)
    filtered_tr = tr.copy().astype(np.float32)
    mask = np.isin(gids, filt, invert=True)
    np.place(filtered_tr, mask, 0)
    return filtered_tr

def calc_ecp_chunked(im_dset, tr, chunksize, local_chunksize):
    i = rank

    chunk_start, chunk_end = i*chunksize, (i+1)*chunksize
    n_segs, n_electrodes = tr.shape
    
    ecp = np.empty(shape=(chunksize, n_electrodes), dtype=np.float32)
    for j, (lcl_chunk_start, output_start) in enumerate(zip(
            range(chunk_start, chunk_end, local_chunksize),
            range(0, chunksize, local_chunksize))):
        log.info("calculating local chunk #{}".format(j))
        lcl_chunk_end = min(lcl_chunk_start+local_chunksize, chunk_end)
        output_end = output_start + (lcl_chunk_end - lcl_chunk_start)
        im = im_dset[lcl_chunk_start:lcl_chunk_end, :]
        ecp[output_start:output_end, :] = np.dot(im, tr)
    
    return ecp




def get_groupings(sim_jobnum, groupby, array_task):
    # groups = dict mapping group name to list of GIDs in that group (or None, for all GIDs)
    if groupby is None:
        groups = {'all': None}
    elif groupby == 'layer':
        layer_gids = utils.get_layer_gids(sim_jobnum)
        groups = {'L{}'.format(layer): (layer_gids.get(layer, set())) for layer in range(1, 7)}
    elif groupby == 'layer_ei':
        groups = utils.get_layer_ei_gids(sim_jobnum)
    elif groupby == 'parts':
        log.error("DO NOT USE calc_ecp.py FOR PARTS; USE calc_ecp_parts.py")
        groups = {i: [] for i in range(4)}
    elif groupby == '100um':
        log.error("DO NOT USE calc_ecp.py FOR 100um SLICES; USE calc_ecp_100um.py")
        planesize = float(groupby[:-2])
        groups = utils.get_spatial_chunk_gids(sim_jobnum, planesize)
    elif groupby.endswith('um_cell'):
        planesize = float(groupby[:-7])
        groups = utils.get_spatial_chunk_gids(sim_jobnum, planesize)
    else:
        log.info("Unrecognized groupby: {}. Not grouping")
        groups = {'all': None}

    if array_task:
        k = sorted(groups.keys())[array_task]
        groups = {k: groups[k]}
        
    return groups


def tmp_output_file_for(jobnum, groupname, tmpdir=None):
    output_dir = tmpdir or utils.get_output_dir(jobnum)
    return os.path.join(output_dir, 'ecp_{}.h5'.format(groupname))

    
def write_nwb(sim_jobnum, jobnum, n_timepts, groupnames, outfilename, block, tmpdir=None):
    """
    Write data from h5 files (needed for parallel) into .nwb
    and delete the files when done
    """

    nsenwb = NSENWB.from_block_name(block)
    mark_track = mark[nsenwb.stim['name']]
    mark_rate = 10000.
    nsenwb.add_mark(mark_track, mark_rate)

    all_h5_files = [(groupname, tmp_output_file_for(jobnum, groupname, tmpdir=tmpdir)) for groupname in groupnames]

    # add raw datasets
    ecog_tot = None
    ecog_i_tot = None
    poly_tot = None
    for groupname, fn in all_h5_files:
        with h5py.File(fn, 'r') as infile:
            log.info("loaded {} for writing to nwb".format(fn))
            ecp_dset = infile['data']
            ecog = np.average(ecp_dset[:, :100], axis=1) # Average over 4 electrodes
            ecog_i = ecp_dset[:, :100] # Store individual also
            poly = ecp_dset[:, 100:]

            if len(all_h5_files) > 1: # only acquisition is 'all', equivalent to 'Raw' added below
                nsenwb.add_raw(ecog[:, np.newaxis], 'ECoG')
                # nsenwb.add_raw(ecog_i, device_name='ECoG_i', acq_name=str(groupname))
                nsenwb.add_raw(poly, 'Poly')

            if ecog_tot is not None:
                ecog_tot += ecog
                # ecog_i_tot += ecog_i
                poly_tot += poly
            else:
                ecog_tot = ecog.copy()
                # ecog_i_tot = ecog_i.copy()
                poly_tot = poly.copy()

        log.info("Put group {} in nwb".format(groupname))

    nsenwb.add_raw(ecog_tot[:, np.newaxis], 'ECoG')
    # nsenwb.add_raw(ecog_i, device_name='ECoG_i', acq_name='Raw')
    nsenwb.add_raw(poly_tot, 'Poly')

    nsenwb.write(outfilename)
    log.info("wrote nwb to {}".format(outfilename))

    # delete h5 files
    for groupname, fn in all_h5_files:
        log.info("deleting {}".format(fn))
        os.remove(fn)
    

def main(jobnum, array_task, sim_jobnum, chunksize, local_chunksize, groupby, outfile, block, electrodes_file=None, new_electrodes_file=None, tmp_output_dir=None, im_file=None):
    """
    Calculate ecp for this rank's chunk, write into the outfile

    jobnum: slurm number of this job
    sim_jobnum: where to find im.h5
    chunksize: size of each rank's chunk
    local_chunksize: step size within this chunk
    groupby: split the ECP calculation by contribution.
             options: 'all', 'layer', 'layer_ei'
    """
    i = rank

    log.info("Starting ECP calculation for job {}".format(sim_jobnum))

    groups = get_groupings(sim_jobnum, groupby, array_task)

    # Args for opening h5 file
    if mpi_loaded:
        kwargs = {'driver': 'mpio', 'comm': MPI.COMM_WORLD}
    else:
        kwargs = {}

    sim_output_dir = utils.get_output_dir(sim_jobnum)
    im_filename = im_file or os.path.join(sim_output_dir, 'im.h5')
    log.info("Using {}".format(im_filename))

    with h5py.File(im_filename, 'r', **kwargs) as infile:
        im_dset = infile['im/data'] if 'im' in infile.keys() else infile['data']
        ordering = infile['mapping/gids'][:]
        n_timepts_im, n_seg = im_dset.shape
        n_timepts_tot = chunksize * n_threads

        log.info("Got dset and ordering")

        full_tr, gids = get_transfer_resistances_and_gids(sim_jobnum, ordering, electrodes_file=electrodes_file, new_electrodes_file=new_electrodes_file)
        log.info("Got transfer resistances")

        for groupname, groupfilt in groups.items():
            log.info("Doing calculation for group {}".format(groupname))

            filtered_tr = filter_transfer_matrix(full_tr, gids, groupfilt) 
            log.info("Filtered transfer matrix for group {}".format(groupname))

            outfn = tmp_output_file_for(jobnum, groupname, tmpdir=tmp_output_dir)
            log.info("Opening {}".format(outfn))
            with h5py.File(outfn, 'w', **kwargs) as out:
                ecp = calc_ecp_chunked(im_dset, filtered_tr, chunksize, local_chunksize)
                n_timepts_chunk, n_ch = ecp.shape

                out.create_dataset('data', shape=(n_timepts_tot, n_ch), dtype=np.float)
                start = i*chunksize
                stop = min( (i+1)*chunksize, n_timepts_tot )
                log.info("start, stop, sum(ecp) = {}, {}, {}".format(start, stop, np.sum(ecp)))

                if start < stop:
                    out['data'][start:stop, :] = ecp

            del filtered_tr

            log.info("Done with calculation for group {}".format(groupname))

    if mpi_loaded:
        comm.Barrier()

    if i == 0 and array_task is None:
        # TODO: Compute a default value for outfile?
        log.info("Rank 0 about to write nwb")
        write_nwb(sim_jobnum, jobnum, n_timepts_tot, groups.keys(), outfile, block, tmpdir=tmp_output_dir)

    log.info("COMPLETED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobnum', type=str, required=True, help='slurm number of this calc_ecp job')
    parser.add_argument('--array-task', type=int, required=False, default=None,
                        help='if passed, runs one group per array task. Must be aggregated to nwb separately.')
    parser.add_argument('--sim-jobnum', type=str, required=True, help='slurm number of simulation job')
    parser.add_argument('--block', type=str, required=True)
    
    parser.add_argument('--electrodes-file', '--electrode-file', type=str, required=False, default=None)
    parser.add_argument('--new-electrodes-file', '--electrode-file', type=str, required=False, default=None)
    parser.add_argument('--tmp-output-dir', type=str, required=False, default=None,
                        help='If passed, will write temp output files here')
    parser.add_argument('--im-file', type=str, required=False, default=None,
                        help='If passed, will use this instead of --sim-jobnum to find im.h5')
    parser.add_argument('--outfile', type=str, required=False, default='ecp.nwb')
    parser.add_argument('--groupby', type=str, default=None)
    parser.add_argument('--local-chunksize', type=int, default=100)

    # Parallelization flags
    parser.add_argument('--chunksize', type=int, default=1000,
                        help="number of timepoints to calculate in each chunk")


    args = parser.parse_args()

    main(args.jobnum, args.array_task, args.sim_jobnum, args.chunksize, args.local_chunksize, args.groupby, args.outfile, args.block, electrodes_file=args.electrodes_file, new_electrodes_file=args.new_electrodes_file, tmp_output_dir=args.tmp_output_dir, im_file=args.im_file)
