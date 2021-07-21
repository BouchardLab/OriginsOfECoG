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

SLICE = 100
NSLICES = 21

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

    tr = np.zeros((nsites, nseg))

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

def _transfer_resistance_for(gid, coords_dir, pos):
    """
    Return a matrix whose elements are the corresponding elements from the transfer resistance
    for cells whose gid's are in `filt`, zero for other cells
    """
    with h5py.File(os.path.join(coords_dir, '{}.h5'.format(gid)), 'r') as coordsfile:
        nsites = pos.shape[1]
        nseg = coordsfile['p0'].shape[1]
        return _transfer_resistance(
            coordsfile['p0'][:], coordsfile['p05'][:], coordsfile['p1'][:],
            nsites=nsites,
            pos=pos
        ).T

def _depths_for(gid, coords_dir):
    with h5py.File(os.path.join(coords_dir, '{}.h5'.format(gid)), 'r') as coordsfile:
        return coordsfile['p05'][1, :].astype(np.double)

def get_transfer_resistances_and_depths(sim_jobnum, ordering, electrodes_file=None):
    # Read electrode file
    sim_network_dir = utils.get_network_dir(sim_jobnum)
    electrodes_file = electrodes_file or os.path.join(sim_network_dir, 'electrodes.csv')
    el_df = pd.read_csv(electrodes_file, sep=' ')
    el_pos = el_df[['x_pos', 'y_pos', 'z_pos']].T.values
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

    tr_by_cell = [(gid, _transfer_resistance_for(gid, coords_dir, el_pos)) for gid in cells]
    this_tr = np.concatenate([cell_tr for gid, cell_tr in tr_by_cell])
    depths = np.expand_dims(np.concatenate([_depths_for(gid, coords_dir) for gid in cells]), axis=-1)
    log.debug("calculated this rank's transfer matrix")

    if mpi_loaded:
        log.debug("about to run AllGatherv")
        log.debug("this_tr.shape = {}".format(this_tr.shape))
        full_tr = utils.AllGatherv_unknown_shape(this_tr, comm, rank, n_threads, mpi_dtype=MPI.DOUBLE, axis=1)
        log.debug("gathered tr")
        all_depths = utils.AllGatherv_unknown_shape(depths, comm, rank, n_threads, mpi_dtype=MPI.DOUBLE, axis=1, log=log).squeeze()
        log.debug("gathered depths")
    else:
        full_tr = this_tr
        all_depths = depths

    return full_tr, all_depths

def calc_ecp_chunked(im_dset, filtered_tr, chunksize, local_chunksize):
    i = rank
    
    def iter_im():
        """
        iterate over this rank's chunk in smaller chunks,
        whose size are given by local_chunksize
        """
        start, end = i*chunksize, (i+1)*chunksize
        for idx in range(start, end, local_chunksize):
            yield im_dset[idx:min(idx+local_chunksize,end), :]
    
    ecps = []
    for j, im in enumerate(iter_im()):
        log.info("calculating local chunk #{}".format(j))
        ecps.append(np.dot(im, filtered_tr))
    ecp = np.vstack(ecps)
    return ecp



def tmp_output_file_for(jobnum, groupname):
    output_dir = utils.get_output_dir(jobnum)
    return os.path.join(output_dir, 'ecp_{}.h5'.format(groupname))

    
def write_nwb(sim_jobnum, jobnum, n_timepts, groupnames, outfilename, block):
    """
    Write data from h5 files (needed for parallel) into .nwb
    and delete the files when done
    """

    nsenwb = NSENWB.from_block_name(block)
    mark_track = mark[nsenwb.stim['name']]
    mark_rate = 10000.
    nsenwb.add_mark(mark_track, mark_rate)

    all_h5_files = [(groupname, tmp_output_file_for(jobnum, groupname)) for groupname in groupnames]

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
                nsenwb.add_raw(ecog[:, np.newaxis], device_name='ECoG', acq_name=str(groupname))
                nsenwb.add_raw(ecog_i, device_name='ECoG_i', acq_name=str(groupname))
                nsenwb.add_raw(poly, device_name='Poly', acq_name=str(groupname))

            if ecog_tot is not None:
                ecog_tot += ecog
                ecog_i_tot += ecog_i
                poly_tot += poly
            else:
                ecog_tot = ecog.copy()
                ecog_i_tot = ecog_i.copy()
                poly_tot = poly.copy()

        log.info("Put group {} in nwb".format(groupname))

    nsenwb.add_raw(ecog_tot[:, np.newaxis], device_name='ECoG', acq_name='Raw')
    nsenwb.add_raw(ecog_i, device_name='ECoG_i', acq_name='Raw')
    nsenwb.add_raw(poly_tot, device_name='Poly', acq_name='Raw')

    nsenwb.write(outfilename)
    log.info("wrote nwb to {}".format(outfilename))

    # delete h5 files
    for groupname, fn in all_h5_files:
        log.info("deleting {}".format(fn))
        os.remove(fn)
    

def main(jobnum, array_task, sim_jobnum, chunksize, local_chunksize, outfile, block, electrodes_file=None):
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

    # Args for opening h5 file
    if mpi_loaded:
        kwargs = {'driver': 'mpio', 'comm': MPI.COMM_WORLD}
    else:
        kwargs = {}

    sim_output_dir = utils.get_output_dir(sim_jobnum)
    im_filename = os.path.join(sim_output_dir, 'im.h5')

    with h5py.File(im_filename, 'r', **kwargs) as infile:
        im_dset = infile['im/data'] if 'im' in infile.keys() else infile['data']
        ordering = infile['mapping/gids'][:]
        n_timepts_im, n_seg = im_dset.shape
        n_timepts_tot = chunksize * n_threads

        log.info("Got dset and ordering")

        full_tr, depths = get_transfer_resistances_and_depths(sim_jobnum, ordering, electrodes_file=electrodes_file)
        log.info("Got transfer resistances")

        all_slices = [array_task] if array_task is not None else range(NSLICES)

        for slice_i in all_slices:
            # slice is depths [i*100, (i+1)*100] um below surface
            log.info("Doing calculation for slice {}".format(slice_i))

            # Zero out out-of-slice segments in the transfer matrix
            mask = ((2083-depths) // SLICE) == slice_i
            mask = np.tile(mask, (full_tr.shape[1], 1)).T
            filtered_tr = full_tr.copy()
            np.place(filtered_tr, np.logical_not(mask), 0)
            
            outfn = tmp_output_file_for(jobnum, str(slice_i))
            with h5py.File(outfn, 'w', **kwargs) as out:
                ecp = calc_ecp_chunked(im_dset, filtered_tr, chunksize, local_chunksize)
                n_timepts_chunk, n_ch = ecp.shape

                out.create_dataset('data', shape=(n_timepts_tot, n_ch), dtype=np.float)
                start = i*chunksize
                stop = min( (i+1)*chunksize, n_timepts_tot )
                log.info("start, stop, sum(ecp) = {}, {}, {}".format(start, stop, np.sum(ecp)))

                if start < stop:
                    out['data'][start:stop, :] = ecp

            log.info("Done with calculation for slice {}".format(slice_i))

    if mpi_loaded:
        comm.Barrier()

    if i == 0 and array_task is None:
        # TODO: Compute a default value for outfile?
        log.info("Rank 0 about to write nwb")
        write_nwb(sim_jobnum, jobnum, n_timepts_tot, range(NSLICES), outfile, block)

    log.info("COMPLETED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobnum', type=str, required=True, help='slurm number of this calc_ecp job')
    parser.add_argument('--array-task', type=int, required=False, default=None,
                        help='if passed, runs one slice per array task. Must be aggregated to nwb separately.')
    parser.add_argument('--sim-jobnum', type=str, required=True, help='slurm number of simulation job')
    parser.add_argument('--block', type=str, required=True)
    
    parser.add_argument('--electrodes-file', '--electrode-file', type=str, required=False, default=None)
    parser.add_argument('--output-dir', type=str, required=False, default=None)
    parser.add_argument('--outfile', type=str, required=False, default='ecp.nwb')
    parser.add_argument('--local-chunksize', type=int, default=100)

    # Parallelization flags
    parser.add_argument('--chunksize', type=int, default=1000,
                        help="number of timepoints to calculate in each chunk")


    args = parser.parse_args()

    main(args.jobnum, args.array_task, args.sim_jobnum, args.chunksize, args.local_chunksize, args.outfile, args.block)
