import socket
import functools
import os
import copy
from collections import defaultdict

import h5py
import pandas as pd
import numpy as np

def iter_populations():
    """
    Yield:

    1, i, 1i, 1i
    2, i, 2i, 23i
    2, e, 2e, 23e
    3, i, 3i, 23i
    3, e, 3e, 23e
    4, i, 4i, 4i
    4, e, 4e, 4e
    5, i, 5i, 5i
    5, e, 5e, 5e
    6, i, 6i, 6i
    6, e, 6e, 6e
    """
    for layer, layername in [(1,'1'), (2,'23'), (3,'23'),
                             (4,'4'), (5,'5'), (6,'6')]:
        yield layer, 'i', '{}{}'.format(layer, 'i'), '{}{}'.format(layername, 'i')

        if layer != 1:
            yield layer, 'e', '{}{}'.format(layer, 'e'), '{}{}'.format(layername, 'e')

EXC_M_TYPES = ('PC', 'SS', 'SP', 'STPC', 'UTPC', 'TTPC1', 'TTPC2', 'TPC_L1', 'UTPC', 'TPC_L4', 'BPC', 'IPC')

def node_populations_by_name(graph):
    """
    Return a dict that maps population names to SonataNodes objects

    graph: bmtk.BioNetwork instance
    """
    return {nodepop.name: nodepop for nodepop in graph.node_populations}

def nodes_by_layer_ei(cortex):
    """
    Return a dict that maps layer number + ei to a list of nodes in that layer
    eg mapping['4e'] gives a list of layer 4 excitatory cells

    cortex: bmtk SonataNodes object representing the cortical population
    """
    mapping = defaultdict(list)
    for node in cortex.get_nodes():
        layer_ei = "{}{}".format(node['layer'], node['ei'])
        mapping[layer_ei].append(node)
    return mapping

def get_gids(cells_file, population):
    cells_h5 = h5py.File(cells_file, 'r')
    return cells_h5['/nodes/{}/node_id'.format(population)][:]

def get_nodes_df(cells_file, cell_models_file, population=None):
    cm_df = pd.read_csv(cell_models_file, sep=' ')
    cm_df.set_index('node_type_id', inplace=True)

    cells_h5 = h5py.File(cells_file, 'r')
    # TODO: Use sonata api
    if population is None:
        if len(cells_h5['/nodes']) > 1:
            raise Exception('Multiple populations in nodes file. Please specify one to plot using population param')
        else:
            population = list(cells_h5['/nodes'].keys())[0]

    nodes_grp = cells_h5['/nodes'][population]
    c_df = pd.DataFrame({'node_id': nodes_grp['node_id'], 'node_type_id': nodes_grp['node_type_id'], 'depth': nodes_grp['0']['positions'][:, 1]})
    # c_df = pd.read_csv(cells_file, sep=' ')
    c_df.set_index('node_id', inplace=True)
    nodes_df = pd.merge(left=c_df,
                        right=cm_df,
                        how='left',
                        left_on='node_type_id',
                        right_index=True)  # use 'model_id' key to merge, for right table the "model_id" is an index

    return nodes_df

def get_layer_gids(jobnum, old_dir=False):
    """
    Return a dict mapping layer num to a set of gids of all cells in that layer
    """
    network_dir = get_network_dir(jobnum, old_dir)
    cells_file = os.path.join(network_dir, 'cortical_column_nodes.h5')
    cell_models_file = os.path.join(network_dir, 'cortical_column_node_types.csv')
    nodes_df = get_nodes_df(cells_file, cell_models_file)
    return {name: set(int(gid) for gid in group.index) for name, group in nodes_df.groupby('layer')}

def get_layer_ei_gids(jobnum, old_dir=False):
    """
    Return a dict mapping layer num + e/i to a set of gids of all cells in that layer/ei
    """
    network_dir = get_network_dir(jobnum, old_dir)
    cells_file = os.path.join(network_dir, 'cortical_column_nodes.h5')
    cell_models_file = os.path.join(network_dir, 'cortical_column_node_types.csv')
    nodes_df = get_nodes_df(cells_file, cell_models_file)
    return {'{}{}'.format(layer, ei): set(int(gid) for gid in group.index) for (layer, ei), group in nodes_df.groupby(['layer', 'ei'])}

def get_spatial_chunk_gids(jobnum, chunksize, old_dir=False):
    """
    Return a dict mapping chunk top edge to list of GIDs in that chunk
    """
    network_dir = get_network_dir(jobnum, old_dir)
    cells_file = os.path.join(network_dir, 'cortical_column_nodes.h5')
    cell_models_file = os.path.join(network_dir, 'cortical_column_node_types.csv')
    nodes_df = get_nodes_df(cells_file, cell_models_file)

    _ret = defaultdict(set)
    for index, row in nodes_df.iterrows():
        _ret[int(-row['depth']/chunksize)].add(row.name)

    return _ret

part_ids_enum = {'soma': 0, 'dend': 1, 'apic': 2, 'basal': 3, 'axon': 4}
part_names_by_id = {v:k for k, v in part_ids_enum.items()}

def spikes_table(config_file=None, spikes_file=None):
    #replaces bmtk.analyzer.spikes_table (in __init__)
    if config_file:
        config = _get_config(config_file)
        spikes_file = config['output']['spikes_file']
    elif spikes_file:
        spikes_h5 = h5py.File(spikes_file, 'r')
        gids = np.array(spikes_h5['/spikes/gids'], dtype=np.uint)
        times = np.array(spikes_h5['/spikes/timestamps'], dtype=np.float)
        return pd.DataFrame(times, index=gids, columns=['times'])
        # return pd.DataFrame(data={'gid': gids, 'spike time (ms)': times})
    else:
        raise ValueError("Must pass in either config_file or spikes_file")

DATAROOT = os.environ.get('CCDATAROOT', './runs')

def get_rundir(jobnum, old_dir=False):
    if jobnum in ('local', 'lcl'):
        return '.'
    if old_dir:
        return os.path.join('runs', jobnum)
    else:
        return os.path.join(DATAROOT, jobnum)

def get_network_dir(jobnum, old_dir=False):
    if jobnum in ('local', 'lcl'):
        return 'network'
    return os.path.join(get_rundir(jobnum, old_dir), 'network')

def get_output_dir(jobnum, old_dir=False):
    if jobnum in ('local', 'lcl'):
        return 'output'
    return os.path.join(get_rundir(jobnum, old_dir), 'output')

def get_config_file(jobnum, old_dir=False):
    return os.path.join(get_rundir(jobnum, old_dir), 'config.json')

def get_stim_intervals_file(jobnum, old_dir=False):
    return os.path.join(get_network_dir(jobnum, old_dir), 'stim_intervals.json')

# Return a decorator that runs the function only if rank == 0,
# otherwise waits for a broadcast from rank 0
def broadcast_from_zero(rank, comm, root=0):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if comm is None:
                return func(*args, **kwargs)
            
            if rank == 0:
                val = func(*args, **kwargs)
            else:
                val = None
            comm.bcast(val, root=root)
            return val
        return wrapped
    return decorator

def AllGatherv_unknown_shape(data, comm, rank, n_threads, mpi_dtype, axis=1, log=None):
    """
    Use MPI to compute shape of result array by asking every thread for their chunk's shape,
    then perform the AllGatherv operation (concatenating along the given axis) and return the result.
    """
    if log:
        log.debug("IN ALLGATHERV")
        
    if axis != 1:
        data = np.swapaxes(data, axis, 1)

    # Require row-major order for concatenation along axis 0
    data = np.require(data, requirements='C')

    dim_1_len = data.shape[1]

    if log:
        log.debug("  IN ALLGATHERV data.size = {}".format(data.size))

    # Gather size of each chunk
    sizes = comm.allgather(int(data.size))
    if log:
        log.debug("  IN ALLGATHERV sizes = {}".format(sizes))
    offsets = np.zeros(len(sizes), dtype=np.int)
    offsets[1:] = np.cumsum(sizes)[:-1]

    comm.Barrier()

    # Create big array
    tot_size = sum(sizes)
    full_shape = int(tot_size/dim_1_len), dim_1_len
    if log:
        log.debug("  IN ALLGATHERV full_shape = {}".format(full_shape))
        log.debug("  IN ALLGATHERV data.dtype = {}".format(data.dtype))
    all_data = np.zeros(shape=full_shape, dtype=data.dtype, order='C')
    
    # Perform AllGatherv
    comm.Allgatherv(data, [all_data, sizes, offsets, mpi_dtype])

    if axis != 1:
        all_data = np.swapaxes(all_data, axis, 1)

    return all_data

def log_spaced_cfs(fmin, fmax, nbin=6):
    """
    Center frequencies that are uniform in log space
    """
    noct = np.ceil(np.log2(fmax/fmin))
    return fmin * 2**(np.arange(noct*nbin)/nbin)

wavelet_cfs = log_spaced_cfs(2.6308, 1200.0)
