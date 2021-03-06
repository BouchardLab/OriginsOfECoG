import sys
import os
import argparse
import json
try: 
    from collections.abc import OrderedDict, Mapping
except:
    from collections import OrderedDict, Mapping

import utils
from stimulus import stimulus, save_csv

"""
1. Take base_config.json, apply defaults in ./host_config.json
and anything specified here

Use case: append host/jobid to network and output directories
(see run.sh)

2. Generate spike trains for the stimulus passed in
"""

def get_cells(args):
    cells = args.cells
    if cells is not None and 'all' in cells:
        return 'all'
    if args.cellrange:
        cells.extend(range(*args.cellrange))
        cells = sorted(list(set(cells)))

    return cells
            

# def validate(args):
#     if args.overwrite:
#         for chosen_dir in (args.base_dir, args.network_dir, args.output_dir):
#             if os.path.exists(chosen_dir):
#                 print("directory {} already exists".format(chosen_dir))
#                 print("pass --overwrite to continue")
#                 sys.exit(-1)

#     if args.customkeys or args.customvals:
#         if len(args.customkeys) != len(args.customvals):
#             print("--customkeys and --customvals must have same length")
#             sys.exit(-2)

            
def echo(args):
    print("network directory: {}".format(args.network_dir))
    print("output directory: {}".format(args.output_dir))
    if args.cells or args.cellrange:
        cells = get_cells(args)
        print("{} cells to save: {}".format(len(cells), cells))


def update_two_layers(_dict, updates):
    for k, v in updates.items():
        if isinstance(v, Mapping):
            _dict[k].update(v)
        else:
            _dict[k] = v


def update_config(
        base_config, host_config, **kwargs
        # network_dir=None,
        # output_dir=None,
        # sections=None,
        # dt=None,
        # tstop=None,
        # nsteps_block=None,
        # no_ecp=False,
):
    config = {}
    config.update(base_config)
    update_two_layers(config, host_config)
    config['__COMMENT__'] = "Auto-generated by configure.py"

    if kwargs.get('network_dir'):
        config['manifest']['$NETWORK_DIR'] = kwargs['network_dir']
    if kwargs.get('output_dir'):
        config['manifest']['$OUTPUT_DIR'] = kwargs['output_dir']

    # report_args = ('sections',)
    # report_params = {arg: kwargs.get(arg) for arg in report_args if kwargs.get(arg) is not None}
    # if kwargs.get('cells') is not None:
    #     report_params['cells'] = kwargs.get('cells')
    # for report in config['reports'].values():
    #     report.update(report_params)

    for arg in ('dt', 'tstop', 'nsteps_block', 'optocell'):
        if kwargs.get(arg) is not None:
            config['run'][arg] = kwargs.get(arg)

    if kwargs.get('no_ecp'):
        config['run']['calc_ecp'] = False

    return sorted_keys_first(config, '__COMMENT__', 'manifest')


def sorted_keys_first(_dict, *keys_first):
    """
    Return copy of `_dict` as an OrderedDict in sorted by key,
    but with the given keys first
    """
    return OrderedDict(
        [(k, _dict[k]) for k in keys_first] +
        sorted([x for x in _dict.items() if x[0] not in keys_first])
    )


def generate_spikes(stim, network_dir, output_dir):
    for netwk in ('thalamus', 'bkg'):
        cells_file = os.path.join(network_dir, '{}_nodes.h5'.format(netwk))
        cell_models_file = os.path.join(network_dir, '{}_node_types.csv'.format(netwk))
        gids = utils.get_gids(cells_file, netwk)
        save_csv(
            stimulus[stim][netwk],
            gids,
            os.path.join(network_dir, '{}_spikes.csv'.format(netwk))
        )


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-outfile', type=str, default='config.json',
                        help="name of config file this script will output")
    parser.add_argument('--base-config', type=str, default='base_config.json',
                        help="name of file containing base config options")
    parser.add_argument('--host-config', type=str, default='host_config.json',
                        help="name of file containing host-specific config options")
    parser.add_argument('--network-dir', '--network', type=str, default='network',
                        help="network dir the simulation will be configured to use")
    parser.add_argument('--output-dir', type=str, default='output',
                        help="output dir the simulation will be configured to write")
    parser.add_argument('--stim', '--stimulus', type=str, default='wn_simulation_v0',
                        help='named stimulus from mars block directory') # Might want to take in the whole block name, which has the stimulus name within
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help="cells to save")
    parser.add_argument('--cellrange', type=int, nargs='+', default=None,
                        help="start end [step] for range of cell ids to save")
    # parser.add_argument('--sections', type=str, nargs='+', default=None,
    #                     help="sections to save from each cell (soma, apical, basal, etc.)")
    parser.add_argument('--nsteps-block', '--block', '--blocksize', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help="don't complain about overwriting dirs")
    parser.add_argument('--dt', '--timestep', type=float, default=None)
    parser.add_argument('--tstop', type=float, default=None)
    parser.add_argument('--no-ecp', action='store_true', default=None)
    parser.add_argument('--optocell', type=str, nargs='+', default=None, 
                        help='[celltype][layer] groups of cells to which opto channels are added')
    # parser.add_argument('--customkeys', type=str, nargs='+', default=[],
    #                     help="jsonpath expressions locating items to write. " +
    #                     "To be used with --customvals")
    # parser.add_argument('--customvals', type=str, nargs='+', default=[],
    #                     help="values to write. To be used with --customkeys")
    args = parser.parse_args()

    echo(args)

    with open(args.base_config) as base_cfg_file:
        base_config = json.load(base_cfg_file)
    if os.path.exists(args.host_config):
        with open(args.host_config) as host_cfg_file:
            host_config = json.load(host_cfg_file)
    else:
        host_config = {}

    cells = get_cells(args)

    newconfig = update_config(
        base_config, host_config, cells=cells,
        network_dir=args.network_dir,
        output_dir=args.output_dir,
        # sections=args.sections,
        dt=args.dt,
        tstop=args.tstop,
        nsteps_block=args.nsteps_block,
        no_ecp=args.no_ecp,
        optocell=args.optocell
    )

    with open(args.config_outfile, 'w') as outf:
        json.dump(newconfig, outf, indent=4)

    generate_spikes(args.stim, args.network_dir, args.output_dir)
    

    
