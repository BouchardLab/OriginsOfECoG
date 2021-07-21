import os
import itertools
import argparse
import json
import logging
from collections import OrderedDict

import numpy as np
import h5py

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.aux.node_params import xiter_random

import utils
from layers import layer_depths
from build_network import generate_electrodes
from cells import cells, m_type_from_layer_m_type

log = logging.getLogger('BBP_build_network')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
log.addHandler(ch)



NUM_BBP_CELLS = 31346
# NUM_BBP_L4_CELLS = 4656 # sum of m-type counts in layer_download.json
NUM_BBP_L4_CELLS = 6081 # sum of e-type counts in layer_download.json

E_TYPE_KEY = "No. of neurons per electrical types"
connections_processed = 0
BBP_conn_cache = {}

def divide_choice(choice, categories, p=None):
    """
    Divde a list into different categories according to some probability
    """
    selected_categories = np.random.choice(
        categories, replace=True, size=len(choice), p=p)
    return {category: [choice[i] for i, _categ in enumerate(selected_categories) if _categ == category]
            for category in categories}

def get_e_type_fractions(popname, layers_json):
    layer_name = popname.split('_')[0]
    layer_etype_counts = layers_json[layer_name][E_TYPE_KEY] # etypes in this layer
    these_etypes = cells[popname].keys() # etypes for this mtype
    # e_type_counts = {k:v for k, v in layer_etype_counts.items() if k in these_etypes}
    e_type_counts = {k:v for (k, v) in layer_etype_counts.items() if k in these_etypes}
    return OrderedDict(( e_type, float(count) / sum(e_type_counts.values()) )
                       for e_type, count in e_type_counts.items())

def iter_populations_split_23(circuitfile):
    """
    circuitfile = open BBP circuit file (args.circuit_file)

    Iterate over each population, returning population name and all data, with
    layers 2/3 separated out into separate return values
    """
    for popname, popinfo in circuitfile['populations'].items():
        if popname[:3] == 'L23':
            l23_cutoff = 2082 - (165+149)
            l2_idx = (popinfo['locations'][:, 1] > l23_cutoff)
            l3_idx = np.logical_not(l2_idx)
            if np.any(l2_idx):
                yield popname, {k: d[l2_idx, :] for k, d in popinfo.items()}, '2'
            if np.any(l3_idx):
                yield popname, {k: d[l3_idx, :] for k, d in popinfo.items()}, '3'
        else:
            yield popname, popinfo, popname[1]
    

def build_BBP_network(args):
    log.info("Creating {} cells".format(int(NUM_BBP_CELLS * args.reduce)))
    circuit_filename = os.path.basename(args.circuit_file)[:-3]
    # net = NetworkBuilder('BBP_{}'.format(circuit_filename))
    net = NetworkBuilder('cortical_column')
    with h5py.File(args.circuit_file, 'r') as circuitfile, \
         open(args.layers_file, 'r') as layersfile:
        layers_json = json.load(layersfile)
        # for popname, popinfo in circuitfile['populations'].items():
        for popname, popinfo, layer in iter_populations_split_23(circuitfile):
            m_type = m_type_from_layer_m_type(popname)
            ei = 'e' if m_type in utils.EXC_M_TYPES else 'i'

            # Randomly choose a subset of the cells of this population
            num_total = len(popinfo['nCellAff']) # read any attribute to get total # cells
            num_pop = int(args.reduce * num_total)
            choice = sorted(np.random.choice(num_total, num_pop, replace=False))

            if not choice:
                continue

            # TODO: Store nCellAff, etc, on the cell object, then decrement it each time a connection is made?

            # Randomly choose e_types for these cells, according to their prevalence
            e_type_fractions = get_e_type_fractions(popname, layers_json)
            e_type_list = np.random.choice(
                list(e_type_fractions.keys()), replace=True, size=num_pop, p=list(e_type_fractions.values()))
            choice_by_e_type = {e_type: [choice[i] for i, _etype in enumerate(e_type_list) if _etype == e_type]
                                for e_type in e_type_fractions.keys()}
            # for e_type, sub_choice in choice_by_e_type.items():
            for e_type, sub_choice in divide_choice(choice, list(e_type_fractions.keys()), p=list(e_type_fractions.values())).items():
                if not sub_choice:
                    continue
                
                celldata_list = cells[popname][e_type]
                bad_morpho = ['C140201A1_-_Scale_x1.000_y0.975_z1.000_-_Clone_4.asc',
                              'C140201A1_-_Scale_x1.000_y1.050_z1.000_-_Clone_6.asc',
                              'rp110110_L5-3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_9.asc',
                              'C140201A1_-_Clone_6.asc',
                              'C140201A1_-_Clone_1.asc',
                ]
                celldata_list = [x for x in celldata_list if x['morphology'] not in bad_morpho]
                
                for instance, final_choice in divide_choice(sub_choice, range(len(celldata_list))).items():
                    if not final_choice:
                        continue

                    net.add_nodes(
                        N=len(final_choice),
                        layer=layer,
                        m_type=m_type,
                        e_type=e_type,
                        ei=ei,
                        model_type='biophysical',
                        rotation_angle_yaxis=xiter_random(N=len(final_choice), min_x=0.0, max_x=2*np.pi),
                        positions=popinfo['locations'][final_choice, :],
                        BBP_population=popname,
                        population_idx=final_choice,
                        instance=instance,
                        **celldata_list[instance]
                    )
                    log.debug("Created {} {} cells of e_type {}, instance {}".format(
                        len(sub_choice), popname, e_type, instance)
                    )

        log.info("Created cells")

        def BBP_connector(source, target, nsyn_min=1, nsyn_max=10):
            # TODO: Try loading the population->population connectivity matrix, then doing all those (would require BMTK to find all the nodes for a given population)
            # TODO: Or buffer into Python dict
            global connections_processed, BBP_conn_cache
            src_pop, dst_pop = source['BBP_population'], target['BBP_population']
            src_idx, dst_idx = source['population_idx'], target['population_idx']

            cache_key = (src_pop, dst_pop)

            if cache_key not in BBP_conn_cache:
                try:
                    BBP_conn_cache[cache_key] = circuitfile['connectivity'][src_pop][dst_pop]['cMat'][:, :]
                except:
                    # Account for silly BBP typo
                    BBP_conn_cache[cache_key] = circuitfile['connectivty'][src_pop][dst_pop]['cMat'][:, :]
                    
            conn = BBP_conn_cache[cache_key][src_idx, dst_idx]

            connections_processed += 1
            if connections_processed % 10000000 == 0:
                log.debug(connections_processed)

            return np.random.randint(nsyn_min, nsyn_max) if conn else None

        ei_map = {'e': 'Exc', 'i': 'Inh'}
        def dynamics_params_for(pre_ei, post_ei):
            receptor = 'AMPA' if pre_ei == 'e' else 'GABA'
            return '{}_{}To{}.json'.format(receptor, ei_map[pre_ei], ei_map[post_ei])

        for pre_ei, post_ei in itertools.product('ei', repeat=2):
            net.add_edges(
                source={'ei': pre_ei},
                target={'ei': post_ei},
                connection_rule=BBP_connector,
                distance_range=[30.0, 150.0],
                target_sections=['basal', 'apical'],
                weight_function='distributed_weights',
                syn_weight=args.ctx_ctx_weight,
                weight_sigma=args.ctx_ctx_weight_std,
                weight_distribution=args.weight_distn,
                delay=2.0, # TODO: Get from pathways_physiology_factsheets_simplified.json
                dynamics_params=dynamics_params_for(pre_ei, post_ei),
                model_template='exp2syn',
            )
            log.debug('cortical {}-{} connections'.format(pre_ei, post_ei))
            
        log.info("Created ctx-ctx connections")

        log.info("Building cortical column...")
        net.build()
        net.save_nodes(output_dir=args.output)
        net.save_edges(output_dir=args.output)
        log.info("done")


    ####################
    ## THALAMIC INPUT ##
    ####################

    nsyn_min = args.thal_ctx_nsyn[0]
    nsyn_max = args.thal_ctx_nsyn[1]

    n_e = 900 * args.reduce # number of efferent cells per thalamic fiber
    n_l = 350 # number of thalamic synapses per L4 cell
    n_s = 12 # number of synapses per thalamocortical connection
    # n_4 = n_cells_by_layer[4][0] # number of L4 excitatory cells
    n_4 = NUM_BBP_L4_CELLS * args.reduce

    # num_thal = int(float(2*n_l*n_4) / float(n_e*n_s) * args.reduce)
    num_thal = args.num_thal or int(float(2*n_l*n_4) / float(n_e*n_s))
    thal_l4_prob = float(n_e) / (2. * float(n_4))
    thal_l5_prob = thal_l4_prob / 1.5
    thal_l6_prob = thal_l4_prob / 2.0
    thal_ctx_prob = thal_l4_prob / 7.5

    log.info("Creating {} virtual thalamic neurons".format(num_thal))
    log.info('thalamus -> L4 connection probability: {}'.format(thal_l4_prob))

    # TODO: make these input arguments to the script
    thal_prob_peaks = '-672.0,-1300.0' # depths where thalamic targets are most likely to end up
    thal_prob_peak_std = '80.0,60.0' # spread around the peak

    def thalamocortical_connector(source, target, p, nsyn_min, nsyn_max):
        if np.random.random() < p:
            return np.random.randint(nsyn_min, nsyn_max)
        else:
            return 0

    thalamus = NetworkBuilder(name='thalamus')
    thalamus.add_nodes(
        N=num_thal,
        pop_name='spike_trains',
        potential='exc',
        model_type='virtual',
    )

    ## THALAMUS --> ALL LAYERS excitatory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='e', model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p':thal_ctx_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0],
        # prob_peaks=thal_prob_peaks, 
        # prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> ALL LAYERS inhibitory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='i', model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p':thal_ctx_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight * args.ei_weight_ratio,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0],
        # prob_peaks=thal_prob_peaks, 
        # prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> L4 excitatory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='e', layer=4, model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p': thal_l4_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        # target_sections=['basal', 'apical'],
        # distance_range=[30.0, 150.0],
        prob_peaks=thal_prob_peaks, 
        prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='exp2syn',
    )
 
    ## THALAMUS --> L4 inhibitory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='i', layer=4, model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p': thal_l4_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight * args.ei_weight_ratio,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        # target_sections=['basal', 'apical'],
        # distance_range=[30.0, 150.0],
        prob_peaks=thal_prob_peaks, 
        prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> L5 excitatory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='e', layer=5, model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p': thal_l5_prob, #args.thal_l5_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        # target_sections=['basal', 'apical'],
        # distance_range=[30.0, 150.0],
        prob_peaks=thal_prob_peaks, 
        prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> L5 inhibitory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='i', layer=5, model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p': thal_l5_prob, #args.thal_l5_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight * args.ei_weight_ratio,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        # target_sections=['basal', 'apical'],
        # distance_range=[30.0, 150.0],
        prob_peaks=thal_prob_peaks, 
        prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> L6 excitatory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='e', layer=6, model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p': thal_l6_prob, #args.thal_l6_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        # target_sections=['basal', 'apical'],
        # distance_range=[30.0, 150.0],
        prob_peaks=thal_prob_peaks, 
        prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> L6 inhibitory
    thalamus.add_edges(
        source=thalamus.nodes(),
        target=net.nodes(ei='i', layer=6, model_type='biophysical'),
        connection_rule=thalamocortical_connector,
        connection_params={'p': thal_l6_prob, #args.thal_l6_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.thal_ctx_weight * args.ei_weight_ratio,
        weight_function='distributed_weights',
        weight_sigma=args.thal_ctx_weight_std,
        weight_distribution=args.weight_distn,
        # target_sections=['basal', 'apical'],
        # distance_range=[30.0, 150.0],
        prob_peaks=thal_prob_peaks, 
        prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )

    log.info("Building thalamus...")
    thalamus.build()
    thalamus.save_nodes(output_dir=args.output)
    thalamus.save_edges(output_dir=args.output)
    log.info("done")

    
    ######################
    ## BACKGROUND INPUT ##
    ######################

    NUM_BKG = args.num_bkg
    NUM_BKG_E = int(NUM_BKG * args.num_bkg_exc_frac)
    NUM_BKG_I = int(NUM_BKG - NUM_BKG_E)
    nsyn_min = args.bkg_nsyn[0]
    nsyn_max = args.bkg_nsyn[1]

    log.info("Creating {} virtual background neurons ({} exc, {} inh)".format(NUM_BKG, NUM_BKG_E, NUM_BKG_I))
    log.info("Creating {} virtual background neurons".format(NUM_BKG))

    bkg = NetworkBuilder(name='bkg')

    bkg.add_nodes(
        N=NUM_BKG_E,
        pop_name='bkg',
        potential='exc',
        ei='e',
        model_type='virtual',
    )
    
    bkg.add_nodes(
        N=NUM_BKG_I,
        pop_name='bkg_i',
        potential='inh',
        ei='i',
        model_type='virtual',
    )

    bkg_connector = thalamocortical_connector

    log.info("Creating connections from bkg into cortical column")

    log.debug("e-e bkg->cc connections")
    bkg.add_edges(
        source=bkg.nodes(ei='e'),
        target=net.nodes(ei='e'),
        connection_rule=bkg_connector,
        connection_params={'p': args.bkg_exc_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.bkg_weight,
        weight_function='distributed_weights',
        weight_sigma=args.bkg_weight_std,
        weight_distribution=args.weight_distn,
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0], # TODO: What is this?
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='exp2syn',
    )

    log.debug("e-i bkg->cc connections")
    bkg.add_edges(
        source=bkg.nodes(ei='e'),
        target=net.nodes(ei='i'),
        connection_rule=bkg_connector,
        connection_params={'p': args.bkg_exc_prob/args.bkg_ei_ratio,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.bkg_weight * args.ei_weight_ratio,
        weight_function='distributed_weights',
        weight_sigma=args.bkg_weight_std,
        weight_distribution=args.weight_distn,
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0], # TODO: What is this?
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )

    log.debug("i-e bkg->cc connections")
    bkg.add_edges(
        source=bkg.nodes(ei='i'),
        target=net.nodes(ei='e'),
        connection_rule=bkg_connector,
        connection_params={'p': args.bkg_exc_prob,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.bkg_weight,
        weight_function='distributed_weights',
        weight_sigma=args.bkg_weight_std,
        weight_distribution=args.weight_distn,
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0], # TODO: What is this?
        delay=2.0,
        dynamics_params='GABA_InhToExc.json',
        model_template='exp2syn',
    )

    log.debug("i-i bkg->cc connections")
    bkg.add_edges(
        source=bkg.nodes(ei='i'),
        target=net.nodes(ei='i'),
        connection_rule=bkg_connector,
        connection_params={'p': args.bkg_exc_prob/args.bkg_ei_ratio,
                           'nsyn_min': nsyn_min,
                           'nsyn_max': nsyn_max},
        syn_weight=args.bkg_weight,
        weight_function='distributed_weights',
        weight_sigma=args.bkg_weight_std,
        weight_distribution=args.weight_distn,
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0], # TODO: What is this?
        delay=2.0,
        dynamics_params='GABA_InhToInh.json',
        model_template='exp2syn', 
   )

    log.info("building bkg...")
    bkg.build()
    bkg.save_nodes(output_dir=args.output)
    bkg.save_edges(output_dir=args.output)
    log.info("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', '--network', type=str, default='BBP_network',
                        help='network output file directory')

    parser.add_argument('--layers-file', type=str, default='layer_download.json')
    parser.add_argument('--circuit-file', '--circuit', type=str, default='average/cons_locs_pathways_mc0_Column/cons_locs_pathways_mc0_Column.h5', help='BBP circuit file to recreate')

    parser.add_argument('--reduce', '--scale', type=float, default=1.0)

    parser.add_argument('--ctx-ctx-weight', type=float, default=1e-4)
    parser.add_argument('--ctx-ctx-weight-std', type=float, default=5e-5)

    # Thalamic population
    parser.add_argument('--num-thal', type=int, required=False, default=None)
    parser.add_argument('--thal-ctx-nsyn', type=int, nargs=2, default=[7, 17])
    parser.add_argument('--thal-ctx-weight', type=float, default=5e-5)
    parser.add_argument('--thal-ctx-weight-std', type=float, default=1e-5)
    
    # Background population
    parser.add_argument('--num-bkg', type=int, default=10000)
    parser.add_argument('--num-bkg-exc-frac', type=float, default=0.5,
                        help="fraction of bkg neurons that are exc")
    parser.add_argument('--bkg-nsyn', type=int, nargs=2, default=[1, 6])
    parser.add_argument('--bkg-exc-prob', type=float, default=0.025,
                        help="probability that a given bkg neuron connects to a given excitatory neuron")
    parser.add_argument('--bkg-ei-ratio', type=float, default=1.3,
                        help="ratio of e-i connection probs")
    parser.add_argument('--bkg-weight', type=float, default=1e-4)
    parser.add_argument('--bkg-weight-std', type=float, default=5e-5)

    # General weight parameters
    parser.add_argument('--ei-weight-ratio', type=float, default=2.0,
                        help="ratio of e to i weights for all populations")
    parser.add_argument('--weight-distn', type=str, default='lognormal',
                        help="any member of np.random which takes mean, sigma")
    
    args = parser.parse_args()

    build_BBP_network(args)
    generate_electrodes(args)
