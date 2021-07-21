from __future__ import print_function

import os
import argparse
import json
import csv

import numpy as np
import h5py

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.aux.node_params import positions_columinar, xiter_random
from bmtk.builder.aux.edge_connectors import connect_random

import utils
from cells import cells, m_type_from_layer_m_type
from layers import layer_depths, layer_thicknesses
from stimulus import stimulus, save_csv

TOT_THICKNESS = sum(layer_thicknesses.values())

L2_DENSITY = 164600 * 1e-9
L3_DENSITY = 83800 * 1e-9

L2_NUM = L2_DENSITY * layer_thicknesses[2]
L3_NUM = L3_DENSITY * layer_thicknesses[3]

# Fraction of L23 neurons that are in layer 2
L2_FRAC = L2_NUM / (L2_NUM + L3_NUM)

NUM_BBP_CELLS = 31346
BBP_RADIUS = 230.0

# Keys into layer_download.json:
M_TYPE_KEY = "No. of neurons per morphological types"
E_TYPE_KEY = "No. of neurons per electrical types"

too_far = 0 # used to count neurons whose soma-soma distance went out of range

def build_cortical_column(args):
    
    conn_prob_file = args.conn_probs
    output = args.output

    COLUMN_RADIUS = args.column_radius
    scale_factr = COLUMN_RADIUS**2 / BBP_RADIUS**2 * args.reduce

    ###########
    ## CELLS ##
    ###########

    net = NetworkBuilder("cortical_column")

    n_cells_by_layer = {}
    with open(args.layers_file, 'r') as layersfile:
        layers_json = json.load(layersfile)
        # for layername, layerdata in json.load(layersfile).items():
        n_cells_tot, n_cells_tot_e, n_cells_tot_i = 0, 0, 0
        for layer, ei, layer_ei, layername in utils.iter_populations():
            if ei == 'e':
                continue # We iterate over all m-types, which includes exc. and inh.

            L23_scale = L2_FRAC if layer == 2 else (1-L2_FRAC) if layer == 3 else 1

            # Data from BBP layers_download.json
            layerdata = layers_json['L'+layername[:-1]]
            n_tot_layer = int(sum(layerdata[M_TYPE_KEY].values()) * L23_scale)
            n_cells_layer, n_cells_layer_e, n_cells_layer_i = 0, 0, 0
            e_type_counts = layerdata[E_TYPE_KEY]
            
            for layer_m_type, num_tot_m in layerdata[M_TYPE_KEY].items():
                num_tot_m = num_tot_m * L23_scale
                n_cells_m = 0
                
                # layer_m_type = L4_LBC  --->  m_type = LBC
                m_type = m_type_from_layer_m_type(layer_m_type)
                ei = 'e' if m_type in utils.EXC_M_TYPES else 'i'
                
                sum_e_type = sum(e_type_counts[e_type] for e_type in cells[layer_m_type].keys())
                for e_type, cell_list in cells[layer_m_type].items():
                    # Fraction of cells of the current m_type which are also the current e_type
                    frac_this_e_type = e_type_counts[e_type]/float(sum_e_type)
                    n_pop = num_tot_m * frac_this_e_type * scale_factr / len(cell_list)

                    # Nonzero chance of all cells being instantiated
                    if n_pop < 1:
                        n_pop = 1 if np.random.random() < n_pop else 0
                    else:
                        n_pop = int(n_pop)
                    
                    for i, celldata in enumerate(cell_list):
                        net.add_nodes(
                            N=n_pop,
                            layer=layer,
                            m_type=m_type,
                            e_type=e_type,
                            instance=i,
                            ei=ei,
                            model_type='biophysical',
                            rotation_angle_yaxis=xiter_random(N=n_pop, min_x=0.0, max_x=2*np.pi),
                            positions=positions_columinar(
                                N=n_pop,
                                center=[0.0, layer_depths[int(layer)], 0.0],
                                height=layer_thicknesses[int(layer)],
                                max_radius=COLUMN_RADIUS,
                            ),
                            **celldata
                        )
                        
                        n_cells_m += n_pop
                        n_cells_layer += n_pop
                        n_cells_tot += n_pop
                        if ei == 'e':
                            n_cells_layer_e += n_pop
                            n_cells_tot_e += n_pop
                        else:
                            n_cells_layer_i += n_pop
                            n_cells_tot_i += n_pop
                            
                
                print("   {} ({}): N={}".format(layer_m_type, ei, n_cells_m))
                
            print("Created layer {} cells, N={} ({} exh, {} inh)".format(layer, n_cells_layer, n_cells_layer_e, n_cells_layer_i))
            n_cells_by_layer[layer] = (n_cells_layer_e, n_cells_layer_i)
            print('')
            print('-' * 80)
            print('')

    print("Created {} total cells ({} exc, {} inh)".format(n_cells_tot, n_cells_tot_e, n_cells_tot_i))
                        

    ###########
    ## EDGES ##
    ###########

    def custom_dist_connector(conn, bins):
        def num_synapses(source, target, nsyn_min, nsyn_max):
            global too_far
            # distance between source and target
            r = np.linalg.norm(np.array(source['positions'])
                               - np.array(target['positions']))

            # If the distance is larger than the range of distances calculated,
            # use the last bin
            i = np.digitize(r, bins)
            if i > len(conn):
                too_far += 1
                # if too_far % 100 == 0:
                #     import ipdb; ipdb.set_trace()
                #     print(too_far)
            connection_prob = conn[min(i, len(conn)-1)] / scale_factr

            if np.random.random() > connection_prob:
                return None

            return np.random.randint(nsyn_min, nsyn_max)

        return num_synapses


    ei_map = {'e': 'Exc', 'i': 'Inh'}
    def dynamics_params_for(pre_ei, post_ei):
        receptor = 'AMPA' if pre_ei == 'e' else 'GABA'
        return '{}_{}To{}.json'.format(receptor, ei_map[pre_ei], ei_map[post_ei])

    # Load connection probability file
    print("Creating cortex-cortex connections")
    with h5py.File(conn_prob_file, 'r') as f:
        for pre_layer, pre_ei, _, pre_layer_key in utils.iter_populations():
            for post_layer, post_ei, _, post_layer_key in utils.iter_populations():
                # To biophysical neurons
                weight_scale = args.ei_weight_ratio if pre_ei == 'e' and post_ei == 'i' else 1.0
                net.add_edges(
                    source={'layer': pre_layer, 'ei': pre_ei},
                    target={'layer': post_layer, 'ei': post_ei,
                            'model_type': 'biophysical'},
                    connection_rule=custom_dist_connector(
                        f['conn'][pre_layer_key][post_layer_key][:],
                        f['bins'][pre_layer_key][post_layer_key][:]
                    ),
                    connection_params={'nsyn_min': 1, 'nsyn_max': 10},
                    # TODO: read from params file (per layer pair):
                    distance_range=[30.0, 150.0], # TODO: What does this do?
                    target_sections=['basal', 'apical'],
                    weight_function='distributed_weights',
                    syn_weight=args.ctx_ctx_weight * weight_scale,
                    weight_sigma=args.ctx_ctx_weight_std,
                    weight_distribution=args.weight_distn,
                    delay=2.0,
                    dynamics_params=dynamics_params_for(pre_ei, post_ei),
                    model_template='exp2syn',
                )
    

    ####################           
    ## THALAMIC INPUT ##
    ####################

    nsyn_min = args.thal_ctx_nsyn[0]
    nsyn_max = args.thal_ctx_nsyn[1]

    n_e = 900 * float(n_cells_tot) / float(NUM_BBP_CELLS) # number of efferent cells per thalamic fiber
    n_l = 350 # number of thalamic synapses per L4 cell
    n_s = 12 # number of synapses per thalamocortical connection
    n_4 = n_cells_by_layer[4][0] # number of L4 excitatory cells

    num_thal = int(float(2*n_l*n_4) / float(n_e*n_s) * args.reduce)
    thal_l4_prob = float(n_e) / (2. * float(n_4))
    thal_l5_prob = thal_l4_prob / 1.5
    thal_l6_prob = thal_l4_prob / 2.0
    thal_ctx_prob = thal_l4_prob / 7.5

    # TODO: make these input arguments to the script
    thal_prob_peaks = '-672.0,-1300.0' # depths where thalamic targets are most likely to end up
    thal_prob_peak_std = '80.0,60.0' # spread around the peak

    def thalamocortical_connector(source, target, p, nsyn_min, nsyn_max):
        if np.random.random() < p:
            return np.random.randint(nsyn_min, nsyn_max)
        else:
            return 0

    print("Creating {} virtual thalamic neurons".format(num_thal))
    thalamus = NetworkBuilder(name='thalamus')
    thalamus.add_nodes(
        N=num_thal,
        pop_name='spike_trains',
        potential='exc',
        model_type='virtual',
    )

    print("Creating random thalamocortical connections to all layers")
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
    print("Creating thalamus --> L4 excitatory connections")
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
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0],
        # prob_peaks=thal_prob_peaks, 
        # prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> L5 excitatory
    print("Creating thalamus --> L5 excitatory connections")
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
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0],
        # prob_peaks=thal_prob_peaks, 
        # prob_peak_std=thal_prob_peak1_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )

    ## THALAMUS --> L6 excitatory
    print("Creating thalamus --> L6 excitatory connections")
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
        target_sections=['basal', 'apical'],
        distance_range=[30.0, 150.0],
        # prob_peaks=thal_prob_peaks, 
        # prob_peak_std=thal_prob_peak_std, 
        delay=2.0,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='exp2syn',
    )


    ######################
    ## BACKGROUND INPUT ##
    ######################

    NUM_BKG = args.num_bkg * args.reduce
    NUM_BKG_E = int(NUM_BKG * args.num_bkg_exc_frac)
    NUM_BKG_I = int(NUM_BKG - NUM_BKG_E)
    nsyn_min = args.bkg_nsyn[0]
    nsyn_max = args.bkg_nsyn[1]

    print("Creating {} virtual background neurons ({} exc, {} inh)".format(NUM_BKG, NUM_BKG_E, NUM_BKG_I))
    print("Creating {} virtual background neurons".format(NUM_BKG))

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

    print("Creating e-e connections from bkg into cortical column")
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

    print("Creating e-i connections from bkg into cortical column")
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

    print("Creating i-e connections from bkg into cortical column")
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

    print("Creating i-i connections from bkg into cortical column")
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

    ##################
    ## SAVE NETWORK ##
    ##################

    net.build()
    net.save_nodes(output_dir=output)
    net.save_edges(output_dir=output)

    thalamus.build()
    thalamus.save_nodes(output_dir=output)
    thalamus.save_edges(output_dir=output)

    bkg.build()
    bkg.save_nodes(output_dir=output)
    bkg.save_edges(output_dir=output)

    print("{} connections were beyond the max distance in the bluebrain model".format(too_far))



def generate_electrodes(args):
    ELECTRODE_SPACING = 100.0
    ECOG_RAD = 25.0
    ECOG_N = 10 # number of points to average for ECoG
    PIXEL_N = 32
    with open(os.path.join(args.output, 'electrodes.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(["x_pos", "y_pos", "z_pos"])
        # The signal on these 4 electrodes will be averaged to compute the ECoG signal
        # y is depth
        # for i in range(ECOG_N):
        #     xz = np.random.random(2) * ECOG_RAD - ECOG_RAD/2.0
        #     writer.writerow([xz[0], 2083, xz[1]])
        # writer.writerow([ ECOG_RAD/2, 2083, 0])
        # writer.writerow([-ECOG_RAD/2, 2083, 0])
        # writer.writerow([0, 2083,  ECOG_RAD/2])
        # writer.writerow([0, 2083, -ECOG_RAD/2])
        # for y in np.arange(ELECTRODE_SPACING, TOT_THICKNESS, ELECTRODE_SPACING):
        #     writer.writerow([0, y, 0])

        # for y in np.arange(ELECTRODE_SPACING, 2000.0, ELECTRODE_SPACING):
        #     for i in range(PIXEL_N):
        #         x, z = np.random.random(2) * 12 - 6
        #         writer.writerow([x, y, z])

        # Bottom of column is (0, 0, 0)

        # Location of bottom corner:
        X, Y, Z = 0, 1000, 0
        # Pixels are in the X-Y plane (Z = 0 throughout)
        VSPACE = 20
        HSPACE = 16

        ROWS, COLS = 10, 4
        idx = 1
        for row in range(ROWS):
            for col in range(COLS):
                if (row + col) % 2 == 0: # checkerboard pattern
                    x = X + col * HSPACE
                    y = Y + row * VSPACE
                    z = Z
                    print(x, y, z)

                    for i in range(PIXEL_N):
                        dx, dy = np.random.random(2) * 12 - 6
                        writer.writerow([x+dx, y+dy, z])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', '--network', type=str, default='network',
                        help='network output file directory')
    parser.add_argument('--conn-probs', type=str,
                        default='connection/conn_probs_FINAL.h5')
    parser.add_argument('--layers-file', type=str, default='layer_download.json')
    parser.add_argument('--reduce', '--scale', type=float, default=1.0)
    parser.add_argument('--column-radius', type=float, default=60.0)
    parser.add_argument('--tstop', type=float, required=False, default=2500.0)

    # Thalamic population
    # parser.add_argument('--num-thal', type=int, default=400) # This is now determined by formula
    parser.add_argument('--thal-ctx-nsyn', type=int, nargs=2, default=[7, 17])
    # parser.add_argument('--thal-ctx-prob', type=float, default=0.0005) # This is now determined by formula
    # parser.add_argument('--thal-l4-prob', type=float, default=0.005) # This is now determined by formula
    # For now: just use L4/2 and L4/4
    # parser.add_argument('--thal-l5-prob', type=float, default=0.05)
    # parser.add_argument('--thal-l6-prob', type=float, default=0.025)
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

    # Cortico-cortico weights
    parser.add_argument('--ctx-ctx-weight', type=float, default=1e-4)
    parser.add_argument('--ctx-ctx-weight-std', type=float, default=5e-5)

    # General weight parameters
    parser.add_argument('--ei-weight-ratio', type=float, default=2.0,
                        help="ratio of e to i weights for all populations")
    parser.add_argument('--weight-scale', type=float, default=1.0,
                        help="multiplier for all weights and weight std's. Usefor for param sweeps")
    parser.add_argument('--weight-std-scale', type=float, default=1.0,
                        help="multiplier for weight std's. Total multiplier is --weight-scale * this")
    parser.add_argument('--weight-distn', type=str, default='lognormal',
                        help="any member of np.random which takes mean, sigma")
    parser.add_argument('--weight-std', type=float, required=False, default=None,
                        help='overrides --*-weight-std and --*-weight-scale')

    args = parser.parse_args()

    # multiply weights and stds by --weight-scale
    args.bkg_weight = args.bkg_weight * args.weight_scale
    args.thal_ctx_weight = args.thal_ctx_weight * args.weight_scale
    args.ctx_ctx_weight = args.ctx_ctx_weight * args.weight_scale
    args.bkg_weight_std = args.bkg_weight_std * args.weight_scale
    args.thal_ctx_weight_std = args.thal_ctx_weight_std * args.weight_scale
    args.ctx_ctx_weight_std = args.ctx_ctx_weight_std * args.weight_scale
    args.weight_scale = "Already multiplied into weights"

    # then multiply only stds by --weight-std-scale
    # taking args.weight_std (if present) as a final override
    args.bkg_weight_std = args.weight_std or (args.bkg_weight_std * args.weight_std_scale)
    args.thal_ctx_weight_std = args.weight_std or (args.thal_ctx_weight_std * args.weight_std_scale)
    args.ctx_ctx_weight_std = args.weight_std or (args.ctx_ctx_weight_std * args.weight_std_scale)
    args.weight_std_scale = "Irrelevent; --weight-std was passed" if args.weight_std else "Already multiplied into weights"

    # if any([args.bkg_weight_std > args.bkg_weight,
    #         args.ctx_ctx_weight_std > args.ctx_ctx_weight,
    #         args.thal_ctx_weight_std > args.thal_ctx_weight]):
    #     raise ValueError('weight std cannot be greater than weight mean')

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    print("Args to build_network:")
    print(json.dumps(args.__dict__, indent=4, sort_keys=True))
    with open(os.path.join(args.output, 'build_network_params.json'), 'w') as outfile:
        print(json.dumps(args.__dict__, indent=4, sort_keys=True), file=outfile)
    
    build_cortical_column(args)
    generate_electrodes(args)

