# -*- coding: utf-8 -*-

"""
This file was created by

$ python -m bmtk.utils.sim_setup -n network --membrane_report-vars v --membrane_report-sections soma --run-time 3000.0 bionet

and renamed from 'run_bionet.py'

To run w/ the gui:
 - Remove the MPI import
 - call `h.load_file("nrngui.hoc")`
 - os.system("pause)"
"""

print("starting")
import os, sys
import glob
import h5py
from bmtk.simulator.bionet.io_tools import io
from mpi4py import MPI # needed to load NEURON with parallel
io.log_info("imported MPI")
from neuron import h
io.log_info("imported h from neuron")
import numpy as np
import pandas as pd
from bmtk.simulator import bionet
from bmtk.simulator.bionet.pyfunction_cache import add_cell_model, add_weight_function

from myplotters import get_spike_rate

io.log_info("done imports")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def load_BBP_templates(templates_dir):
    morphology_templates = glob.glob(os.path.join(templates_dir, '*/morphology.hoc'))
    biophys_templates = glob.glob(os.path.join(templates_dir, '*/biophysics.hoc'))
    syn_templates = glob.glob(os.path.join(templates_dir, '*/synapses/synapses.hoc'))
    hoc_templates = glob.glob(os.path.join(templates_dir, '*/template.hoc'))

    for i, template in enumerate(morphology_templates + biophys_templates + syn_templates + hoc_templates):
        # if 'L5_BP_dSTUT214_5/biophysics.hoc' in template:
        # if 'L5_LBC_dNAC222_4/biophysics.hoc' in template:
        # if 'morpho' not in template:
        #     import ipdb; print ipdb.__file__; ipdb.set_trace()
        #     exit()
        try:
            h.load_file(template)
            # if i == 500:
            #     break
        except RuntimeError:
            io.log_info("Tried to redefine template, ignoring")

def init():
    io.log_info("in init()")
    def distributed_weights(edge_props, src_props, trg_props):
        distn = edge_props['weight_distribution']
        w0 = edge_props['syn_weight']
        sigma = edge_props['weight_sigma']
        scale = 1.0

        if distn.lower() in ['const', 'constant', 'delta']:
            return w0
        if distn.lower() == 'lognormal':
            scale = w0
            w0 = 0

        distn_fcn = getattr(np.random, distn)

        return scale * distn_fcn(w0, sigma)

    add_weight_function(distributed_weights, 'distributed_weights', overwrite=False)

    io.log_info("leaving init")

def runtime_init(conf):
    io.log_info("in runtime_init()")
    h.load_file('import3d.hoc')
    load_BBP_templates(conf.templates_dir)

    def BBP(cell, template_name, dynamics_params):
        """Load a cell from a BlueBrain template"""
        # TODO: uses `conf` from enclosing scope, which is less than ideal
        SYNAPSES = 1
        NO_SYNAPSES = 0

        cwd = os.getcwd()
        os.chdir(os.path.join(conf.templates_dir, cell['model_directory']))
        
        # io.log_info("loading template_name = {}".format(template_name))
        hobj = getattr(h, template_name)(NO_SYNAPSES)
        # io.log_info("finished loading template_name = {}".format(template_name))
        os.chdir(cwd)
        return hobj

    add_cell_model(BBP, directive='BBP', model_type='biophysical', overwrite=False)
    io.log_info("leaving runtime_init()")

def save_seg_coords(sim, conf):
    gid_list = sim.net.get_local_cells().keys()
    io.log_info("Save segment coordinates for cells: {}".format(gid_list))
    outdir = os.path.join(conf.output_dir, conf.output.get('seg_coords_dir', 'seg_coords'))
    if rank == 0:
        os.mkdir(outdir)

    comm.Barrier()

    for gid, cell in sim.net.get_local_cells().items():
        outfilename = os.path.join(outdir, '{}.h5'.format(gid))
        with h5py.File(outfilename, 'w') as outfile:
            coords = cell.get_seg_coords()
            outfile.create_dataset('p0', data=coords['p0'])
            outfile.create_dataset('p05', data=coords['p05'])
            outfile.create_dataset('p1', data=coords['p1'])

            outfile.create_dataset('d0', data=coords['d0'])
            outfile.create_dataset('d1', data=coords['d1'])
            outfile.create_dataset('ei', data=cell.node['ei'])
            outfile.create_dataset('part_ids', data=cell.get_part_ids())
            outfile.create_dataset('m_type', data=cell.node['m_type'])
            outfile.create_dataset('e_type', data=cell.node['e_type'])
            outfile.create_dataset('layer', data=cell.node['layer'])
            outfile.create_dataset('soma_pos', data=cell.soma_position)

def save_master_file(sim, conf):
    pop = sim.net.get_node_population('cortical_column')
    num_cells = pop.n_nodes()
    masterfilename = os.path.join(conf.output_dir, 'master.h5')
    with h5py.File(masterfilename, 'w') as masterfile:
        masterfile.create_dataset('neuro-data-format-version', data=0.3)
        masterfile.create_dataset('num_cells', data=num_cells)

        # population spikerates
        thal_spikes_df = pd.read_csv(conf.inputs['ext_spikes']['input_file'], sep=' ')
        thal_spikes = [
            float(t) for spike_times_str in thal_spikes_df['spike-times']
            for t in spike_times_str.split(',')
        ]
        thal_bins, thal_spikerate = get_spike_rate(
            thal_spikes, len(thal_spikes_df), 0, sim.tstop, binsize=1.0)

        bkg_spikes_df = pd.read_csv(conf.inputs['bkg_spikes']['input_file'], sep=' ')
        bkg_spikes = [
            float(t) for spike_times_str in thal_spikes_df['spike-times']
            for t in spike_times_str.split(',')
        ]
        bkg_bins, bkg_spikerate = get_spike_rate(
            bkg_spikes, len(bkg_spikes_df), 0, sim.tstop, binsize=1.0)

        assert np.all(bkg_bins == thal_bins)
        masterfile.create_dataset('spikerate_bins', data=thal_bins)
        masterfile.create_dataset('thal_spikerate', data=thal_spikerate)
        masterfile.create_dataset('bkg_spikerate', data=bkg_spikerate)
        

    # TODO: check sim._spikes and sim._spikes_table after sim runs for cortical spike data

def create_graph(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    runtime_init(conf)

    graph = bionet.BioNetwork.from_config(conf)
    
    return graph, conf

def create_sim(config_file):
    graph, conf = create_graph(config_file)
    sim = bionet.BioSimulator.from_config(conf, network=graph)

    return sim, conf

def intensity(depth):
    n = 1.36
    NA = 0.37
    r = 1000
    S = 10.3
    rho = r * ((n/NA)**2 - 1)**(0.5)
    I_0 = 1e25

    return I_0 * (rho**2)/((S + 1)*(depth + rho)**2)

def run(config_file):
    io.log_info("in run")
    
    sim, conf = create_sim(config_file)

    # Start inserting optogenetic channels
    net = sim.net
    gids = net.get_local_cells()
    for gid in gids:
        cell = net.get_cell_gid(gid)
        #print('Cell type: ' + str(cell.node['layer']) + str(cell.node['ei']))
        if str(cell.node['layer']) + str(cell.node['ei']) in sim.optocell:
            h('objref rho')
            secs = cell.get_sections()
            seg_depths = cell.get_seg_coords()['p05'][1] # Get the y of the midpoint coordinate of every segment
            # Keep track of which sec we're on
            curr_sec = secs[0]
            num_segs = curr_sec.n3d()
            seg_num = 0
            for i in range(len(secs)):
                sec = secs[i]
                if not sec == curr_sec:
                    curr_sec = sec
                    num_segs = curr_sec.n3d()
                    seg_num = 0
                if num_segs > 0:
                    sec_id = h.this_section(sec=sec)
                    h.push_section(sec_id)
                    offset =  1.0 /(2*num_segs)
                    seg_loc = float(seg_num) / num_segs + offset
                    #print('Inserting RhO4 channel at depth ' + str(seg_depths[i]) + ', section ' + str(sec) + ', segment ' + str(seg_num))
                    h('rho = new RhO4(' + str(seg_loc)  + ', sec=sec)')
                    light_intensity = intensity(-(seg_depths[i] - 1000))
                    h.rho.phiOn = light_intensity
                    h.pop_section(sec_id)
                    seg_num += 1
    # End inserting optogenetic channels
    io.log_info("created simulation")

    save_seg_coords(sim, conf)
    io.log_info("saved segment coordinates")

    if rank == 0:
        save_master_file(sim, conf)
        io.log_info("saved master file")
    
    sim.run()
    sim.report_load_balance()

    io.log_info("DONE SIMULATION")
    


if __name__ == '__main__':
    init()
    
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')

    bionet.nrn.quit_execution()
