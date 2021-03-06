# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import h5py
import numpy as np
from neuron import h

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.io_tools import io

from bmtk.utils.io import cell_vars
try:
    # Check to see if h5py is built to run in parallel
    if h5py.get_config().mpi:
        MembraneRecorder = cell_vars.CellVarRecorderParallel
    else:
        MembraneRecorder = cell_vars.CellVarRecorder

except Exception as e:
    MembraneRecorder = cell_vars.CellVarRecorder

MembraneRecorder._io = io

pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


def first_element(lst):
    return lst[0]


transforms_table = {
    'first_element': first_element,
}


class MembraneReport(SimulatorMod):
    def __init__(self, tmp_dir, file_name, variable_name, cells, sections='all', buffer_data=True, transform={}):
        """Module used for saving NEURON cell properities at each given step of the simulation.

        :param tmp_dir:
        :param file_name: name of h5 file to save variable.
        :param variables: list of cell variables to record
        :param gids: list of gids to to record
        :param sections:
        :param buffer_data: Set to true then data will be saved to memory until written to disk during each block, reqs.
        more memory but faster. Set to false and data will be written to disk on each step (default: True)
        """
        self._all_variables = list(variable_name)
        self._variables = list(variable_name)
        self._transforms = {}
        # self._special_variables = []
        for var_name, fnc_name in transform.items():
            if fnc_name is None or len(fnc_name) == 0:
                del self._transforms[var_name]
                continue

            fnc = transforms_table[fnc_name]
            self._transforms[var_name] = fnc
            self._variables.remove(var_name)

        self._tmp_dir = tmp_dir

        self._file_name = file_name if os.path.isabs(file_name) else os.path.join(tmp_dir, file_name)
        self._all_gids = cells
        self._local_gids = []
        self._sections = sections

        self._var_recorder = MembraneRecorder(self._file_name, self._tmp_dir, self._all_variables,
                                              buffer_data=buffer_data, mpi_rank=MPI_RANK, mpi_size=N_HOSTS)

        self._gid_list = []  # list of all gids that will have their variables saved
        self._data_block = {}  # table of variable data indexed by [gid][variable]
        self._block_step = 0  # time step within a given block

    def _get_gids(self, sim):
        # get list of gids to save. Will only work for biophysical cells saved on the current MPI rank
        selected_gids = set(sim.net.get_node_set(self._all_gids).gids())
        self._local_gids = list(set(sim.biophysical_gids) & selected_gids)

    def _save_sim_data(self, sim):
        self._var_recorder.tstart = 0.0
        self._var_recorder.tstop = sim.tstop
        self._var_recorder.dt = sim.dt

    def initialize(self, sim):
        self._get_gids(sim)
        self._save_sim_data(sim)

        # TODO: get section by name and/or list of section ids
        # Build segment/section list
        sec_list = []
        seg_list = []
        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)
            cell.store_segments()
            for sec_id, sec in enumerate(cell.get_sections()):
                for seg in sec:
                    # TODO: Make sure the seg has the recorded variable(s)
                    sec_list.append(sec_id)
                    seg_list.append(seg.x)

            # sec_list = [cell.get_sections_id().index(sec) for sec in cell.get_sections()]
            # seg_list = [seg.x for seg in cell.get_segments()]
                    
            self._var_recorder.add_cell(gid, sec_list, seg_list)

        self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)

    def step(self, sim, tstep):
        # save all necessary cells/variables at the current time-step into memory
        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)

            for var_name in self._variables:
                seg_vals = [getattr(seg, var_name) for seg in cell.get_segments()]
                self._var_recorder.record_cell(gid, var_name, seg_vals, tstep)

            for var_name, fnc in self._transforms.items():
                seg_vals = [fnc(getattr(seg, var_name)) for seg in cell.get_segments()]
                self._var_recorder.record_cell(gid, var_name, seg_vals, tstep)

        self._block_step += 1

    def block(self, sim, block_interval):
        # write variables in memory to file
        self._var_recorder.flush()

    def finalize(self, sim):
        # TODO: Build in mpi signaling into var_recorder
        pc.barrier()
        self._var_recorder.close()

        pc.barrier()
        self._var_recorder.merge()


class SomaReport(MembraneReport):
    """Special case for when only needing to save the soma variable"""
    def __init__(self, tmp_dir, file_name, variable_name, cells, sections='soma', buffer_data=True, transform={}):
        super(SomaReport, self).__init__(tmp_dir=tmp_dir, file_name=file_name, variable_name=variable_name, cells=cells,
                                         sections=sections, buffer_data=buffer_data, transform=transform)

    def initialize(self, sim):
        self._get_gids(sim)
        self._save_sim_data(sim)

        for gid in self._local_gids:
            self._var_recorder.add_cell(gid, [0], [0.5])
        self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)

    def step(self, sim, tstep, rel_time=0.0):
        # save all necessary cells/variables at the current time-step into memory
        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)
            for var_name in self._variables:
                var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                self._var_recorder.record_cell(gid, var_name, [var_val], tstep)

            for var_name, fnc in self._transforms.items():
                var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                new_val = fnc(var_val)
                self._var_recorder.record_cell(gid, var_name, [new_val], tstep)

        self._block_step += 1

class SectionReport(MembraneReport):
    """For variables like im which have one value per section, not segment"""

    def initialize(self, sim):
        self._get_gids(sim)
        self._save_sim_data(sim)

        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)
            sec_list = range(len(cell.get_sections()))
            self._var_recorder.add_cell(gid, sec_list, sec_list)

        self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)

    def step(self, sim, tstep):
        for gid in self._local_gids:
            for var in self._variables:
                cell = sim.net.get_cell_gid(gid)
                if var == 'im':
                    vals = cell.get_im()
                elif var =='v':
                    vals = np.array([sec.v for sec in cell.get_sections()])
                self._var_recorder.record_cell(gid, var, vals, tstep)
            
        self._block_step += 1
