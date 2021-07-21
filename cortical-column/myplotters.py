import os
import csv
import json
import itertools
from collections import defaultdict

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import utils

get_nodes_df = utils.get_nodes_df

def get_ei_groups(nodes_df, groupkey='layer'):
    """
    Return two dicts of layer: dataframe
    (one for exc, one for inh)
    """
    groupings = nodes_df.groupby(['layer', 'ei'])
    igroups = {name[0]: grp for name, grp in groupings if name[1] == 'i'}
    egroups = {name[0]: grp for name, grp in groupings if name[1] == 'e'}

    return egroups, igroups


def get_ei_cmaps(num_egroups, num_igroups, real=False):
    if real:
        color_norm_e = colors.Normalize(vmin=0, vmax=num_egroups+1)
        scalar_map_e = cmx.ScalarMappable(norm=color_norm_e, cmap='Greys')
        cmap_e = [scalar_map_e.to_rgba(i+1) for i in range(0, num_egroups+1)][1:]

        color_norm_i = colors.Normalize(vmin=0, vmax=num_igroups+1)
        scalar_map_i = cmx.ScalarMappable(norm=color_norm_i, cmap='Reds')
        cmap_i = [scalar_map_i.to_rgba(i+1) for i in range(0, num_igroups+1)][1:]
    else:
        cmap_e = ['black'] * num_egroups
        cmap_i = ['r'] * num_igroups

    return cmap_e, cmap_i

def get_layer_cmap(nlayers=6):
    color_norm = colors.Normalize(vmin=0, vmax=nlayers)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    _colors = [scalar_map.to_rgba(i+1) for i in range(0, nlayers+1)]

    return {
        '{}{}'.format(layer, ei): color
        for ei, (layer, color)
        in itertools.product(['e', 'i'], zip(range(nlayers+1), _colors))
    }
        

def get_cmap(num_egroups=6, num_igroups=6, real=False):
    cmap_e, cmap_i = get_ei_cmaps(num_egroups, num_igroups, real=real)
    cmap = {}
    for idx, layer in enumerate(range(1, 7)):
        cmap["{}{}".format(layer, 'i')] = cmap_i[idx]
    for idx, layer in enumerate(range(1, 7)):
        cmap["{}{}".format(layer, 'e')] = cmap_e[idx]
    # point just the layer to the excitatory color
    for layer in range(1, 7):
        cmap[layer] = cmap['{}e'.format(layer)]
        cmap[str(layer)] = cmap[layer]
    return cmap


def iter_spike_data(cells_file, cell_models_file, spikes_file, population, groupby='layer_ei'):
    """
    Return a dict mapping layer or layer_ei to dataframe containing spike times
    Also return all spike times as 2nd retval for convenience
    """
    nodes_df = get_nodes_df(cells_file, cell_models_file, population=population)

    # TODO: Uses utils.SpikesReader to open
    spikes_h5 = h5py.File(spikes_file, 'r')
    spike_gids = np.array(spikes_h5['/spikes/gids'], dtype=np.uint)
    spike_times = np.array(spikes_h5['/spikes/timestamps'], dtype=np.float)
    # spike_times, spike_gids = np.loadtxt(spikes_file, dtype='float32,int', unpack=True)
    # spike_gids, spike_times = np.loadtxt(spikes_file, dtype='int,float32', unpack=True)

    egroups, igroups = get_ei_groups(nodes_df, 'layer')

    def insert_layer_ei(df, groups, ei):
        for layer, group_df in groups.items():
            indexes = np.in1d(spike_gids, group_df.index) # 'index' within the pandas df = gid
            df['{}{}'.format(layer, ei)] = {'gids': spike_gids[indexes], 'spike_times': spike_times[indexes]}
            
    spike_dfs = {}
    insert_layer_ei(spike_dfs, egroups, 'e')
    insert_layer_ei(spike_dfs, igroups, 'i')

    if groupby == 'layer':
        # combine the e/i data structures
        spike_dfs['1e'] = {'gids': [], 'spike_times': []}
        spike_dfs = {
            layer: {
                'gids': np.concatenate(list(
                    spike_dfs['{}{}'.format(layer, ei)]['gids']
                    for ei in 'ei')),
                'spike_times': np.concatenate(list(
                    spike_dfs['{}{}'.format(layer, ei)]['spike_times']
                    for ei in 'ei')),
            }
            for layer in [1, 2, 3, 4, 5, 6]
        }

    return spike_dfs, spike_times

    
def get_stim_intervals(stim_intervals_file=None):
    if stim_intervals_file:
        try:
            with open(stim_intervals_file, 'r') as infile:
                stim_intervals = json.load(infile)
        except IOError:
            print("Could not find stim intervals file {}".format(stim_intervals_file))
            stim_intervals = []
        except ValueError:
            print("Could not decode stim_intervals.json")
            stim_intervals = []
        return stim_intervals
    return None

def get_ei_population_axes(ss=None):
    """
    Return two axes, one for exc and one for inh
    
    ss - subplot_spec to put the axes on. If None, creates new top-level subplot/gridspec
    """
    if ss is None:
        gs = gridspec.GridSpec(2, 1)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ss)

    ax_i, ax_e = plt.subplot(gs[0]), plt.subplot(gs[1])
    return ax_e, ax_i

def get_layer_ei_axes(ss=None):
    """
    Return the horizontal (time) axis, and a dict mapping layer/ei to axes

    ss - subplot_spec to put the axes on. If None, creates new top-level subplot/gridspec
    """
    if ss is None:
        gs = gridspec.GridSpec(11, 1)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(11, 1, subplot_spec=ss)

    cmap = get_cmap()

    axes = {}
    for i, (_, _, layer_ei, _) in enumerate(utils.iter_populations()):
        ax = plt.subplot(gs[i])
        ax.axes.get_xaxis().set_visible(False)
        ax.text(1.09, 0.5, layer_ei, transform=ax.transAxes, color=cmap[layer_ei])
        axes[layer_ei] = ax

    axes[layer_ei].axes.get_xaxis().set_visible(True)

    return axes
        

def get_raster_axes(plot_mode, raster_ratio):
    """
    Return a dict mapping layer/ei to axes, and the two population spike rate axes
    """
    top_gs = gridspec.GridSpec(raster_ratio, 1)
    raster_ss = top_gs[:raster_ratio-2]

    if plot_mode.lower() in('same_axes',):
        ax_e = ax_i = plt.subplot(raster_ss)
        raster_axes = {layer_ei: ax_e if ei == 'e' else ax_i for (_, ei, layer_ei, _) in utils.iter_populations()}
    elif plot_mode.lower() in ('ei', 'ei_axes',):
        ax_e, ax_i = get_ei_population_axes(ss=raster_ss)
        raster_axes = {layer_ei: ax_e if ei == 'e' else ax_i for (_, ei, layer_ei, _) in utils.iter_populations()}
    elif plot_mode.lower() in ('individual', 'layer',):
        raster_axes = get_layer_ei_axes(ss=raster_ss)
    else:
        raise ValueError("plot_mode must be one of 'individual', 'ei', 'same_axes'")

    for ax in raster_axes.values():
        ax.axes.get_xaxis().set_visible(False)

    ctx_rate_axes = plt.subplot(top_gs[-2])
    ctx_rate_axes.axes.get_xaxis().set_visible(False)
    bkg_rate_axes = plt.subplot(top_gs[-1])

    return raster_axes, ctx_rate_axes, bkg_rate_axes

def plot_spikes(cells_file, cell_models_file, spikes_file,
                stim_intervals_file=None, population=None,
                thal_spikes_file=None, bkg_spikes_file=None,
                save_as=None, show=True, marker='o', plot_every=None,
                plot_depth=False, plot_mode='same_axes', alpha=0.1, tstart=None,
                tstop=None, overlay_rate=False):

    plt.figure(figsize=(6.5, 6))
    plt.subplots_adjust(right=0.8)

    axes, ctx_rate_ax, bkg_rate_ax = get_raster_axes(plot_mode=plot_mode, raster_ratio=12 if plot_mode=='individual' else 8)
    cmap = get_cmap()

    nodes_df = get_nodes_df(cells_file, cell_models_file, population=population)
    spike_dfs, spike_times = iter_spike_data(cells_file, cell_models_file, spikes_file, population)

    for layer_ei, spikedata in spike_dfs.items():
        # if plot_every:
        #     to_plot = np.random.choice(group_df.index, size=len(group_df)/plot_every)
        # else:
        #     to_plot = group_df.index
        ax = axes[layer_ei]
        if not plot_depth:
            ax.axes.get_yaxis().set_visible(False)
        else:
            if layer_ei == '6e': # always last plot
                ax.axes.set_ylabel('Depth (um)')

        y = nodes_df['depth'][spikedata['gids']] if plot_depth else -np.int32(spikedata['gids'])
        ax.scatter(spikedata['spike_times'], y, marker=marker, facecolors=cmap[layer_ei],
                   label=layer_ei, lw=0, s=5, alpha=alpha)


    # y = nodes_df['depth'] if plot_depth else -np.int32(spike_gids)
    # y = nodes_df['depth'] if plot_depth else -np.arange(len(nodes_df), dtype=np.int32)
    # y_min = min(y)
    # y_max = max(y)
    tstart = tstart or 0
    tstop = tstop or max(spike_times)

    plot_spike_rate(ctx_rate_ax, spike_times, len(nodes_df), tstart, tstop, "Cortex", color='green')
    ctx_rate_ax.legend(markerscale=2, scatterpoints=1, loc='center left',
                       bbox_to_anchor=(1.0, 0.5))
    
    if bkg_spikes_file:
        bkg_spikes_df = pd.read_csv(bkg_spikes_file, sep=' ')
        bkg_spike_times = [
            float(t) for spike_times_str in bkg_spikes_df['spike-times']
            for t in spike_times_str.split(',')
        ]
        num_bkg_nodes = len(bkg_spikes_df['spike-times'])
        plot_spike_rate(bkg_rate_ax, bkg_spike_times, num_bkg_nodes, tstart, tstop, "Bkg")

    if thal_spikes_file:
        thal_spikes_df = pd.read_csv(thal_spikes_file, sep=' ')
        thal_spike_times = [
            float(t) for spike_times_str in thal_spikes_df['spike-times']
            for t in spike_times_str.split(',')
        ]
        num_thal_nodes = len(thal_spikes_df['spike-times'])
        plot_spike_rate(bkg_rate_ax, thal_spike_times, num_thal_nodes, tstart, tstop, "Thalamus")

    bkg_rate_ax.legend(markerscale=2, scatterpoints=1, loc='center left',
                         bbox_to_anchor=(1.0, 0.5))

    for ax in axes.values():
        ax.set_xlim([tstart, tstop])

    if overlay_rate:
        plot_rates(cells_file, cell_models_file, spikes_file, show=False, axes=axes, double_6e=True, tstart=tstart, tstop=tstop)

    if save_as is not None:
        plt.savefig(save_as)

    if show:
        plt.show()

def get_spike_rate(spikes, num_nodes, tstart, tstop, binsize=1.0):
    # binsize in ms
    bins = np.arange(tstart, tstop, binsize)
    num_spikes, _ = np.histogram(spikes, bins=bins)
    return bins, np.float64(num_spikes)/num_nodes

def plot_spike_rate(axs, spikes, num_nodes, tstart, tstop, name_for_label,
                    color=None, binsize=1.0):
    bins, p_spike = get_spike_rate(spikes, num_nodes, tstart, tstop, binsize=binsize)
    axs.plot(bins[:-1], p_spike, color=color, label=name_for_label)
    axs.set_xlabel('time (s)')
    axs.set_xlim([tstart, tstop])
    # TODO: Set max appropriately for multi-plots
    bin_100ms = int(50. / binsize)
    # ignore first 50ms for y range
    axs.set_ylim([0, max(p_spike[bin_100ms:]) * 1.1])

        
def get_random_colors(n):
    color_norm = colors.Normalize(vmin=0, vmax=(n-1))
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    return [scalar_map.to_rgba(i) for i in range(0, n)]


def plot_traces(datakey, cell_vars_h5, cells_file, cell_models_file,
                max_per_layer=None, show=True, save_as=None, yticks=None,
                stim_intervals_file=None, title=None, tstart=None, tstop=None):
    nodes_df = get_nodes_df(cells_file, cell_models_file)
    egroups, igroups = get_ei_groups(nodes_df, 'layer')
    cmap_e, cmap_i = get_ei_cmaps(len(egroups), len(igroups))

    data_h5 = h5py.File(cell_vars_h5, 'r')
    membrane_trace = data_h5[datakey]

    time_ds = data_h5['/mapping/time']
    tstart = tstart or time_ds[0]
    tstop = tstop or time_ds[1]
    fs = time_ds[2]
    # x_axis = np.linspace(tstart, tstop, fs, endpoint=True)
    x_axis = np.arange(time_ds[0], time_ds[1], time_ds[2])
    time_idx = (x_axis >= tstart) & (x_axis < tstop)

    gids_ds = data_h5['/mapping/gids']
    index_ds = data_h5['/mapping/index_pointer']
    index_lookup = {gids_ds[i]: (index_ds[i], index_ds[i+1]) for i in range(len(gids_ds))}

    gs = gridspec.GridSpec(len(igroups) + len(egroups), 1)

    stim_intervals = get_stim_intervals(stim_intervals_file)

    def _plot_traces(cmap, groups, start, ei):
        for i, color, (layer, group_df) in zip(range(start, start+len(groups)), cmap, groups.items()):
            ax = plt.subplot(gs[i])

            # Grab some membrane traces
            indexes = np.in1d(gids_ds, group_df.index)
            num_traces = sum(indexes)
            if num_traces == 0:
                print("skipping layer {}, no cells saved".format(layer))
                ax.remove()
                continue
            if max_per_layer and num_traces > max_per_layer:
                idx = np.where(indexes == True)[0]
                traces = np.random.choice(idx, size=max_per_layer, replace=False)
                group_traces = membrane_trace[:, sorted(traces)]
            else:
                group_traces = membrane_trace[:, indexes]

            if max_per_layer == 1:
                colors = [color] * len(group_traces.T)
            else:
                colors = get_random_colors(max_per_layer)

            for trace, col in zip(group_traces.T, colors):
                ax.plot(x_axis[time_idx], trace[time_idx], label=layer, color=col, linewidth=0.75)

            ax.text(1.02, 0.5, "{}{}".format(layer, ei), transform=ax.transAxes, color=color)

            ax.axes.get_xaxis().set_visible(False)

            if yticks is not None:
                ax.axes.set_yticks(yticks)

            for s in stim_intervals:
                left, right = ax.axes.get_xlim()
                bottom, top = ax.axes.get_ylim()
                region = (x_axis[time_idx] > s[0]*1000) & (x_axis[time_idx] < s[1]*1000)
                ax.fill_between(
                    x_axis[time_idx], bottom, top,
                    where=region,
                    alpha=0.1,
                    facecolor='blue'
                )
                    

    plt.gcf().set_size_inches(6.4, 8)

    _plot_traces(cmap_i, igroups, 0, 'i')
    _plot_traces(cmap_e, egroups, len(igroups), 'e')

    if title:
        plt.gcf().get_axes()[0].set_title(title)

    last_ax = plt.gcf().get_axes()[-1]
    last_ax.axes.get_xaxis().set_visible(True)
    last_ax.axes.set_xlabel('Time (ms)')
    bottom = last_ax.axes.get_ylim()[0]
    for s in stim_intervals:
        left, right = last_ax.axes.get_xlim()
        last_ax.axhline(
            y=bottom-3,
            xmin=(s[0]*1000-left)/(right-left),
            xmax=(s[1]*1000-left)/(right-left),
            linewidth=2.0
        )

    if save_as:
        plt.savefig(save_as)

    if show:
        plt.show()

def plot_potentials(cell_vars_h5, cells_file, cell_models_file, **kwargs):
    plot_traces('v/data', cell_vars_h5, cells_file, cell_models_file, yticks=np.arange(-75, 25, step=25), title='Membrane Potential', **kwargs)

def plot_calciums(cell_vars_h5, cells_file, cell_models_file, **kwargs):
    plot_traces('cai/data', cell_vars_h5, cells_file, cell_models_file, title='Calcium influx', **kwargs)

def moving_average(a, n=20):
    """
    https://stackoverflow.com/a/14314054
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    padding = np.zeros(shape=((n)/2,))
    return np.hstack([padding, ret[n - 1:] / n, padding])

def plot_rates(cells_file, cell_models_file, spikes_file, show=True, save_as=None,
               tstart=None, tstop=None, axes=None, double_6e=False, full_layers=False):
    # if axes:
    #     axes = {k: ax.twinx() for k, ax in axes.items()}
    #     for ax in axes.values():
    #         ax.axes.get_yaxis().set_label_position('right')
    #         ax.axes.get_yaxis().set_ticks_position('right')
    # else:
    #     axes = get_layer_ei_axes()
    plt.figure(figsize=(10, 3))
    axes = defaultdict(plt.gca)
    cmap = get_cmap(real=True)
    spike_dfs, spike_times = iter_spike_data(
        cells_file, cell_models_file, spikes_file, None,
        groupby='layer' if full_layers else 'layer_ei'
    )

    tstart = tstart or 0
    tstop = tstop or max(spike_times)
    bins = np.arange(tstart, tstop, 1.0)
    for popn in sorted(spike_dfs.keys()):
        spikedata = spike_dfs[popn]
        ax = axes[popn]
        avg_spikerate = np.mean(np.vstack(
            moving_average(
                np.histogram(
                    spikedata['spike_times'][spikedata['gids'] == gid],
                    bins
                )[0],
                n=5 # # bins (ms) to average for each point
            ) * 1000 # 1/ms --> 1/s = hz
            for gid in set(spikedata['gids'])
        ), axis=0)
        # DEBUG
        offset = len(bins) - len(avg_spikerate)
        if offset > 0:
            avg_spikerate = np.append(avg_spikerate, [0]*offset)
        # END DEBUG
        # ignore first 50ms for yrange calculation
        ymax = max(max(avg_spikerate[50:]), ax.get_ylim()[1])
        ax.plot(bins, avg_spikerate/max(avg_spikerate), color=cmap[popn], linewidth=1,
                # label='23e' if '2' in popn else popn)
                label=popn)
        if popn == '6e' and double_6e:
            ax.plot(avg_spikerate/max(avg_spikerate), color='white', linewidth=0.1)
        ax.set_xlim([tstart, tstop])
        ax.set_ylim([0, 1])

    # axes['6e'].axes.set_ylabel("spike\nrate")
    axes['6e'].axes.set_xlabel("time (ms)")
    axes['6e'].legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)

    if show:
        plt.show()
    

