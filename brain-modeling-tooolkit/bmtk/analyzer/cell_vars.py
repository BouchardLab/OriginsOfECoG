import os
import matplotlib.pyplot as plt

from .io_tools import load_config
from .utils import listify
from bmtk.utils.cell_vars import CellVarsFile

# In the case reports are missing units, try to guess based on
missing_units = {
    'V_m': 'mV',
    'cai': 'mM',
    'v': 'mV'
}


def _get_cell_report(config_file, report_name):
    cfg = load_config(config_file)
    if report_name is not None:
        return cfg.reports[report_name], report_name

    else:
        cell_var_reports = [(r_name, r_dict) for r_name, r_dict in cfg.reports.items()
                            if r_dict['module'] == 'membrane_report']
        if len(cell_var_reports) == 0:
            raise Exception('Could not find any membrane_reports in {}'.format(config_file))

        elif len(cell_var_reports) > 1:
            raise Exception('Found more than one membrane_report, please specify report_name')

        else:
            report_name = cell_var_reports[0][0]
            report = cell_var_reports[0][1]
            report_fname = report['file_name'] if 'file_name' in report else '{}.h5'.format(report_name)
            return report_name, os.path.join(cfg.output_dir, report_fname)


def plot_report(config_file=None, report_file=None, report_name=None, variables=None, gids=None):
    if report_file is None:
        report_name, report_file = _get_cell_report(config_file, report_name)

    var_report = CellVarsFile(report_file)
    variables = listify(variables) if variables is not None else var_report.variables
    gids = listify(gids) if gids is not None else var_report.gids
    time_steps = var_report.time_trace

    def __units_str(var):
        units = var_report.units(var)
        if units == CellVarsFile.UNITS_UNKNOWN:
            units = missing_units.get(var, '')
        return '({})'.format(units) if units else ''

    n_plots = len(variables)
    if n_plots > 1:
        # If more than one variale to plot do so in different subplots
        f, axarr = plt.subplots(n_plots, 1)
        for i, var in enumerate(variables):
            for gid in gids:
                axarr[i].plot(time_steps, var_report.data(gid=gid, var_name=var), label='gid {}'.format(gid))

            axarr[i].legend()
            axarr[i].set_ylabel('{} {}'.format(var, __units_str(var)))
            if i < n_plots - 1:
                axarr[i].set_xticklabels([])

        axarr[i].set_xlabel('time (ms)')

    elif n_plots == 1:
        # For plotting a single variable
        plt.figure()
        for gid in gids:
            plt.plot(time_steps, var_report.data(gid=0, var_name=variables[0]), label='gid {}'.format(gid))
        plt.ylabel('{} {}'.format(variables[0], __units_str(variables[0])))
        plt.xlabel('time (ms)')

    else:
        return

    plt.show()

    #for gid in gids:
    #    plt.plot(times, var_report.data(gid=0, var_name='v'), label='gid {}'.format(gid))


    '''



    plt.ylabel('{} {}'.format('v', units_str))
    plt.xlabel('time (ms)')
    plt.legend()
    plt.show()
    '''



