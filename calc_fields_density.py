#!/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:00:00 2020
Last modified on Wed March 12 2025
Author: Fabio R Herpich
Email: fherpich@lna.br
Company: Laboratorio Nacional de Astrofisica - LNA
Copyrigth: Fabio R Herpich - 2025 - All rights reserved
Github: https://github.com/splus-collab/splus-daily-field-density
Description: Calculate the density of fields for a
given night within a user-defined period of time.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
import pandas as pd
import datetime
from datetime import timedelta
import os
import sys
import argparse
from multiprocessing import Pool
from itertools import repeat
import logging
from time import perf_counter
import subprocess
plt.style.use('classic')

__author__ = 'Fabio R Herpich'
path_to_running_file = os.path.dirname(os.path.abspath(__file__))
latest_tag = subprocess.check_output(
    ['git', 'describe', '--tags'], cwd=path_to_running_file).decode('utf-8').strip()
__version__ = latest_tag.lstrip('v')
__date__ = '2025-03-18'
__email__ = 'fherpich@lna.br'

# create a parser function to get user args for init and end dates


def args_parser():
    """
    Parse the user arguments.
    """
    parser = argparse.ArgumentParser(
        description='Calculate the density of fields for a given night within \
        a user-defined period of time.')
    parser.add_argument('-ns', '--start_date',
                        help='Date to start symulation and/or plot',
                        type=str, required=True)
    parser.add_argument('-ne', '--end_date', help='End date to symulation \
                        and/or plot', type=str, required=True)
    parser.add_argument('-f', '--fields', help='S-PLUS fields file',
                        type=str, required=True)
    parser.add_argument('-o', '--output_file', help='Output filename for the\
                        table containing the field density',
                        type=str, required=False, default='tile_density.csv')
    parser.add_argument('-w', '--workdir', help='Working directory',
                        type=str, required=False, default=os.getcwd())
    parser.add_argument('--outdir', help='Output directory',
                        type=str, required=False, default=os.getcwd())
    parser.add_argument('-n', '--nprocs', help='Number of processes to spawn',
                        type=int, required=False, default=1)
    parser.add_argument('-op', '--output_plot', help='Name of the output plot',
                        required=False, default='tile_density.png')
    parser.add_argument('-sp', '--save_plot', help='Save figure',
                        required=False, action='store_true')
    parser.add_argument('-t', '--oper_type', help='Type of operation. Valid \
                        options are: sym, plot or both', type=str,
                        required=False, default='both')
    parser.add_argument('-s', '--sym_file', help='Ouput file from symulation',
                        required=False, default=None)
    parser.add_argument('-u', '--last_update',
                        help='Date of S-PLUS observations last update.\
                        Format YYYY-MM-DD (default: today)',
                        required=False, type=str)
    parser.add_argument('--loglevel', help='Set the log level. Options are:\
                        DEBUG, INFO, WARNING, ERROR, CRITICAL',
                        required=False, default='INFO')
    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    return args


# use a function to restart the logger before every run

def call_logger(logfile=None,
                loglevel=logging.INFO
                ):
    """
    Set the logging level and the output log file

    Parameters
    ----------
    logfile : str
        The name of the log file
    loglevel : str
        The logging level

    Returns
    -------
    logger : logging.Logger
        The logger object
    """
    logger = logging.getLogger(__name__)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] @%(module)s.%(funcName)s() %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calc_field_track(f: pd.DataFrame,
                     night_starts: str,
                     logger=logging.getLogger(__name__)
                     ):
    """
    Calculate the field track for a given night.

    Parameters
    ----------
    f : pd.DataFrame
        The footprint file
    night_starts : str
        The night to calculate the field track
    logger : logging.Logger
        The logger object

    Returns
    -------
    t : pd.DataFrame
        The table with the field track
    """
    # TODO: Adapt to any site if required by the user
    logger.warning('Site is set to CTIO')
    mysite = EarthLocation(lat=-30.2*u.deg, lon=-70.8 * u.deg, height=2200*u.m)
    utcoffset = 0 * u.hour

    logger.info('night_starts is:', night_starts)
    ns = Time('%s 00:00:00' % night_starts, format='iso').datetime
    inithour = '23:59:00'
    night_starts = ns.strftime("%Y-%m-%d")
    ne = ns + datetime.timedelta(days=1)
    night_ends = ne.strftime("%Y-%m-%d")

    set_time = Time('%s %s' % (night_starts, inithour)) - utcoffset
    midnight = Time('%s 00:00:00' % night_ends) - utcoffset
    delta_midnight = np.linspace(-4, 10, 200)*u.hour

    times_time_overnight = midnight + delta_midnight
    frame_time_overnight = AltAz(obstime=times_time_overnight, location=mysite)
    sunaltazs_time_overnight = get_body('sun', times_time_overnight, mysite).transform_to(
        frame_time_overnight)

    # moon phase
    sun_pos = get_body('sun', set_time, mysite)
    moon_pos = get_body('moon', set_time, mysite)
    elongation = sun_pos.separation(moon_pos)
    moon_phase = np.arctan2(sun_pos.distance*np.sin(elongation),
                            moon_pos.distance - sun_pos.distance*np.cos(elongation))
    moon_brightness = (1. + np.cos(moon_phase))/2.0

    inivalue = delta_midnight[sunaltazs_time_overnight.alt < -
                              18*u.deg].min().value
    endvalue = delta_midnight[sunaltazs_time_overnight.alt < -
                              18*u.deg].max().value
    is_observable = np.zeros(f['NAME'].size).astype(int)

    i = 0
    for index in f.index:
        mycoords = SkyCoord(ra=f['RA'][index], dec=f['DEC'][index],
                            unit=(u.hourangle, u.deg))
        if (moon_pos.separation(mycoords).value < 40):
            logger.info(f['NAME'][index] + ' - moon separation: ' +
                        str(moon_pos.separation(mycoords).value))
        elif (moon_brightness >= .65):
            if 'SPLUS-b' in f['NAME'][index] or 'SPLUS-d' in f['NAME'][index]:
                myaltazs_time_overnight = mycoords.transform_to(
                    frame_time_overnight)
                mask = myaltazs_time_overnight.alt.value > 35.
                mask &= (delta_midnight.value > inivalue) & (
                    delta_midnight.value < endvalue)

                on_the_sky = delta_midnight.value[mask]

                if len(on_the_sky) > 1:
                    time_on_the_sky = on_the_sky.max() - on_the_sky.min()
                    if time_on_the_sky > 2.:
                        is_observable[i] = 1
                        logger.info(f['NAME'][index], night_starts,
                                    'state: on the sky:', time_on_the_sky, 'h')
                    else:
                        logger.info(f['NAME'][index], night_starts, 'state: too \
                            low:', time_on_the_sky, 'h')
                else:
                    logger.info(f['NAME'][index], night_starts,
                                'state: not on the sky')
            else:
                logger.info(f['NAME'][index], night_starts, 'state: moon or separation; \
                    moon', moon_brightness, 'sep:\
                    ', moon_pos.separation(mycoords).value)
        else:
            if 'SPLUS-b' in f['NAME'][index] or 'SPLUS-d' in f['NAME'][index]:
                logger.info(f['NAME'][index], 'is galactic. Skipping')
            else:
                myaltazs_time_overnight = mycoords.transform_to(
                    frame_time_overnight)
                mask = myaltazs_time_overnight.alt.value > 35.
                mask &= (delta_midnight.value > inivalue) & (
                    delta_midnight.value < endvalue)

                on_the_sky = delta_midnight.value[mask]

                if len(on_the_sky) > 1:
                    time_on_the_sky = on_the_sky.max() - on_the_sky.min()
                    if time_on_the_sky > 2.:
                        is_observable[i] = 1
                        logger.info(f['NAME'][index], night_starts,
                                    'state: on the sky:\
                            ', time_on_the_sky, 'h')
                    else:
                        logger.info(f['NAME'][index], night_starts,
                                    'state: too low:', time_on_the_sky, 'h')
                else:
                    logger.info(f['NAME'][index], night_starts,
                                'state: not on the sky')
        i += 1

    t = Table([f['NAME'], is_observable], names=['NAME', night_starts],
              dtype=['S20', 'i1'])

    t = t.to_pandas()
    return t


def generate_date_range(start_date: str,
                        end_date: str,
                        logger=logging.getLogger(__name__)
                        ):
    """
    Generate a range of dates between start_date and end_date

    Parameters
    ----------
    start_date : str
        The start date
    end_date : str
        The end date
    logger : logging.Logger
        The logger object

    Returns
    -------
    date_range : list
        The list of dates between start_date and end_date
    """
    date_range = [start_date.strftime('%Y-%m-%d')]
    current_date = start_date

    while current_date <= end_date:
        current_date += timedelta(days=1)
        date_range.append(current_date.strftime('%Y-%m-%d'))
    logger.debug('Generated %i dates between %s and %s' %
                 (len(date_range), start_date, end_date))

    return date_range


def run_calc_field_track(f: pd.DataFrame,
                         night: str,
                         logger=logging.getLogger(__name__)
                         ):
    """
    Run the calc_field_track function for a given night

    Parameters
    ----------
    f : pd.DataFrame
        The footprint file
    night : str
        The night to calculate the field track
    logger : logging.Logger
        The logger object

    Returns
    -------
    t : pd.DataFrame
        The table with the field track
    """
    logger.info('Starting calc_field_track for %s' % night)
    return calc_field_track(f, night, logger=logger)


def stack_tables(results: list,
                 path_to_final_output: str,
                 logger=logging.getLogger(__name__)
                 ):
    """
    Stack the tables into a single table. Saves the resulting table to disk

    Parameters
    ----------
    results : list
        The list of tables to stack
    path_to_final_output : str
        The path to the final output file
    logger : logging.Logger
        The logger object
    """
    final_table = results[0].copy()
    for df in results[1:]:
        final_table = final_table.merge(df, on='NAME', how='outer')

    logger.info(' Stacking the tables into %s' %
                path_to_final_output)
    final_table.to_csv(path_to_final_output, index=False)


def main_sym(args,
             logger=logging.getLogger(__name__)
             ):
    """
    Run the simulation for the given dates

    Parameters
    ----------
    args : argparse.Namespace
        The user arguments
    logger : logging.Logger
        The logger object
    """
    workdir = args.workdir
    night_starts = args.start_date
    night_ends = args.end_date
    field_file = args.fields
    output_file = args.output_file
    num_procs = args.nprocs

    if output_file is None:
        logger.debug('Creating output file name')
        path_to_final_output = os.path.join(args.workdir,
                                            args.output_file)
    else:
        logger.info('Using output file name %s' % output_file)
        path_to_final_output = os.path.join(workdir, output_file)

    if os.path.isfile(path_to_final_output):
        logger.warning('The output file %s already exists. Exiting.' %
                       path_to_final_output)
        return
    else:
        logger.info('The output file %s does not exist. Running silulation\
                    for %s to %s' %
                    (path_to_final_output, night_starts, night_ends))

    try:
        footprint = pd.read_csv(os.path.join(workdir, field_file))
        logger.info('Footprint file read successfully')
    except FileNotFoundError:
        logger.error('Footprint file not found. Exiting.')
        return
    mask_todo = footprint['STATUS'] == -5
    mask_todo |= footprint['STATUS'] == -2
    mask_todo |= footprint['STATUS'] == -1
    mask_todo |= footprint['STATUS'] == 0
    mask_todo |= footprint['STATUS'] == 3
    mask_todo |= footprint['STATUS'] == 5

    f = footprint[mask_todo]

    # calculate a range of dates between night_starts and night_ends
    start_date = Time(night_starts)
    end_date = Time(night_ends)
    date_range = generate_date_range(start_date, end_date)
    logger.debug('Dates generated successfully')

    if num_procs < 2 or args.loglevel == 'DEBUG':
        logger.warning(
            'Running in single process mode. This may take a while.')
        results = []
        for date in date_range:
            out_df = run_calc_field_track(f, date, logger=logger)
            results.append(out_df)
            if args.loglevel == 'DEBUG':
                logger.debug('Results for %s: %s' % (date, out_df))
                import pdb
                pdb.set_trace()
        logger.info('All dates calculated successfully')
    else:
        logger.info(
            'Running in multi-process mode with %i processes' % num_procs)
        with Pool(num_procs) as pool:
            results = pool.starmap(run_calc_field_track,
                                   zip(repeat(f), date_range, repeat(logger)))
            pool.close()
            pool.join()
        logger.info('All dates calculated successfully')

    logger.info(' Stacking the tables into %s' % path_to_final_output)
    stack_tables(results, path_to_final_output, logger=logger)


def check_dates(workdir: str,
                fname: str,
                start_date: str,
                end_date: str
                ):
    """
    Check if the start_date and end_date are within the input file

    Parameters
    ----------
    workdir : str
        The working directory
    fname : str
        The footprint input file name
    start_date : str
        The start date
    end_date : str
        The end date
    """
    try:
        start_date = Time(start_date)
        end_date = Time(end_date)
    except ValueError:
        raise ValueError(
            'Invalid date format for start_date or end_date or date not in calendar')

    f = pd.read_csv(os.path.join(workdir, fname))
    dates = f.keys()
    if start_date.strftime('%Y-%m-%d') not in dates:
        raise ValueError('Start date not in the input file')
    if end_date.strftime('%Y-%m-%d') not in dates:
        raise ValueError('End date not in the input file')


def plot_density(args: argparse.Namespace,
                 logger=logging.getLogger(__name__)
                 ):
    """
    Plot the tile density

    Parameters
    ----------
    args : argparse.Namespace
        The user arguments
    logger : logging.Logger
        The logger object
    """
    # define last date of update
    last_update = datetime.datetime.now().strftime(
        "%Y/%m/%d") if args.last_update is None else args.last_update
    # read the input file and the footprint file
    fi = pd.read_csv(os.path.join(args.workdir, args.sym_file))
    # foot = pd.read_csv(os.path.join(args.workdir, args.fields))
    logger.info("Input file and footprint file read successfully.")

    # get the list of dates
    dates = [Time(key) for key in fi.keys() if '-' in key]
    logger.debug("Dates extracted from the input file.")
    # make a list of days
    days = [date.datetime.day for date in dates if args.start_date <=
            date <= args.end_date]
    logger.debug("Days extracted within the specified range.")
    # make a list of months
    months = np.unique(
        [date.datetime.month for date in dates if args.start_date <= date <= args.end_date])
    logger.debug("Months extracted within the specified range.")
    # make a list of year-month
    myear = [date.datetime.strftime("%Y-%m")
             for date in dates if args.start_date <= date <= args.end_date]
    logger.debug("Year-month extracted within the specified range.")

    nfields = [int(fi[date.datetime.strftime("%Y-%m-%d")].sum())
               for date in dates if args.start_date <= date <= args.end_date]
    logger.debug("Number of fields calculated within the specified range.")
    logger.debug("Plotting the tile density.")

    # plot the tile density
    h = int(len(months) / 2) if int(len(months) / 2) >= 1 else 1
    plt.figure(figsize=(16, h))
    sc = plt.scatter(days[:-1], myear[:-1], c=nfields[:-1], cmap='hot_r',
                     edgecolor='none', marker='s', s=400)

    cb = plt.colorbar(sc)
    cb.set_label('$\mathrm{Ntiles / night}$', fontsize=18)

    plt.xlabel('$\mathrm{Day}$', fontsize=18)
    plt.ylabel('$\mathrm{Year-month}$', fontsize=18)
    plt.xlim(0, 32)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.title('$\mathrm{Total\ of\ tiles\ remaining\ as\ of\ %s:\ %i}$' %
              (last_update, fi['NAME'].size), fontsize=16)

    plt.tight_layout()

    if args.save_plot:
        outputname = args.output_plot if args.output_plot is not None else 'tile_density.png'
        logger.info('Saving figure output to %s' % outputname)
        plt.savefig(os.path.join(args.outdir, outputname), format='png',
                    bbox_inches='tight', dpi=150)
        logger.info('Figure saved successfully to %s' % outputname)
    else:
        logger.warning(
            'Plot was not saved. If you want to save it, use the -s option.')

    logger.debug('Showing plot.')
    plt.show()


def main_plot(args: argparse.Namespace,
              logger=logging.getLogger(__name__)
              ):
    """
    Run the plot function
    """
    # get the arguments
    finput = args.output_file
    start_date = args.start_date
    end_date = args.end_date
    workdir = args.workdir

    # check if start_date and end_date are within the input file
    check_dates(workdir, finput, start_date, end_date)

    plot_density(args, logger=logger)


if __name__ == '__main__':
    start_time = perf_counter()
    args = args_parser()
    logger = call_logger(loglevel=args.loglevel)
    if args.oper_type not in ['plot', 'sym', 'both']:
        raise ValueError('Invalid operation type. Use plot, sym or both.')
    else:
        operation = args.oper_type

    if operation == 'plot':
        if args.sym_file is None:
            sym_path = os.path.join(args.workdir, args.output_file)
            if os.path.isfile(sym_path):
                args.sym_file = sym_path
            else:
                raise ValueError(
                    'Symulation file not found or dates are out of range.')

        main_plot(args, logger=logger)

    elif operation == 'sym':
        main_sym(args, logger=logger)
        print('Simulation took:', perf_counter() - start_time)
    else:
        main_sym(args, logger=logger)
        print('Simulation took:', perf_counter() - start_time)
        sym_path = os.path.join(args.workdir, args.output_file)
        args.sym_file = sym_path if args.sym_file is None else args.sym_file
        args.output_plot = 'tile_density_%s_%s.png' % (
            args.start_date, args.end_date) if args.output_plot is None else args.output_plot
        main_plot(args, logger=logger)

    print('Total time:', perf_counter() - start_time)
