#!/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:00:00 2020
Last modified on Mon Apr  6 11:00:00 2020
Author: Fabio R Herpich
Github: https://github.com/splus-collab/splus-daily-field-density
Description: Calculate the density of fields for a
given night within a user-defined period of time.
"""

from __future__ import print_function
import numpy as np
import astropy
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_moon, get_sun
import datetime
from datetime import timedelta
import os
import sys
import argparse
import multiprocessing as mp
import logging
# import warnings
import colorlog
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')

# create a parser function to get user args for init and end dates


def parser():
    """Take care of all the argparse stuff."""
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
    parser.add_argument('-o', '--output_file', help='Output file',
                        type=str, required=False)
    parser.add_argument('-w', '--workdir', help='Working directory',
                        type=str, required=False, default=os.getcwd())
    parser.add_argument('-n', '--ncores', help='Number of cores',
                        type=int, required=False, default=1)
    parser.add_argument('-op', '--output_plot', help='Name of the output plot',
                        required=False)
    parser.add_argument('-v', '--verbose', help='Verbose mode',
                        required=False, action='store_true')
    parser.add_argument('-sp', '--save_plot', help='Save figure',
                        required=False, action='store_true')
    # add argument to chose if calc, plot or both
    parser.add_argument('-t', '--oper_type', help='Type of operation. Valid \
                        options are: calc, plot or both', type=str,
                        required=False, default='both')
    # add argument to require sym_file if --oper_type=plot
    parser.add_argument('-s', '--sym_file', help='Ouput file from symulation',
                        required=False)

    args = parser.parse_args()

    return args


# use a function to restart the logger before every run

def call_logger():
    # reset logging config
    logging.shutdown()
    logging.root.handlers.clear()

    # configure the module with colorlog
    logger = colorlog.getLogger()
    logger.setLevel(logging.INFO)

    # create a formatter with green color for INFO level
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:%(name)s:%(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        })

    # create handler and set the formatter
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(ch)


# Calculate the if a field is observable for a given night
def calc_field_track(f, night_starts, tab_name):
    """
    Calculate the if a field is observable for a given night.
    """
    call_logger()
    logger = logging.getLogger(__name__)

    # warn user that site is hardcoded to CTIO
    logger.warning(datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S") + ' - Site is set to CTIO')
    # warnings.warn(datetime.datetime.now().strftime(
    #     "%Y-%m-%d %H:%M:%S") + ' - Site is set to CTIO')
    mysite = EarthLocation(lat=-30.2*u.deg, lon=-70.8 * u.deg, height=2200*u.m)
    utcoffset = 0 * u.hour

    ns = Time('%s 00:00:00' % night_starts).datetime
    inithour = '23:59:00'
    night_starts = ns.strftime("%Y-%m-%d")
    ne = ns + datetime.timedelta(days=1)
    night_ends = ne.strftime("%Y-%m-%d")

    set_time = Time('%s %s' % (night_starts, inithour)) - utcoffset
    midnight = Time('%s 00:00:00' % night_ends) - utcoffset
    delta_midnight = np.linspace(-4, 10, 200)*u.hour

    times_time_overnight = midnight + delta_midnight
    frame_time_overnight = AltAz(obstime=times_time_overnight, location=mysite)
    sunaltazs_time_overnight = get_sun(
        times_time_overnight).transform_to(frame_time_overnight)

    # moon phase
    sun_pos = get_sun(set_time)
    moon_pos = get_moon(set_time)
    elongation = sun_pos.separation(moon_pos)
    moon_phase = np.arctan2(sun_pos.distance*np.sin(elongation),
                            moon_pos.distance - sun_pos.distance*np.cos(elongation))
    moon_brightness = (1. + np.cos(moon_phase))/2.0

    inivalue = delta_midnight[sunaltazs_time_overnight.alt < -
                              18*u.deg].min().value
    endvalue = delta_midnight[sunaltazs_time_overnight.alt < -
                              18*u.deg].max().value
    is_observable = np.zeros(f['NAME'].size)

    for i in range(f['RA'].size):
        mycoords = SkyCoord(ra=f['RA'][i], dec=f['DEC'][i],
                            unit=(u.hourangle, u.deg))
        if (moon_pos.separation(mycoords).value < 50):
            logger.info(' ' + datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S") + ' - ' + f['NAME'][i] + ' - moon \
                separation: ' + str(moon_pos.separation(mycoords).value))
        elif (moon_brightness >= .85):
            if (f['NAME'][i].split('-')[-1][0] not in ['b', 'd']):
                # myaltaz = mycoords.transform_to(
                #     AltAz(obstime=set_time, location=mysite))

                # myaltazs_tonight = mycoords.transform_to(frame_tonight)

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
                        print(datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"),
                            f['NAME'][i], night_starts,
                            'state: on the sky:', time_on_the_sky, 'h')
                    else:
                        print(datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"),
                            f['NAME'][i], night_starts, 'state: too \
                            low:', time_on_the_sky, 'h')
                else:
                    print(datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"),
                        f['NAME'][i], night_starts, 'state: not on the sky')
            else:
                print(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"),
                    f['NAME'][i], night_starts, 'state: moon or separation; \
                    moon', moon_brightness, 'sep:\
                    ', moon_pos.separation(mycoords).value)
        else:
            # myaltaz = mycoords.transform_to(
            #     AltAz(obstime=set_time, location=mysite))
            #
            # myaltazs_tonight = mycoords.transform_to(frame_tonight)

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
                    print(datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"),
                        f['NAME'][i], night_starts,
                        'state: on the sky:\
                          ', time_on_the_sky, 'h')
                else:
                    print(datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"),
                        f['NAME'][i], night_starts,
                        'state: too low:\
                          ', time_on_the_sky, 'h')
            else:
                print(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"),
                    f['NAME'][i], night_starts, 'state: not on the sky')

    t = Table([f['NAME'], is_observable], names=['NAME', night_starts],
              dtype=['S20', 'i1'])

    print(datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"), 'Saving table', tab_name)

    ascii.write(t, tab_name, format='csv', overwrite=True)


def generate_date_range(start_date, end_date):
    date_range = [start_date.strftime('%Y-%m-%d')]
    current_date = start_date

    while current_date <= end_date:
        current_date += timedelta(days=1)
        date_range.append(current_date.strftime('%Y-%m-%d'))

    return date_range


def run_calc_field_track(workdir, f, night_range):
    """Run the calc_field_track function for a range of nights."""

    call_logger()
    logger = logging.getLogger(__name__)

    outdir = os.path.join(workdir, 'outputs')
    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)

    for night in night_range:
        if night == 'dummydate':
            continue
        else:
            tab_name = os.path.join(outdir, night + '.csv')
            if os.path.isfile(tab_name) is False:
                calc_field_track(f, night, tab_name)
            else:
                logger.info(' ' + datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S") + ' - table %s already exists' %
                    tab_name)


def stack_tables(workdir, night_range, path_to_final_output):
    """Stack the tables for the different nights."""

    call_logger()
    logger = logging.getLogger(__name__)

    outdir = os.path.join(workdir, 'outputs')
    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)

    for night in night_range:
        if night == 'dummydate':
            continue
        else:
            tab_name = os.path.join(outdir, night + '.csv')
            if os.path.isfile(tab_name) is False:
                logger.error(' ' + datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S") + ' - Table', tab_name, 'does \
                    not exist')
                sys.exit(1)
            else:
                t = ascii.read(tab_name)
                if night == night_range[0]:
                    t0 = t
                else:
                    t0 = astropy.table.join(t0, t, keys='NAME',
                                            join_type='outer')

    t0.write(os.path.join(workdir, path_to_final_output), format='csv',
             overwrite=True)


def main_sym(args):
    """Main function."""
    call_logger()
    logger = logging.getLogger(__name__)

    workdir = args.workdir
    night_starts = args.start_date
    night_ends = args.end_date
    field_file = args.fields
    output_file = args.output_file
    num_procs = args.ncores

    if output_file is None:
        path_to_final_output = os.path.join(workdir,
                                            'tiles_nc_density_%s_%s.csv' %
                                            (night_starts, night_ends))
    else:
        path_to_final_output = os.path.join(workdir, output_file)

    if os.path.isfile(path_to_final_output):
        logger.info(' ' +
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                    ' The output file %s already exists. Exiting.' %
                    path_to_final_output)
        return
    else:
        logger.info(' ' +
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                    ' The output file %s does not exist. Running silulation\
                    for %s to %s' %
                    (path_to_final_output, night_starts, night_ends))

    f = ascii.read(os.path.join(workdir, field_file))

    # calculate a range of dates between night_starts and night_ends
    start_date = Time(night_starts)
    end_date = Time(night_ends)
    date_range = generate_date_range(start_date, end_date)

    # find the for which len(date_range) % num_procs == 0

    if len(date_range) % num_procs > 0:
        increase_to = int(np.ceil(len(date_range) / num_procs)) * num_procs
        print(datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'),
            'Increasing the number of days to', increase_to)
        while len(date_range) < increase_to:
            print(datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'),
                'Including a dummy date to make the number of days divisible\
                by', num_procs)
            date_range.append('dummydate')
        else:
            logger.info(' ' + datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + ' Dates now meet the requirement\
                of ncores')
    dates = np.array(date_range).reshape(
        (num_procs, int(len(date_range) / num_procs)))
    print(datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
        'Calculating for a total of', len(date_range), 'days')
    print(datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
        'Using', num_procs, 'processes')

    jobs = []
    for date_list in dates:
        process = mp.Process(target=run_calc_field_track,
                             args=(workdir, f, date_list))
        jobs.append(process)

    # start the processes
    logger.info(' ' + datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + ' Starting the processes')
    for j in jobs:
        j.start()

    # ensure all processes have finished execution
    for j in jobs:
        j.join()

    print('All done!')

    # stack the tables
    logger.info(' ' + datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + ' Stacking the tables into \
        %s' % path_to_final_output)
    stack_tables(workdir, date_range, path_to_final_output)


def check_dates(workdir, fname, start_date, end_date):
    # create function to check if start_date and end_date are within the input file
    """Check if start_date and end_date are within the input file"""
    # check is start_date and end_date are valid dates
    try:
        start_date = Time(start_date)
        end_date = Time(end_date)
    except ValueError:
        raise ValueError(
            'Invalid date format for start_date or end_date or date not in calendar')

    f = ascii.read(os.path.join(workdir, fname))
    dates = f.keys()
    if start_date not in dates:
        raise ValueError('Start date not in the input file')
    if end_date not in dates:
        raise ValueError('End date not in the input file')


def plot_density(workdir, finput, ffoot, start_date, end_date, output, verbose, save):
    """Plot the tile density using the input file"""
    # read the input file and the footprint file
    fi = ascii.read(os.path.join(workdir, finput))
    foot = ascii.read(os.path.join(workdir, ffoot))
    if verbose:
        print("Input file and footprint file read successfully.")

    # get the list of dates
    dates = [Time(key) for key in fi.keys() if '-' in key]
    if verbose:
        print("Dates extracted from the input file.")
    # make a list of days
    days = [date.datetime.day for date in dates if start_date <= date <= end_date]
    if verbose:
        print("Days extracted within the specified range.")
    # make a list of months
    months = np.unique(
        [date.datetime.month for date in dates if start_date <= date <= end_date])
    if verbose:
        print("Months extracted within the specified range.")
    # make a list of year-month
    myear = [date.datetime.strftime("%Y-%m")
             for date in dates if start_date <= date <= end_date]
    if verbose:
        print("Year-month extracted within the specified range.")

    # make mask all non-observed tiles
    mask = (foot['STATUS'] == 0) | (foot['STATUS'] == 3)
    if verbose:
        print("Mask created for non-observed tiles.")

    # make a list of number of fields
    nfields = [int(fi[date.datetime.strftime("%Y-%m-%d")][mask].sum())
               for date in dates if start_date <= date <= end_date]
    if verbose:
        print("Number of fields calculated within the specified range.")
        print("Plotting the tile density.")

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
              (datetime.datetime.now().strftime("%Y/%m/%d"),
               foot['NAME'][~mask].size), fontsize=16)

    plt.tight_layout()

    if save:
        outputname = output if output else 'tile_density_%s_%s.png' % (
            start_date, end_date)
        # make verbose
        if verbose:
            print('Saving figure output to %s' % outputname)
        plt.savefig(os.path.join(workdir, outputname), format='png',
                    bbox_inches='tight', dpi=150)
        print('Figure saved successfully to %s' % outputname)
    else:
        if verbose:
            print('Plot was not saved. If you want to save it, use the -s option.')

    if verbose:
        print('Showing plot.')
    plt.show()


def main_plot(args):
    """Main function"""
    # get the arguments
    finput = args.sym_file
    ffoot = args.fields
    start_date = args.start_date
    end_date = args.end_date
    output = args.output_plot
    verbose = args.verbose
    save = args.save_plot
    workdir = args.workdir

    # check if start_date and end_date are within the input file
    check_dates(workdir, finput, start_date, end_date)

    # plot the tile density using the input file
    plot_density(workdir, finput, ffoot, start_date,
                 end_date, output, verbose, save)


if __name__ == '__main__':
    args = parser()
    if args.oper_type not in ['plot', 'sym', 'both']:
        raise ValueError('Invalid operation type. Use plot, sym or both.')
    else:
        operation = args.oper_type

    if operation == 'plot':
        if args.sym_file is None:
            sym_path = os.path.join(args.workdir,
                                    'tiles_nc_density_%s_%s.csv' %
                                    (args.start_date, args.end_date))
            if os.path.isfile(sym_path):
                args.sym_file = sym_path
            else:
                raise ValueError(
                    'Symulation file not found or dats are out of range.')

        main_plot(args)

    elif operation == 'sym':
        main_sym(args)
    else:
        main_sym(args)
        sym_path = os.path.join(args.workdir,
                                'tiles_nc_density_%s_%s.csv' %
                                (args.start_date, args.end_date))
        args.sym_file = sym_path if args.sym_file is None else args.sym_file
        args.output_plot = 'tile_density_%s_%s.png' % (
            args.start_date, args.end_date) if args.output_plot is None else args.output_plot
        main_plot(args)
