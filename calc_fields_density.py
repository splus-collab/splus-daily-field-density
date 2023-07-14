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


# create a parser function to get user args for init and end dates
def parser():
    """Take care of all the argparse stuff."""
    parser = argparse.ArgumentParser(
        description='Calculate the density of fields for a given night within a user-defined period of time.')
    parser.add_argument('-ns', '--night_starts',
                        help='Night starts', type=str, required=True)
    parser.add_argument('-ne', '--night_ends',
                        help='Night ends', type=str, required=True)
    parser.add_argument('-f', '--fields', help='S-PLUS fields file',
                        type=str, required=True)
    parser.add_argument('-w', '--workdir', help='Working directory',
                        type=str, required=False, default=os.getcwd())
    parser.add_argument('-n', '--ncores', help='Number of cores',
                        type=int, required=False, default=1)
    args = parser.parse_args()
    return args


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
    mysite = EarthLocation(lat=-30.2*u.deg, lon=-70.8 *
                           u.deg, height=2200*u.m)
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
    date_range = []
    current_date = start_date

    while current_date <= end_date:
        date_range.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

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
                    "%Y-%m-%d %H:%M:%S") + ' - table', tab_name, 'already \
                    exists')


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


def main():
    """Main function."""
    call_logger()
    logger = logging.getLogger(__name__)

    args = parser()
    workdir = args.workdir
    night_starts = args.night_starts
    night_ends = args.night_ends
    field_file = args.fields

    path_to_final_output = os.path.join(workdir,
                                        'tiles_nc_density_%s-%s.csv' %
                                        (night_starts, night_ends))
    if os.path.isfile(path_to_final_output):
        logger.info(' ' +
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                    ' The output file %s already exists. Exiting.' %
                    path_to_final_output)
        sys.exit()
    else:
        logger.info(' ' +
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                    ' The output file %s does not exist. Running silulation\
                    for %s to %s' %
                    (path_to_final_output, night_starts, night_ends))

    f = ascii.read(field_file)

    # calculate a range of dates between night_starts and night_ends
    start_date = Time(night_starts)
    end_date = Time(night_ends)
    date_range = generate_date_range(start_date, end_date)

    if args.ncores is None:
        num_procs = 4
    else:
        num_procs = args.ncores

    if len(date_range) % num_procs > 0:
        increase_to = len(date_range) / num_procs + 1
        i = 0
        while i < (increase_to * num_procs - len(date_range)):
            date_range.append('dummydate')
            i += 1
        else:
            logger.info(' ' + datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + ' Dates already meet the requirement\
                of num_procs')
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


if __name__ == '__main__':
    main()
