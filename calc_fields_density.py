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
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_moon, get_sun
import datetime
import os
import argparse
import concurrent.futures


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
    args = parser.parse_args()
    return args


# Calculate the if a field is observable for a given night
def calc_field_track(workdir, f, night_starts, night_ends):
    mysite = EarthLocation(lat=-30.2*u.deg, lon=-70.8 *
                           u.deg, height=2200*u.m)  # pyright: ignore
    utcoffset = 0 * u.hour  # pyright: ignore

    ns = Time('%s 00:00:00' % night_starts).datetime
    inithour = '23:59:00'
    night_starts = ns.strftime("%Y-%m-%d")
    ne = ns + datetime.timedelta(days=1)
    night_ends = ne.strftime("%Y-%m-%d")

    set_time = Time('%s %s' % (night_starts, inithour)) - utcoffset
    midnight = Time('%s 00:00:00' % night_ends) - utcoffset
    delta_midnight = np.linspace(-4, 10, 200)*u.hour
    # frame_tonight = AltAz(obstime=midnight + delta_midnight,
    #                       location=mysite)

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
                              18*u.deg].min().value  # pyright: ignore
    endvalue = delta_midnight[sunaltazs_time_overnight.alt < -
                              18*u.deg].max().value  # pyright: ignore
    is_observable = np.zeros(f['NAME'].size)

    for i in range(f['RA'].size):
        mycoords = SkyCoord(ra=f['RA'][i], dec=f['DEC'][i],
                            unit=(u.hourangle, u.deg))
        if (moon_pos.separation(mycoords).value < 50):
            # is_observable[i] = 0
            print(f['NAME'][i], night_starts, 'state: moon or separation; moon',
                  moon_brightness, 'sep:', moon_pos.separation(mycoords).value)
        elif (moon_brightness >= .85):
            if (f['NAME'][i].split('-')[-1][0] not in ['b', 'd']):
                myaltaz = mycoords.transform_to(
                    AltAz(obstime=set_time, location=mysite))

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
                        print(f['NAME'][i], night_starts,
                              'state: on the sky:', time_on_the_sky, 'h')
                    else:
                        print(f['NAME'][i], night_starts,
                              'state: too low:', time_on_the_sky, 'h')
                else:
                    print(f['NAME'][i], night_starts, 'state: not on the sky')
            else:
                print(f['NAME'][i], night_starts, 'state: moon or separation; moon',
                      moon_brightness, 'sep:', moon_pos.separation(mycoords).value)
        else:
            # myaltaz = mycoords.transform_to(
            #     AltAz(obstime=set_time, location=mysite))
            #
            # myaltazs_tonight = mycoords.transform_to(frame_tonight)

            myaltazs_time_overnight = mycoords.transform_to(
                frame_time_overnight)
            mask = myaltazs_time_overnight.alt.value > 35.  # pyright: ignore
            mask &= (delta_midnight.value > inivalue) & (
                delta_midnight.value < endvalue)

            on_the_sky = delta_midnight.value[mask]

            if len(on_the_sky) > 1:
                time_on_the_sky = on_the_sky.max() - on_the_sky.min()
                if time_on_the_sky > 2.:
                    is_observable[i] = 1
                    print(f['NAME'][i], night_starts,
                          'state: on the sky:', time_on_the_sky, 'h')
                else:
                    print(f['NAME'][i], night_starts,
                          'state: too low:', time_on_the_sky, 'h')
            else:
                print(f['NAME'][i], night_starts, 'state: not on the sky')

    outdir = os.path.join(workdir, 'outputs')
    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)
    tab_name = os.path.join(outdir, night_starts + '.csv')
    t = Table([f['NAME'], is_observable], names=['NAME', night_starts],
              dtype=['S20', 'i1'])

    print('saving table', tab_name)

    ascii.write(t, tab_name, format='csv', overwrite=True)


def main():
    args = parser()
    workdir = args.workdir
    night_starts = args.night_starts
    night_ends = args.night_ends
    field_file = args.fields

    f = ascii.read(field_file)
    calc_field_track(workdir, f, night_starts, night_ends)


if __name__ == '__main__':
    main()
