#!/bin/python
# -*- coding: utf-8 -*-
# plot the tile density of the SPLUS Main Survey along the period available
# in the database
# Author: Fabio R Herpich - CASU/IoA - Cambridge
# Date: 2019-07-01
# Version: 0.0.1

from __future__ import print_function
import os
import numpy as np
from astropy.io import ascii
from astropy.time import Time
import argparse
import matplotlib.pyplot as plt
# import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

# user a parser to get start and end dates and the input file


def parser():
    parser = argparse.ArgumentParser(
        description='Plot the tile density of the SPLUS Main Survey along the \
        period available in the database')
    parser.add_argument('-f', '--file', help='Input file', required=True)
    parser.add_argument('-ns', '--start_date',
                        help='Start date', required=True)
    parser.add_argument('-ne', '--end_date', help='End date', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=False)
    parser.add_argument('-v', '--verbose', help='Verbose mode',
                        required=False, action='store_true')
    parser.add_argument('-s', '--save', help='Save figure',
                        required=False, action='store_true')
    parser.add_argument('-w', '--workdir', help='Working directory',
                        required=False, default=os.getcwd())
    args = parser.parse_args()
    return args

# create function to check if start_date and end_date are within the input file


def check_dates(fname, start_date, end_date):
    """Check if start_date and end_date are within the input file"""
    f = ascii.read(fname)
    dates = f.keys()
    if start_date not in dates:
        print('Start date not in the input file')
        return False
    if end_date not in dates:
        print('End date not in the input file')
        return False


f = ascii.read('tiles_new20190701_clean_2019B2020A.csv')

nfields = []
days = []
myear = []

mask = (f['STATUS'] != 1) & (f['STATUS'] != 2) & (f['STATUS'] != 4)
for key in f.keys()[6:]:
    ndate = Time(key)
    days.append(ndate.datetime.day)
    myear.append(ndate.datetime.strftime("%Y-%m"))
    nfields.append(int(f[key][mask].sum()/100))

plt.figure(figsize=(10, 5))
# sc = plt.scatter(days[:-1], myear[:-1], c=np.log10(nfields[:-1]), cmap='hot_r',
sc = plt.scatter(days[:-1], myear[:-1], c=nfields[:-1], cmap='hot_r',
                 edgecolor='none', marker='s', s=200, vmin=0, vmax=24)


# for i in range(len(nfields)):
#    plt.add_patch(Rectangle((days[i] - .5,
#                            myear[i] - .5), 1., 1.,
#                           ec='k', fc=,
#                           alpha=1, zorder=-1))

cb = plt.colorbar(sc, ticks=np.arange(0, 25, 2))
cb.set_label('$\mathrm{Ntiles / night}$', fontsize=18)

plt.xlabel('$\mathrm{Day}$', fontsize=18)
plt.ylabel('$\mathrm{Year-month}$', fontsize=18)
plt.xlim(0, 32)
plt.gca().invert_yaxis()
plt.grid()
plt.title('$\mathrm{Total\ of\ tiles\ remaining:\ %i}$' % f['PID'][mask].size)

plt.tight_layout()

plt.show(block=False)
