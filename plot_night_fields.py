#!/bin/python
# -*- coding: utf-8 -*-
# plot the tile density of the SPLUS Main Survey along the period available
# in the database
# Author: Fabio R Herpich - CASU/IoA - Cambridge
# Date: 2019-07-01
# Version: 0.0.1

from __future__ import print_function
import os
from astropy.io import ascii
from astropy.time import Time
import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')

# user a parser to get start and end dates and the input file


def parser():
    parser = argparse.ArgumentParser(
        description='Plot the tile density of the SPLUS Main Survey along the \
        period available in the database')
    parser.add_argument('-f', '--file_input', help='Input file', required=True)
    parser.add_argument('-fo', '--footprint', help='Footprint file',
                        required=True)
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


def check_dates(fname, start_date, end_date):
    # create function to check if start_date and end_date are within the input file
    """Check if start_date and end_date are within the input file"""
    # check is start_date and end_date are valid dates
    try:
        start_date = Time(start_date)
        end_date = Time(end_date)
    except ValueError:
        raise ValueError('Invalid date format for start_date or end_date \
                         or date not in calendar')

    f = ascii.read(fname)
    dates = f.keys()
    if start_date not in dates:
        raise ValueError('Start date not in the input file')
    if end_date not in dates:
        raise ValueError('End date not in the input file')

# create function to plot the tile density using the input file


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
    plt.figure(figsize=(10, 5))
    sc = plt.scatter(days[:-1], myear[:-1], c=nfields[:-1], cmap='hot_r',
                     edgecolor='none', marker='s', s=200)  # , vmin=0, vmax=24)

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
        if verbose:
            print('Figure saved successfully.')
    else:
        if verbose:
            print('Plot was not saved. If you want to save it, use the -s option.')

    if verbose:
        print('Showing plot.')
    plt.show()


def main():
    """Main function"""
    # get the arguments
    args = parser()
    finput = args.file_input
    ffoot = args.footprint
    start_date = args.start_date
    end_date = args.end_date
    output = args.output
    verbose = args.verbose
    save = args.save
    workdir = args.workdir

    # check if start_date and end_date are within the input file
    check_dates(finput, start_date, end_date)

    # plot the tile density using the input file
    plot_density(workdir, finput, ffoot, start_date,
                 end_date, output, verbose, save)


if __name__ == '__main__':
    main()
