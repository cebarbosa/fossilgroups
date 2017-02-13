# -*- coding: utf-8 -*-
"""

Created on 10/02/2017

@Author: Carlos Eduardo Barbosa

Calculate Lick indices.
"""

import os

import numpy as np
import pyfits as pf
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve1d
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt

from ppxf.ppxf import reddening_curve

from config import *
from run_ppxf import pPXF, ppload, wavelength_array, set_badpixels
from lick import Lick

def check_intervals(setupfile, bands, vel):
    """ Check which indices are defined in the spectrum. """
    c = 299792.458 # speed of light in km/s
    with open(setupfile) as f:
        lines = [x for x in f.readlines()]
    lines = [x for x in lines if x.strip()]
    intervals = np.array(lines[5:]).astype(float)
    intervals = intervals.reshape((len(intervals)/2, 2))
    bands = np.loadtxt(bands, usecols=(2,7))
    bands *= np.sqrt((1 + vel/c)/(1 - vel/c))
    goodbands = np.zeros(len(bands))
    for i, (b1, b2) in enumerate(bands):
        for (i1, i2) in intervals:
            if i1 < b1 and b2 < i2:
                goodbands[i] = 1
    return np.where(goodbands == 1, 1, np.nan)

def losvd_convolve(spec, losvd):
    """ Apply LOSVD to a given spectra given that both wavelength and spec
     arrays are log-binned. """
    global velscale
    # Convert to pixel scale
    pars = np.copy(losvd)
    pars[:2] /= velscale
    dx = int(np.ceil(np.max(abs(pars[0]) + 5*pars[1])))
    nl = 2*dx + 1
    x = np.linspace(-dx, dx, nl)   # Evaluate the Gaussian using steps of 1/factor pixel
    vel = pars[0]
    w = (x - vel)/(pars[1])
    w2 = w**2
    gauss = np.exp(-0.5*w2)
    profile = gauss/gauss.sum()
    # Hermite polynomials normalized as in Appendix A of van der Marel & Franx (1993).
    # Coefficients for h5, h6 are given e.g. in Appendix C of Cappellari et al. (2002)
    if losvd.size > 2:        # h_3 h_4
        poly = 1 + pars[2]/np.sqrt(3)*(w*(2*w2-3)) \
                 + pars[3]/np.sqrt(24)*(w2*(4*w2-12)+3)
        if len(losvd) == 6:  # h_5 h_6
            poly += pars[4]/np.sqrt(60)*(w*(w2*(4*w2-20)+15)) \
                  + pars[5]/np.sqrt(720)*(w2*(w2*(8*w2-60)+90)-15)
        profile *= poly
    profile /= profile.sum()
    return convolve1d(spec, profile)

def run_lick(group, specs=None, logdir="ppxf_mom4_bias0.7"):
    """ Calculates the Lick indices using the reduced spectra. """
    global velscale
    wdir = os.path.join(data_dir, group)
    os.chdir(wdir)
    if specs == None:
            specs = sorted([x for x in os.listdir(wdir)
                     if x.endswith(".fits")])
    bandsfile = os.path.join(home, "tables/bands.txt")
    bands = np.loadtxt(bandsfile, usecols=np.arange(2,8))
    types = np.loadtxt(bandsfile, usecols=(8,))
    tempfile = os.path.join(home, "MILES/templates/"
                                  "templates_w3540.5_7409.6_res4.7.fits")
    ssps = pf.getdata(tempfile, 0).T
    wssps = np.exp(wavelength_array(tempfile, axis=1, extension=0))
    for spec in specs:
        data = pf.getdata(spec)
        w = wavelength_array(spec, axis=1, extension=0)

        pp = ppload("{}/{}".format(logdir, spec.replace(".fits", "")))
        pp = pPXF(pp, velscale)
        ##################################################################
        # Produce bestfit templates convolved with LOSVD/redshifted
        # Make the combination
        ssps_unbroad_v0 = ssps.dot(pp.w_ssps)
        ssps_broad = losvd_convolve(ssps_unbroad_v0, pp.sol[0])
        ssps_unbroad = losvd_convolve(ssps_unbroad_v0,
                            np.array([pp.sol[0][0], 0.1*velscale, 0, 0]))
        #######################################################################
        # get approximate multiplicative polynomial
        ftempl = interp1d(wssps, ssps_broad, kind="linear",
                      fill_value="extrapolate", bounds_error=False)
        goodpixels = np.arange(len(w), dtype=float)
        gaps= set_badpixels(group)
        for gap in gaps:
            idx = np.where((w > gap[0] ) & (w < gap[1]))[0]
            goodpixels[idx] = np.nan
        goodpixels = goodpixels[~np.isnan(goodpixels)].astype(int)
        fdata = interp1d(w[goodpixels], data[goodpixels], kind="linear",
                      fill_value="extrapolate", bounds_error=False)
        poly = np.poly1d(np.polyfit(w, fdata(w) / ftempl(w), 10))(w)
        #######################################################################
        # Interpolate bestfit templates to obtain linear dispersion
        b0 = interp1d(wssps, ssps_unbroad, kind="linear",
                      fill_value="extrapolate", bounds_error=False)
        b1 = interp1d(wssps, ssps_broad, kind="linear",
                      fill_value="extrapolate", bounds_error=False)
        best_unbroad = b0(w) * poly
        best_broad = b1(w) * poly
        ##################################################################
        # Correct for emission lines
        em = interp1d(pp.w, pp.gas, kind="linear", bounds_error=False,
                      fill_value=0.)
        emission = em(w)
        data -= emission
        ###################################################################
        # Get goodpixels
        # badpix = np.zeros(len(pp.galaxy))
        # badpix[pp.goodpixels] = 1
        # bp = interp1d(pp.w, badpix, kind="linear", bounds_error=False,
        #               fill_value=1.)
        # badpixels = bp(w)
        ###################################################################
        # Run the lector
        l = Lick(w, data, bands, vel=pp.sol[0][0])
        lunb = Lick(w, best_unbroad, bands, vel=pp.sol[0][0])
        lbest = Lick(w, best_broad, bands, vel=pp.sol[0][0])
        ##################################################################
        # LOSVD correction using best fit templates
        ##################################################################
        lickc = correct_lick(l, lunb, lbest, types)
        # plt.plot(np.arange(25) + 0.3, l.classic, "ro", label="data")
        # plt.plot(np.arange(25), lunb.classic, "bo", label="unconvolved")
        # plt.plot(np.arange(25), lbest.classic, "ro", label="bestfit")
        # plt.plot(np.arange(25) + 0.3, lickc, "bo", label="corrected")
        # plt.axhline(y=0, c="k", ls="--")
        # plt.legend()
        # plt.title("{} : {}".format(group, spec))
        # plt.show(block=True)
        ################################################################
        # Convert to string
        ################################################################
        lick = "".join(["{0:14}".format("{0:.5f}".format(x)) for x
                        in l.classic])
        lickc = "".join(["{0:14}".format("{0:.5f}".format(x)) for x
                         in lickc])
    #     # Append to output
        logfile1 = "{}/lick_{}_raw.txt".format(logdir, spec.replace(".fits",
                                                                    ""))
        logfile2 = "{}/lick_{}_losvd_corrected.txt".format(logdir,
                                                    spec.replace(".fits", ""))
        with open(logfile1, "w") as f:
            f.write("{0:16s}".format(pp.name) + lick + "\n")
        with open(logfile2, "w") as f:
            f.write("{0:16s}".format(pp.name) + lickc + "\n")
    return

def mc_lick(group, specs=None, logdir="ppxf_mom4_bias0.7", nsim=100):
    """ Perform Monte Carlo simulations to calculate uncertainties. """
    global velscale
    wdir = os.path.join(data_dir, group)
    os.chdir(wdir)
    if specs == None:
            specs = sorted([x for x in os.listdir(wdir)
                     if x.endswith(".fits")])
    bandsfile = os.path.join(home, "tables/bands.txt")
    bands = np.loadtxt(bandsfile, usecols=np.arange(2,8))
    types = np.loadtxt(bandsfile, usecols=(8,))
    tempfile = os.path.join(home, "MILES/templates/"
                                  "templates_w3540.5_7409.6_res4.7.fits")
    ssps = pf.getdata(tempfile, 0).T
    wssps = np.exp(wavelength_array(tempfile, axis=1, extension=0))
    for spec in specs:
        print "Simulating errors for spectrum {} of system {}".format(spec,
                                                                      group)
        pp = ppload("{}/{}".format(logdir, spec.replace(".fits", "")))
        pp = pPXF(pp, velscale)
        sol = pp.sol[0]
        error = pp.error[0]
        w = wavelength_array(spec, axis=1, extension=0)
        data = pf.getdata(spec)
        #######################################################################
        # get approximate multiplicative polynomial
        ssps_unbroad_v0 = ssps.dot(pp.w_ssps)
        ssps_broad = losvd_convolve(ssps_unbroad_v0, pp.sol[0])
        ftempl = interp1d(wssps, ssps_broad, kind="linear",
                      fill_value="extrapolate", bounds_error=False)
        goodpixels = np.arange(len(w), dtype=float)
        gaps= set_badpixels(group)
        for gap in gaps:
            idx = np.where((w > gap[0] ) & (w < gap[1]))[0]
            goodpixels[idx] = np.nan
        goodpixels = goodpixels[~np.isnan(goodpixels)].astype(int)
        fdata = interp1d(w[goodpixels], data[goodpixels], kind="linear",
                      fill_value="extrapolate", bounds_error=False)
        poly = np.poly1d(np.polyfit(w, fdata(w) / ftempl(w), 10))(w)
        ##################################################################
        # Produce composite stellar population
        csp = ssps.dot(pp.w_ssps)
        ##################################################################
        # Setup simulations
        vpert = np.random.normal(sol[0], np.maximum(error[0], 10),
                                 nsim)
        sigpert = np.random.normal(sol[1], np.maximum(error[1], 10),
                                   nsim)
        h3pert = np.random.normal(sol[2], np.maximum(error[2], 0.02),
                                                     nsim)
        h4pert = np.random.normal(sol[3], np.maximum(error[3], 0.02),
                                                     nsim)
        losvdsim = np.column_stack((vpert, sigpert, h3pert, h4pert))
        licksim = np.zeros((nsim, 25))
        ##################################################################
        for i, losvd in enumerate(losvdsim):
            noise = np.random.normal(0., np.median(pp.noise), len(wssps))
            bestsim = losvd_convolve(csp, losvd)
            bestsim_unb = losvd_convolve(csp, np.array([losvd[0],
                                                    0.1*losvd[1], 0, 0]))
            galsim = bestsim + noise
            ##############################################################
            # Run the lector
            lsim = Lick(wssps, galsim, bands, vel=losvd[0])
            lunb = Lick(wssps, bestsim_unb, bands, vel=losvd[0])
            lbest = Lick(wssps, bestsim, bands, vel=losvd[0])
            ##############################################################
            # LOSVD correction using best fit templates
            ##############################################################
            licksim[i] = correct_lick(lsim, lunb, lbest, types)
        stds = np.zeros(25)
        for i in range(25):
            if np.all(np.isnan(licksim[:,i])):
                stds[i] = np.nan
            else:
                stds[i] = np.nanstd(sigma_clip(licksim[:,i], sigma=5))
        errors = ["{0:.5g}".format(x) for x in stds]
        errors = "".join(["{0:12s}".format(x) for x in errors])
        ###################################################################
        # Storing results
        logfile = "{}/lick_mcerr_{}_nsim{}.txt".format(logdir,
                                            spec.replace(".fits", ""), nsim)
        with open(logfile, "w") as f:
            f.write("{0:16s}".format(pp.name) + errors + "\n")

def correct_lick(lick, unbroad, broad, types):
    """ Make corrections for the broadening in the spectra."""
    w = lick.bands
    Rc = lick.R * (1 - (broad.R - unbroad.R) / broad.R)
    return np.where(types == 1, -2.5 * np.log10(Rc), (1. - Rc) * (w[:,3] - w[:,2]))

def make_tables(group, logdir="ppxf_mom4_bias0.7", nsim=100 ):
    """ Gather information of Lick indices of a given target S/N in a single
    table. """
    global velscale
    wdir = os.path.join(data_dir, group, logdir)
    os.chdir(wdir)
    table1, table2, table3 = [], [], []
    for i in range(22):
        filename1 = "lick_mcerr_spec{}_nsim{}.txt".format(i, nsim)
        if os.path.exists(filename1):
            with open(filename1) as f:
                line = f.readline()
            table1.append(line)
        filename2 = "lick_spec{}_raw.txt".format(i)
        if os.path.exists(filename2):
            with open(filename2) as f:
                line = f.readline()
            table2.append(line)
        filename3 = "lick_spec{}_losvd_corrected.txt".format(i)
        if os.path.exists(filename3):
            with open(filename3) as f:
                line = f.readline()
            table3.append(line)
    outputs = ["lick_mcerr_nsim{}.txt".format(nsim), "lick_raw.txt",
               "lick_losvd_corrected.txt"]
    for i, table in enumerate([table1, table2, table3]):
        with open(outputs[i], "w") as f:
            f.write("".join(table))

if __name__ == "__main__":
    groups = ["RXJ0216", "ESO552"]
    for group in groups:
        # run_lick(group)
        mc_lick(group)
        make_tables(group)