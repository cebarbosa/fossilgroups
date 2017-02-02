# -*- coding: utf-8 -*-
"""

Created on 04/05/16

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
import ppxf.ppxf_util as util
from scipy.ndimage.filters import gaussian_filter1d

from config import *

def filenames_miles10(age, metal, alpha, imf="bi1.30"):
    """ Returns the name of files for the MILES 10 library. """
    msign = "p" if metal >= 0. else "m"
    esign = "p" if alpha >= 0. else "m"
    azero = "0" if age < 10. else ""
    return "M{0}Z{1}{2:.2f}T{6}{3:02.4f}_iT{4}{5:2.2f}_E{4}{5:1.2f}" \
           ".fits".format(imf, msign, abs(metal),age, esign, abs(alpha), azero)

def ranges_miles10(sampling="normal"):
    """ Return valid values for SSPs. """
    if sampling == "all":
        metals = np.array([-0.66, -0.35, -0.25, 0.06, 0.15, 0.26, 0.40])
        age = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                          0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                          0.6, 0.7, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2.,
                          2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4.,
                          4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5,
                          10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.])
        alpha = np.array([0., 0.4])
        return age, metals, alpha
    if sampling == "normal":
        metals = np.array([-0.66, -0.35, -0.25, 0.06, 0.15, 0.26, 0.40])
        age = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1., 2., 3., 4., 5.,
                        6., 7., 8., 9., 10., 11., 12., 13., 14.])
        alpha = np.array([0., 0.4])
        return age, metals, alpha

def load_stars(filenames):
    """ Load files with stellar library used as templates. """
    h = pf.getheader(filenames[0])
    wave = h['CRVAL1'] +   h['CDELT1'] * (np.arange((h['NAXIS1'])) + 1 -
                                          h['CRPIX1'])
    templates = np.empty((wave.size,len(filenames)))
    for j in range(len(filenames)):
        templates[:,j] =  pf.getdata(filenames[j])
    templates /= np.median(templates)
    return wave, templates

def make_emission_template(line, fwhm, wave, resamp=11):
    deltaw = wave[1] - wave[0]
    wave_res= np.linspace(wave[0]-deltaw/2., wave[-1] + deltaw/2.,
                          len(wave+1)*resamp)
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
    spec = np.exp(-(wave_res - line)**2 / (2 * sigma * sigma))
    spec = np.sum(spec.reshape(len(wave), resamp), axis=1) / resamp
    return spec

def make_templates(wave, stars, emission, velscale, w1, w2, fits):
    """ Prepare file with templates. """
    idx = np.where((wave >= w1) & (wave <= w2))[0]
    wave = wave[idx]
    lrange = np.array([wave[0], wave[-1]])
    stars = stars[idx]
    emission = emission[idx]
    # Rebin data
    tmp, logLam, velscale = util.log_rebin(lrange, stars[:,0],
                                               velscale=velscale)
    tmp2, logLam2, velscale = util.log_rebin(lrange, emission[:,0],
                                               velscale=velscale)
    starsnew = np.zeros((len(logLam), len(stars[0])))
    for i in range(len(stars[0])):
        star, l, velscale = util.log_rebin(lrange, stars[:,i],
                                               velscale=velscale)
        starsnew[:,i] = star
    emissionnew = np.zeros((len(logLam), len(emission[0])))
    for i in range(len(emission[0])):
        em, logLam, velscale = util.log_rebin([wave[0], wave[-1]], emission[:,i],
                                               velscale=velscale)
        emissionnew[:,i] = em
    # print range(len(stars))
    # for i in range(len(stars[0])):
    #     plt.plot(logLam, starsnew[i], "-k")
    # plt.show()
    hdu1 = pf.PrimaryHDU(starsnew.T)
    hdu2 = pf.ImageHDU(emissionnew.T)
    hdu1.header["CRVAL1"] = logLam[0]
    hdu1.header["CD1_1"] = logLam[1] - logLam[0]
    hdu1.header["CRPIX1"] = 1.
    hdulist = pf.HDUList([hdu1, hdu2])
    hdulist.writeto(fits, clobber=True)

def broad2resolution(wave, specs, obsres, resout):
    """ Broad resolution of observations to a given value. """
    specs = np.atleast_2d(specs)
    dw = np.diff(wave)[0]
    sigma_diff = np.sqrt(resout ** 2 - obsres ** 2) / 2.3548 / dw
    broad = np.zeros_like(specs)
    print "Processing broadening"
    for i, spec in enumerate(specs):
        print "Spectra {0}/{1}".format(i + 1, len(specs))
        d = np.diag(spec)
        for j in range(len(wave)):
            d[j] = gaussian_filter1d(d[j], sigma_diff, mode="constant",
                                     cval=0.0)
        broad[i] = d.sum(axis=0)
    return broad

if __name__ == "__main__":
    library = "MILES"
    res = 4.7 # instrumental resolution
    ldir = os.path.join(home, library, "specs")
    os.chdir(ldir)
    ages, metals, alphas = ranges_miles10(sampling="normal")
    filenames = []
    for ssp in np.array(np.meshgrid(ages, metals, alphas)).T.reshape(-1,3):
        filenames.append(filenames_miles10(*ssp))
    wave, stars = load_stars(filenames)
    stars = broad2resolution(wave, stars.T, 2.51, res).T
    ##########################################################################
    # Load spectral lines
    em_hbeta = make_emission_template(4861.333, res, wave)
    em_OIII_2 = make_emission_template(4958.91, res, wave)
    em_OIII_1 = make_emission_template(5006.84, res, wave)
    em_NI = make_emission_template(5200.257, res, wave)
    em_halpha = make_emission_template(6564.61, res, wave)
    em_NII_1 = make_emission_template(6585.27, res, wave)
    em_NII_2 = make_emission_template(6549.86, res, wave)
    em_SII_1 = make_emission_template(6718.29, res, wave)
    em_SII_2 = make_emission_template(6732.67, res, wave)
    ##########################################################################
    # Prepare templates for stellar population analysis
    if True:
        wdir = os.path.join(home, library, "templates")
        if not os.path.exists(wdir):
            os.mkdir(wdir)
        os.chdir(wdir)
        linelist = [em_hbeta, em_OIII_2, em_OIII_1, em_halpha, em_NI,
                    em_NII_1, em_NII_2, em_SII_1, em_SII_2]
        lines = np.zeros((len(wave), len(linelist)))
        for i, line in enumerate(linelist):
            lines[:,i] = line
        w1, w2 = wave[0], wave[1]
        make_templates(wave, stars, lines, velscale, w1, w2,
                       "templates_w{0}_{1}_res{2}.fits".format(w1+100, w2,
                                                               res))






