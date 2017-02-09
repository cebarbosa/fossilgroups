# -*- coding: utf-8 -*-
"""

Created on 01/02/2017

@Author: Carlos Eduardo Barbosa

Run pPXF to derive kinematic parameters

"""
import os
import pickle

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.stats import sigma_clip
from pandas import rolling_std

from ppxf.ppxf import ppxf, reddening_curve
import ppxf.ppxf_util as util
from config import *

def wavelength_array(spec, axis=1, extension=0):
    """ Produces array for wavelenght of a given array. """
    w0 = pf.getval(spec, "CRVAL{0}".format(axis), extension)
    deltaw = pf.getval(spec, "CD{0}_{0}".format(axis), extension)
    pix0 = pf.getval(spec, "CRPIX{0}".format(axis), extension)
    npix = pf.getval(spec, "NAXIS{0}".format(axis), extension)
    return w0 + deltaw * (np.arange(npix) + 1 - pix0)

class pPXF():
    """ Class to read pPXF pkl files """
    def __init__(self, pp, velscale):
        self.__dict__ = pp.__dict__.copy()
        self.calc_arrays()
        self.calc_sn()
        return

    def calc_arrays(self):
        """ Calculate the different useful arrays."""
        # Slice matrix into components
        self.m_poly = self.matrix[:,:self.degree + 1]
        self.matrix = self.matrix[:,self.degree + 1:]
        self.m_ssps = self.matrix[:,:self.ntemplates]
        self.matrix = self.matrix[:,self.ntemplates:]
        self.m_gas = self.matrix[:,:self.ngas]
        self.matrix = self.matrix[:,self.ngas:]
        self.m_sky = self.matrix
        # Slice weights
        if hasattr(self, "polyweights"):
            self.w_poly = self.polyweights
            self.poly = self.m_poly.dot(self.w_poly)
        else:
            self.poly = np.zeros_like(self.galaxy)
        if hasattr(self, "mpolyweights"):
            x = np.linspace(-1, 1, len(self.galaxy))
            self.mpoly = np.polynomial.legendre.legval(x, np.append(1,
                                                       self.mpolyweights))
        else:
            self.mpoly = np.ones_like(self.galaxy)
        if self.reddening is not None:
            self.extinction = reddening_curve(self.lam, self.reddening)
        else:
            self.extinction = np.ones_like(self.galaxy)
        self.w_ssps = self.weights[:self.ntemplates]
        self.weights = self.weights[self.ntemplates:]
        self.w_gas = self.weights[:self.ngas]
        self.weights = self.weights[self.ngas:]
        self.w_sky = self.weights
        # Calculating components
        self.ssps = self.m_ssps.dot(self.w_ssps)
        self.gas = self.m_gas.dot(self.w_gas)
        self.bestsky = self.m_sky.dot(self.w_sky)
        return

    def calc_sn(self):
        """ Calculates the S/N ratio of a spectra. """
        self.signal = np.nanmedian(self.galaxy)
        self.noise = np.nanstd(sigma_clip(self.galaxy - self.bestfit, sigma=5))
        self.sn = self.signal / self.noise
        return

    def mc_errors(self, nsim=200):
        """ Calculate the errors using MC simulations"""
        errs = np.zeros((nsim, len(self.error)))
        for i in range(nsim):
            y = self.bestfit + np.random.normal(0, self.noise,
                                                len(self.galaxy))

            noise = np.ones_like(self.galaxy) * self.noise
            sim = ppxf(self.bestfit_unbroad, y, noise, velscale,
                       [0, self.sol[1]],
                       goodpixels=self.goodpixels, plot=False, moments=4,
                       degree=-1, mdegree=-1,
                       vsyst=self.vsyst, lam=self.lam, quiet=True, bias=0.)
            errs[i] = sim.sol
        median = np.ma.median(errs, axis=0)
        error = 1.4826 * np.ma.median(np.ma.abs(errs - median), axis=0)
        # Here I am using always the maximum error between the simulated
        # and the values given by pPXF.
        self.error = np.maximum(error, self.error)
        return

    def plot(self, output=None, fignumber=1, textsize = 12):
        """ Plot pPXF run in a output file"""
        if self.ncomp > 1:
            sol = self.sol[0]
            error = self.error[0]
            sol2 = self.sol[1]
            error2 = self.error[1]
        else:
            sol = self.sol
            error = self.error
        plt.figure(fignumber, figsize=(5,4.2))
        plt.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5,1])
        ax = plt.subplot(gs[0])
        ax.minorticks_on()
        ax.plot(self.w[self.goodpixels], self.galaxy[self.goodpixels] -
                self.bestsky[self.goodpixels], "-k", lw=2.,
                label="Data (S/N={0})".format(np.around(self.sn,1)))
        ax.plot(self.w[self.goodpixels], self.bestfit[self.goodpixels] -
                self.bestsky[self.goodpixels], "-", lw=2., c="r",
                label="SSPs: V={0:.0f} km/s, $\sigma$={1:.0f} km/s".format(
                    sol[0], sol[1]))
        ax.xaxis.set_ticklabels([])
        if self.has_emission:
            ax.plot(self.w[self.goodpixels], self.gas[self.goodpixels], "-b",
                    lw=1.,
                    label="Emission: V={0:.0f} km/s, "
                          "$\sigma$={1:.0f} km/s".format(sol2[0],sol2[1]))
        # if self.sky != None:
        #     ax.plot(self.w[self.goodpixels], self.bestsky[self.goodpixels], \
        #             "-", lw=1, c="g", label="Sky")
        ax.set_xlim(self.w[0], self.w[-1])
        ax.set_ylim(-5 * self.noise, 1.2 * self.galaxy[self.goodpixels[500:-500]].max())
        leg = plt.legend(loc=2, prop={"size":8}, title=self.title,
                         frameon=False)
        # leg.get_frame().set_linewidth(0.0)
        plt.axhline(y=0, ls="--", c="k")
        plt.ylabel(r"Flux (arbitarry units)",
                   size=12)
        ax1 = plt.subplot(gs[1])
        ax1.minorticks_on()
        ax1.set_xlim(self.w[0], self.w[-1])
        ax1.plot(self.w[self.goodpixels], (self.galaxy[self.goodpixels] - \
                 self.bestfit[self.goodpixels]), "-k",
                 label="$\chi^2=${0:.2f}".format(self.chi2))
        leg2 = plt.legend(loc=2, prop={"size":8})
        ax1.axhline(y=0, ls="--", c="k")
        ax1.set_ylim(-3 * self.noise.mean(), 3 * self.noise)
        ax1.set_xlabel(r"$\lambda$ ($\AA$)", size=12)
        ax1.set_ylabel(r"$\Delta$Flux", size=12)
        gs.update(hspace=0.075, left=0.16, bottom=0.14, top=0.98, right=0.96)
        if output is not None:
            plt.savefig(output)
        return

def ppsave(pp, outroot="logs/out"):
    """ Produces output files for a ppxf object. """
    arrays = ["matrix", "w", "bestfit", "goodpixels", "galaxy", "noise"]
    delattr(pp, "star_rfft")
    delattr(pp, "star")
    hdus = []
    for i,att in enumerate(arrays):
        if i == 0:
            hdus.append(pf.PrimaryHDU(getattr(pp, att)))
        else:
            hdus.append(pf.ImageHDU(getattr(pp, att)))
        delattr(pp, att)
    hdulist = pf.HDUList(hdus)
    hdulist.writeto(outroot + ".fits", clobber=True)
    with open(outroot + ".pkl", "w") as f:
        pickle.dump(pp, f)

def ppload(inroot="logs/out"):
    """ Read ppxf arrays. """
    with open(inroot + ".pkl") as f:
        pp = pickle.load(f)
    arrays = ["matrix", "w", "bestfit", "goodpixels", "galaxy", "noise"]
    for i, item in enumerate(arrays):
        setattr(pp, item, pf.getdata(inroot + ".fits", i))
    return pp

def snr(flux, axis=0):
    """ Calculates the S/N ratio of a spectra.

    Translated from the IDL routine der_snr.pro """
    signal = np.nanmedian(flux, axis=axis)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.*flux - \
           np.roll(flux, 2, axis=axis) - np.roll(flux, -2, axis=axis)), \
           axis=axis)
    return signal, noise, signal / noise


def run_ppxf(group, specs, redo=False, ncomp=2, logdir="ppxf", window = 50,
             w1=None, w2=None, **kwargs):
    """ New function to run pPXF. """
    global velscale
    tempfile = os.path.join(home, "MILES/templates/"
                                  "templates_w3540.5_7409.6_res4.7.fits")
    stars = pf.getdata(tempfile, 0)
    emission = pf.getdata(tempfile, 1)
    logLam_temp = wavelength_array(tempfile, axis=1, extension=0)
    ngas = len(emission)
    nstars = len(stars)
    templates = np.column_stack((stars.T, emission.T))
    ##########################################################################
    # Set components
    if ncomp == 1:
        components = np.zeros(nstars + ngas)
        kwargs["component"] = components
    elif ncomp == 2:
        components = np.hstack((np.zeros(nstars), np.ones(ngas))).astype(int)
        kwargs["component"] = components
    ##########################################################################
    for spec in specs:
        os.chdir(os.path.join(data_dir, group))
        outdir = os.path.join(os.getcwd(), logdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        data = pf.getdata(spec)
        w = wavelength_array(spec)
        if w1 == None:
            w1 = w[0]
        if w2 == None:
            w2 = w[-1]
        idx = np.where((w >= w1) & (w <= w2))[0]
        data = data[idx]
        w = w[idx]
        #######################################################################
        # Preparing output
        output = os.path.join(outdir, spec.replace(".fits", ".pkl"))
        outroot = output.replace(".pkl", "")
        if os.path.exists(output) and not redo:
            continue
        print group, spec
        signal, noise, sn = snr(data)
        galaxy, logLam, vtemp = util.log_rebin([w[0], w[-1]],
                                                  data, velscale=velscale)
        lam = np.exp(logLam)
        ###################################################################
        # Trimming spectra
        idx = np.where(np.exp(logLam) > np.exp(logLam_temp[0]))[0]
        didx = int(1500. / velscale) # Space in the beginning of the spec
        idx = idx[didx:]
        galaxy = galaxy[idx]
        lam = lam[idx]
        logLam = logLam[idx]
        #######################################################################
        # Masking bad pixels
        goodpixels = np.arange(len(lam), dtype=float)
        gaps= set_badpixels(group)
        for gap in gaps:
            idx = np.where((lam > gap[0] - window/2.) &
                           (lam < gap[1] + window / 2.))[0]
            goodpixels[idx] = np.nan
        goodpixels = goodpixels[~np.isnan(goodpixels)].astype(int)
        kwargs["goodpixels"] = goodpixels
        #######################################################################
        kwargs["lam"] = lam
        dv = (logLam_temp[0] - logLam[0]) * c
        kwargs["vsyst"] = dv
        noise = np.ones_like(galaxy) * noise
        # First fitting to obtain realistic noise
        pp0 = ppxf(templates, galaxy, noise, velscale, **kwargs)
        pp0.has_emission = True
        pp0.dv = dv
        pp0.w = np.exp(logLam)
        pp0.velscale = velscale
        pp0.ngas = ngas
        pp0.ntemplates = nstars
        pp0.templates = 0
        pp0.name = spec
        pp0.title = ""
        pp0 = pPXF(pp0, velscale)
        pp0.calc_sn()
        res = (pp0.galaxy - pp0.bestfit)
        noise = rolling_std(res, window, center=True)
        noise[:window/2] =  noise[window+1]
        noise[-window/2+1:] = noise[-window/2]
        # Second fitting using results from first interaction
        pp = ppxf(templates, galaxy, noise, velscale, **kwargs)
        title = "Group {} , Spectrum {}".format(group.upper(), spec.replace(
            ".fits", "").replace("spec", ""))
        # Adding other things to the pp object
        pp.has_emission = True
        pp.dv = dv
        pp.w = np.exp(logLam)
        pp.velscale = velscale
        pp.ngas = ngas
        pp.ntemplates = nstars
        pp.templates = 0
        pp.id = id
        pp.name = spec
        pp.title = title
        ppsave(pp, outroot=outroot)
        ppf = ppload(outroot)
        ppf = pPXF(ppf, velscale)
        ppf.plot("{1}/{0}.png".format(pp.name.replace(".fits", ""), outdir))
    return


def make_table(group, logdir="ppxf"):
    """ Make table with results. """
    head4 = "{0:<25}{1:<14}{2:<14}{3:<14}{4:<14}{5:<14}{6:<14}{7:<14}" \
             "{8:<14}{9:<14}{10:<14}{11:<14}{12:<14}\n".format("# FILE",
             "V", "dV", "S", "dS", "h3", "dh3", "h4", "dh4", "chi/DOF",
             "S/N", "ADEGREE", "MDEGREE")
    head2 = "{0:<25}{1:<14}{2:<14}{3:<14}{4:<14}{9:<14}" \
             "{10:<14}{11:<14}{12:<14}\n".format("# FILE",
             "V", "dV", "S", "dS", "h3", "dh3", "h4", "dh4", "chi/DOF",
             "S/N", "ADEGREE", "MDEGREE")
    print "Producing summary for group {0}".format(group)
    os.chdir(os.path.join(data_dir, group, logdir))
    output = "{}_{}.dat".format(group, logdir)
    results = []
    for i in range(1,20):
        fname = "spec{}".format(i)
        try:
            pp = ppload(fname)
            pp = pPXF(pp, velscale)
        except:
            continue
        sol = pp.sol if pp.ncomp == 1 else pp.sol[0]
        sol[0] = sol[0]
        error = pp.error if pp.ncomp == 1 else pp.error[0]
        head = head2 if len(sol) == 2 else head4
        line = np.zeros((sol.size + error.size,))
        line[0::2] = sol
        line[1::2] = error
        line = np.append(line, [pp.chi2, pp.sn])
        line = ["{0:12.3f}".format(x) for x in line]
        line = ["{0:18s}".format(fname)] + line + \
               ["{0:12}".format(pp.degree), "{0:12}".format(pp.mdegree)]
        results.append(" ".join(line) + "\n")
    with open(output, "w") as f:
        f.write(head)
        f.write("".join(results))

def plot_all(group, specs, logdir="ppxf"):
    """ Make plot of all fits. """
    os.chdir(os.path.join(data_dir, group, logdir))
    for i, spec in enumerate(specs):
        print "Print spectrum ", spec
        pp = ppload("{0}".format(spec.replace(".fits", "")))
        pp = pPXF(pp, velscale)
        pp.plot("{0}".format(spec.replace(".fits", ".png")))
        plt.savefig(spec.replace(".fits", ".png"))

def set_badpixels(group):
    """ Set the best ranges for the fitting. """
    if group == "RXJ0216":
        return [[4349, 4353], [4523, 4531], [5569, 5592], [5627, 5642],
                [6437, 6451], [6767, 6900]]
    elif group == "ESO552":
        return [[5563, 5592], [5630, 5642], [5742, 5754], [5926, 5941],
                [6289, 6313], [6764, 6900]]
    else:
        return None

def test_noise(group, specs, logdir):
    """ Using previous fittings to improve weighting of fitting. """
    os.chdir(os.path.join(data_dir, group, logdir))
    for i in range(1,20):
        fname = "spec{}".format(i)
        try:
            pp = ppload(fname)
            pp = pPXF(pp, velscale)
        except:
            continue
        res = (pp.galaxy - pp.bestfit) / pp.mpoly
        from pandas import rolling_std
        window = 50
        noise = rolling_std(res, window, center=True)
        noise[:window/2] =  noise[window+1]
        noise[-window/2+1:] = noise[-window/2]
        plt.plot(pp.lam, res, "-k")
        plt.plot(pp.lam, noise, "-r")
        plt.plot(pp.lam, -noise, "-r")
        plt.show(block=1)


def plot_zoom(group, specs, logdir):
    """ Modified version of the plot to include zooms in selected areas of
    the spectrum. """
    # Set bands to be plotted
    bname = os.path.join(home, "tables/bands.txt")
    features = np.loadtxt(bname, usecols=(0,), dtype="S")
    bands = np.loadtxt(bname, usecols=(2,3,4,5,6,7))
    os.chdir(os.path.join(data_dir, group, logdir))
    fig = plt.figure(1)
    for i, spec in enumerate(specs):
        pp = ppload("{0}".format(spec.replace(".fits", "")))
        pp = pPXF(pp, velscale)
        fname = os.path.join(data_dir, group, spec)
        z = pp.sol[0][0] / c
        print pp.sol[0]
        data = pf.getdata(fname)
        w = wavelength_array(fname)
        lam = pp.lam[pp.goodpixels]
        apoly = pp.poly[pp.goodpixels]
        mpoly = pp.mpoly[pp.goodpixels]
        galaxy = pp.galaxy[pp.goodpixels]
        bestfit = pp.bestfit[pp.goodpixels]
        # plt.plot(w, data, "-ok")
        # plt.plot(lam, bestfit, "-r")
        # plt.plot(lam, bestfit / mpoly, "-g")
        # plt.plot(lam, 1000 * mpoly, "-b")
        # plt.ylim(-300, 3850)
        for j in range(25):
            ax = plt.subplot(5,5,j+1)
            w1 = bands[j,0] * (1 + z)
            w2 = bands[j,-1] * (1 + z)
            idx1 = np.where((w >= w1) & (w <= w2))[0]
            idx2 = np.where((lam >= w1) & (lam <= w2))[0]
            plt.plot(w[idx1], data[idx1], "o-k", label=features[j])
            plt.plot(lam[idx2], bestfit[idx2], "-r")
            plt.legend()
        plt.show(block=1)









if __name__ == "__main__":
    groups = ["RXJ0216", "ESO552"]
    vrefs = [19000., 9000.]
    w1, w2 = 4000, 5400
    for i, (vref, group) in enumerate(zip(vrefs, groups)):
        os.chdir(os.path.join(data_dir, group))
        start = np.array([[vref, 50., 0., 0.], [vref, 30., 0., 0.]])
        bounds = np.array([[[vref -1000., vref + 1000.], [15., 800.],
                            [-0.5, 0.5], [-0.5, 0.5]],
                           [[vref - 1000., vref + 1000.], [15., 200.],
                            [-0.5, 0.5], [-0.5, 0.5]]])
        kwargs = {"start": start,
                  "plot": False, "moments": [4, 2], "degree": -1,
                  "reddening": None, "clean": False,
                  "bounds": bounds, "bias" : 0.7, "mdegree": 16}
        logdir = "ppxf_mom{0}_bias{1}".format(kwargs["moments"][0],
                                              kwargs["bias"])
        specs = [x for x in os.listdir(".") if x.endswith(".fits")]
        # run_ppxf(group, specs, redo=True, logdir=logdir, w1=w1, w2=w2,
        #          **kwargs)
        # make_table(group, logdir=logdir)
        # plot_all(group, specs, logdir=logdir)
        # test_noise(group, specs, logdir)
        plot_zoom(group, specs, logdir)

