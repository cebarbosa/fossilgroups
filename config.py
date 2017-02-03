# -*- coding: utf-8 -*-
"""

Created on 01/02/2017

@Author: Carlos Eduardo Barbosa

"""

home = "/home/kadu/Dropbox/fossils"
data_dir = home + "/data"

velscale = 50. # km / s

# Constants
c = 299792.458 # Speed of light

# Emission lines used in the projects
lines = (("Hbeta_4861", 4861.333), ("OIII_4959", 4958.91),
         ("OIII_5007", 5006.84), ("Halpha_6565", 6564.61),
         ("NI_5200", 5200.257), ("NII_6585", 6585.27),
         ("NII_6550", 6549.86), ("SII_6718", 6718.29),
         ("SII_6733", 6732.67))

linelabels = [r"H$\beta$", r"[OIII]$\lambda 4959$",
              r"[OIII]$\lambda 5007$", r"H$\alpha$",
              r"[NI]$\lambda 5200$", r"[NII]$\lambda 6585$",
              r"[NII]$\lambda 6550$", r"[SII]$\lambda 6718$",
              r"[SII]$\lambda 6733$"]