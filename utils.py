import numpy as np
from scipy.interpolate import *
import astropy.units as u
import astropy.constants as c
from astropy.modeling import *
from astropy.io import fits

def viewX1DFile(filename):
    print(f"x1d file: '{filename}'")
    data=fits.open(filename)
    for i,hdu in enumerate(data):
        print(f"HDU {i}:")
        print(repr(hdu.header))
        print(hdu.header.keys())
    input()