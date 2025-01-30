import numpy as np
from scipy.interpolate import *
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import astropy.stats as stats
import sys
import os
import glob
from diagnostic_plots import plot_clip_lim

def viewX1D(filename):
    data=fits.open(filename)

    spec_hdu=data[1]
    print(spec_hdu.header["EXPTIME"])

    if len(spec_hdu.data["WAVELENGTH"])==0:return

    wave=np.concatenate([
        spec_hdu.data["WAVELENGTH"][1,:],spec_hdu.data["WAVELENGTH"][0,:]
    ])

    flux=np.concatenate([
        spec_hdu.data["FLUX"][1,:],spec_hdu.data["FLUX"][0,:]
    ])

    error=np.concatenate([
        spec_hdu.data["ERROR"][1,:],spec_hdu.data["ERROR"][0,:]
    ])

    dq=np.concatenate([
        spec_hdu.data["DQ"][1,:],spec_hdu.data["DQ"][0,:]
    ])

    fig=plt.figure(figsize=(8,6))
    ax=fig.subplots(2,1,sharex=True)
    ax[0].step(wave,flux)
    ax[0].step(wave,error)

    # Data quality array
    # from https://hst-docs.stsci.edu/stisdhb/chapter-2-stis-data-structure/2-5-error-and-data-quality-array#id-2.5ErrorandDataQualityArray-2.5.2DataQualityFlagging

    #bit | Condition
    #  0 | Error in the Reed-Solomon decoding (an algorithm for error correction in digital communications).
    #  1 | Lost data replaced by fill values
    #  2 | Bad detector pixel (e.g., bad column or row, mixed science and bias for overscan, or beyond aperture).
    #  3 | Data masked by occulting bar.
    #  4 | Pixel having dark rate > 5 σ times the median dark level.
    #  5 | Large blemish, depth > 40% of the normalized p-flat (repeller wire).
    #  6 | Vignetted pixel
    #  7 | Pixel in the overscan region.
    #  8 | Saturated pixel, count rate at 90% of max possible—local non-linearity turns over and is multi-valued; pixels within 10% of turnover and all pixels within 4 pixels of that pixel are flagged.
    #  9 | Bad pixel in reference file.
    # 10 | Small blemish, depth between 40% and 70% of the normalized flat. Applies to MAMA and CCD p-flats.
    # 11 | >30% of background pixels rejected by sigma-clip, or flagged, during 1-D spectral extraction.
    # 12 | Extracted flux affected by bad input data.
    # 13 | Data rejected in input pixel during image combination for cosmic ray rejection.
    # 14 | Extracted flux not CTI corrected because gross counts are ≤ 0.

    dq_labels=[
        "Decoding error",#0
        "Lost data",#1
        "Bad detector pixel",#2
        "Pixel under occulting bar",#3
        "Pixel too dark",#4
        "Large blemish",#5
        "Vignette",#6
        "Overscan",#7
        "Saturated",#8
        "Bad ref. pixel",#9
        "Small blemish",#10
        "bcg. sigma-clipped", #11
        "Bad input (???)",#12
        "Cosmic ray",#13
        "counts<=0",#14
        "(none)",#15
    ]

    # dq=(dq*0)+1*np.mod(np.arange(len(dq)),2)

    ax[1].set_yticks(list(range(16)),labels=dq_labels)

    for i in range(16):
        flag_plot=ax[1].plot(wave,i+0.5*((dq>>i) & 1))
    #     ax[1].text(
    #         wave.min()-10,i,
    #         dq_labels[i],
    #         fontsize=8,
    #         horizontalalignment="right",
    #         verticalalignment="center",
    #         color=flag_plot[0].get_color()
    #     )

    low,high=plot_clip_lim(flux)
    ax[0].set_ylim(-.2e-14,0.2e-13)

    ax[0].set_xlim(wave.min(),wave.max()+20)

    fig.subplots_adjust(left=0.3)

    fig.suptitle(filename)

    return dq

arg_TgtFolder=f"targets/{sys.argv[1]}/raw_g130m/x1d"
filenames = glob.glob(arg_TgtFolder+"/*x1d.fits")


dqs=[]
for filename in filenames:
    dqs.append(viewX1D(filename))

plt.show()
