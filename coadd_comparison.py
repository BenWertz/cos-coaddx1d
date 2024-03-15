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

data=fits.open(sys.argv[1])
ref_data=fits.open(r"C:\Users\benwertz\Dropbox\LocalSpiralsCGM\cos_analysis\targets\3C-66A\coadd_x1d\g130m\yz_data\3C-66A_G130M_coadd_x1d_bin3_m1.fits")
#PG1011-040_G130M+G160M_coadd_x1d_bin3_m1.fits


spec_hdu=data[1]

ref_spec_hdu=ref_data[1]

wave=spec_hdu.data["WAVE"]
flux=spec_hdu.data["FLUX"]
error=spec_hdu.data["ERROR"]

ref_wave=ref_spec_hdu.data["WAVE"][0,:]
ref_flux=ref_spec_hdu.data["FLUX"][0,:]
ref_error=ref_spec_hdu.data["ERROR"][0,:]

plt.figure(figsize=(12,6))

plt.step(
    wave,flux,
    c="k",
    lw=1,
    label="Program output"
)
plt.step(
    ref_wave,ref_flux,
    c="orange",
    lw=1,
    label="Reference file"
)

plt.step(
    wave,error,
    c="k",
    lw=0.5,
    alpha=0.5,
    label="Error"
)
plt.step(
    ref_wave,ref_error,
    c="red",
    lw=0.5,
    alpha=0.5,
    label="Reference Error"
)

plt.legend()

plt.ylim(-1e-14,1e-13)

wave_interp=interp1d(
    ref_wave,
    ref_flux,
    kind="nearest",
    fill_value=0,
    bounds_error=False
)

plt.figure()

grid_refflux=wave_interp(wave)
plt.step(
    wave,
    (flux-grid_refflux)/grid_refflux
)

minwave=wave.min()
maxwave=wave.max()

num_intervals=10

interval_width=10

error_array=np.zeros(num_intervals)

for i in range(num_intervals):
    x0=wave.min()+i*(wave.max()-wave.min())/num_intervals
    x1=x0+interval_width
    
    interval_mask=((wave>=x0) & (wave<x1))
    
    iv_flux=flux[interval_mask]
    iv_refflux=grid_refflux[interval_mask]

    
    error_array[i]=np.mean(
        stats.sigma_clip(
            (iv_flux-iv_refflux)/iv_refflux,
            sigma=3
        )
    )

print(f"% error: {100*np.mean(np.abs(error_array)):.2f}%")
# plt.ylim(0,1.2)

plt.show()