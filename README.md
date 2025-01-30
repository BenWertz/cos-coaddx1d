# cos-coaddx1d
Python code for coadding X1D spectra from the HST Cosmic Origins Spectrograph (HST/COS)

This program was developed as an alternative to existing coadding codes for HST/COS data, particularly calibrated 1-dimensional spectra in the X1D format. The code aligns component spectra performing Gaussian curve fits on strong Milky Way absorption lines, regridding and weighing each result based on exposure time and flagged data quality issues (the choices of weights for each data quality flag were based on those from the IDL program `coadd_x1d.pro` from (Danforth 2010)).

(Note: This code was written for use in a particular project and may require some changes to be used in other projects. In particular, the inputs and outputs may assume a particular filesystem layout. The code was written to handle data in the G130M and G160M gratings exclusively; changing this would also require changes to the grating options and lists of target and diagnostic lines.)

The program runs in the command line with the following options:

`python3 coadd_cos.py <path to root folder with X1D files> <grating> <coadd_mode>`
- `<grating>` can be `g130m` or `g160m`
- `coadd_mode` has two options which modify how the component spectra are weighted:
    - `exptime` sets the initial weight based on the exposure time of each epoch.
    - `modified_exptime` sets initial weights based on an effective exposure time adjusted based on the ratio of flux to photon counts.

The output file is stored in a folder off of the filepath of the input, storing diagnostic plots of the results as well as the final coadded spectrum. This spectrum is in the FITS format and contains the same basic wavelength/flux/error values as in the X1D format.