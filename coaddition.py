import numpy as np
from astropy.io import fits
import datetime
import spectrum_align
from scipy.interpolate import *

def save_spec_data(filename, spec_data): 
    # save a fits file with header information
    today = str(datetime.datetime.now())
    coadd_header=spec_data["header"]

    hdr = fits.Header()
    hdr['TELESCOP'] = coadd_header['TELESCOP']
    hdr['INSTRUME'] = coadd_header['INSTRUME']
    hdr['FILTER'] =   coadd_header['OPT_ELEM']
    hdr['TARGNAME'] = coadd_header['TARGNAME']
    hdr['RA_TARG']  = coadd_header['RA_TARG']
    hdr['DEC_TARG'] = coadd_header['DEC_TARG']
    hdr['EQUINOX'] = 2000.0
    hdr['WAVEUNIT'] = ('vacuum')

    # TODO: Add actual history here

    primary_hdu = fits.PrimaryHDU(header=hdr)

    colWav = fits.Column(name='WAVE', format='F', array=np.around(spec_data["wave"], decimals=4))
    colFlx = fits.Column(name='FLUX', format='F', array=spec_data["flux"])
    # colErr = fits.Column(name='ERR', format='F', array=spec_data["error"])

    cols = fits.ColDefs([
        colWav,
        colFlx,
        # colErr
    ])
    table_hdu = fits.BinTableHDU.from_columns(cols)

    hdu_all = fits.HDUList([primary_hdu, table_hdu])
    hdu_all.writeto(filename+'.fits', overwrite=True)
    print('Saved ', filename+'.fits')

    return filename+'.fits'

def coadd_spec(spec_table,filenames,offset_table,
        weighing="exptime",bin=1):
    """
    Coadds a table of spectra based on their calculated wavelength offsets.
    ----------
    Parameters:
        spec_table:
            Table of spectra objects.
        filenames:
            List of filenames used to index the table of spectra.
        offset_table:
            Astropy table of wavelength offsets for each spectrum and each line.
    ----------
    Returns:
        Returns a dictionary containing the wavelength and flux arrays, and the relevant header information
    """
    ref_idx=0

    # ref grid
    ref_spec_data=spec_table[filenames[ref_idx]]
    ref_wave=ref_spec_data["wave"]
    wlim=((ref_wave.value>min([spec_table[f]["header"]["MINWAVE"] for f in filenames]))&
          (ref_wave.value<max([spec_table[f]["header"]["MAXWAVE"] for f in filenames])))

    ref_wave=ref_wave[wlim]
    ref_wave=np.sort(ref_wave[~np.isnan(ref_wave)])
    ref_flx=ref_spec_data["flux"][wlim]
    ref_err=ref_spec_data["error"][wlim]
    ref_cnt=ref_spec_data["net"][wlim]

    coadded_flux=np.zeros(len(ref_wave))
    weights=np.zeros(len(ref_wave))
    error_accum=np.zeros(len(ref_wave))

    for i,x1d_name in enumerate(filenames):
        spec_data=spec_table[x1d_name]
        if i==ref_idx:
            # spec_data["wave_updated"]=spec_data["wave"]
            mask=((ref_flx!=0)&~np.isnan(ref_err))

            if weighing=="exptime":
                weight=np.ones(len(ref_err))*spec_data["exptime"]
            elif weighing=="modified_exptime":
                weight=(ref_cnt/ref_flx)*spec_data["exptime"]
                weight[ref_flx==0]=0.

            coadded_flux[mask]+=ref_flx[mask]*weight[mask]
            error_accum[mask]+=((ref_err[mask])**2)*weight[mask]
            weights[mask]+=(mask.astype(np.float32)*weight)[mask]
        else:
            interp=interp1d(
                spec_data["wave"]-offset_table["mean_dlambda"][i],
                spec_data["flux"],
                kind="nearest",
                fill_value=0,
                bounds_error=False
            )
            net_interp=interp1d(
                spec_data["wave"]-offset_table["mean_dlambda"][i],
                spec_data["net"],
                kind="nearest",
                fill_value=0,
                bounds_error=False
            )
            err_interp=interp1d(
                spec_data["wave"]-offset_table["mean_dlambda"][i],
                spec_data["error"],
                kind="nearest",
                fill_value=0,
                bounds_error=False
            )

            # spec_data["wave_updated"]=ref_wave
            interp_flux=interp(ref_wave)
            interp_net=net_interp(ref_wave)
            interp_err=err_interp(ref_wave)

            if weighing=="exptime":
                weight=np.ones(len(interp_flux))*spec_data["exptime"]
            elif weighing=="modified_exptime":
                weight=(interp_net/interp_flux)*spec_data["exptime"]
                weight[interp_flux==0]=0.

            mask=((interp_flux!=0) & ~np.isnan(interp_flux))
            coadded_flux[mask]+=interp_flux[mask]*weight[mask]
            error_accum[mask]+=(interp_err[mask]**2)*weight[mask]

            weights[mask]+=(mask.astype(np.float32)*weight)[mask]

            # plt.step(spec_data["wave"],spec_data["flux"],lw=0.5,label=filename)

    coadded_flux/=weights
    coadded_error=(error_accum/weights)**.5

    # coadded_flux=coadded_error
    # Binning
    output_wave,output_flux,output_error=bin_spec(ref_wave,coadded_flux,coadded_error,bin)

    return {
        "wave":output_wave,
        "error":output_error,
        "flux":output_flux,
        "header":ref_spec_data["header"],
        "weighing_method":weighing
    }

def bin_spec(x1, y1, error1, bins):
    if bins==1:
        return x1,y1,error1

    b1 = np.mgrid[0:len(x1):bins]
    dg1 = np.digitize(np.mgrid[0:len(x1.value):1], b1)

    dg_x1 = np.array([np.mean(x1.value[dg1==j]) for j in np.arange(len(b1)+1)[1:]])*x1.unit
    dg_y1 = np.array([np.mean(y1[dg1==j]) for j in np.arange(len(b1)+1)[1:]])
    dg_error1 = np.array([np.mean(error1[dg1==j]**2)**.5 for j in np.arange(len(b1)+1)[1:]])

    return dg_x1, dg_y1, dg_error1
