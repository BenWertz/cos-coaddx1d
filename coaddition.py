import numpy as np
from astropy.io import fits
import datetime
import spectrum_align
import astropy.units as u
from scipy.interpolate import *

def save_spec_data(filepath, spec_data): 
    # save a fits file with header information
    today = str(datetime.datetime.now())
    coadd_header=spec_data["header"]

    filename=f'{filepath}/{coadd_header["TARGNAME"]}_{coadd_header["OPT_ELEM"].lower()}_coadd_x1d_bin{spec_data["binning"]}_{spec_data["weighing_method"]}'

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
    colErr = fits.Column(name='ERROR', format='F', array=spec_data["error"])

    cols = fits.ColDefs([
        colWav,
        colFlx,
        colErr
    ])
    table_hdu = fits.BinTableHDU.from_columns(cols)

    hdu_all = fits.HDUList([primary_hdu, table_hdu])
    hdu_all.writeto(filename+'.fits', overwrite=True)
    print('Saved ', filename+'.fits')

    return filename+'.fits'



def weigh_data_quality(dq):
    """
    Reads the X1D's data quality array and looks for specific flags indicating
    that a pixel should be thrown out or deweighted.

    Returns an array of weights indicating if a pixel should be kept (weight of 1),
    thrown out (weight of 0), or deweighted (weight of 0.5).
    """

    #      1 = 2^0  = Reed-Solomon error         --> throw out
    #      2 = 2^1  = Not used                   --> ignore
    #      4 = 2^2  = Detector shadow            --> de-weight
    #      8 = 2^3  = Poorly calibrated          --> de-weight
    #     16 = 2^4  = Very low response region   --> de-weight
    #     32 = 2^5  = Background feature         --> de-weight
    #     64 = 2^6  = Burst                      --> throw out
    #    128 = 2^7  = Pixel out of bounds        --> throw out
    #    256 = 2^8  = Fill data                  --> throw out
    #    512 = 2^9  = Pulse height out of bounds --> throw out
    #   1024 = 2^10 = Low response region        --> de-weight
    #   2048 = 2^11 = Bad time interval          --> throw out
    #   4096 = 2^12 = Low PHA feature            --> de-weight
    #   8192 = 2^13 = Gain-sag hole              --> throw out
    #  16384 = 2^14 = Not used                   --> ignore

    dq_weights=np.ones(len(dq))

    print(dq.dtype)

    dq_weights[
        ((dq & 2**0)>0) |
        ((dq & 2**6)>0) |
        ((dq & 2**7)>0) |
        ((dq & 2**8)>0) |
        ((dq & 2**9)>0) |
        ((dq & 2**11)>0) |
        ((dq & 2**13)>0)
    ]=0

    dq_weights[
        ((dq & 2**2)>0) |
        ((dq & 2**3)>0) |
        ((dq & 2**4)>0) |
        ((dq & 2**5)>0) |
        ((dq & 2**10)>0) |
        ((dq & 2**12)>0)
    ]*=0.5

    return dq_weights

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
    wl_delta=np.median(np.diff(ref_spec_data["wave"].value))
    wl_min=min([spec_table[f]["header"]["MINWAVE"] for f in filenames])
    wl_max=max([spec_table[f]["header"]["MAXWAVE"] for f in filenames])
    common_grid_wave=np.arange(
        wl_min,
        wl_max,
        wl_delta
    )*u.AA

    # ref_wave=ref_wave[wlim]
    # ref_wave=np.sort(ref_wave[~np.isnan(ref_wave)])
    # ref_flx=ref_spec_data["flux"][wlim]
    # ref_err=ref_spec_data["error"][wlim]
    # ref_cnt=ref_spec_data["net"][wlim]

    coadded_flux=np.zeros(len(common_grid_wave))
    weights=np.zeros(len(common_grid_wave))
    error_accum=np.zeros(len(common_grid_wave))

    for i,x1d_name in enumerate(filenames):
        spec_data=spec_table[x1d_name]
        # else:
        wave=spec_data["wave"]
        counts=spec_data["net"]
        flux=spec_data["flux"]
        error=spec_data["error"]
        dq=spec_data["dq"]

        dq_wgts=weigh_data_quality(dq)

        offset_wave=wave-offset_table["mean_dlambda"][i]

        interp=interp1d(
            offset_wave,
            flux,
            kind="nearest",
            fill_value=0,
            bounds_error=False
        )
        net_interp=interp1d(
            offset_wave,
            counts,
            kind="nearest",
            fill_value=0,
            bounds_error=False
        )
        err_interp=interp1d(
            offset_wave,
            error,
            kind="nearest",
            fill_value=0,
            bounds_error=False
        )
        dq_wgts_interp=interp1d(
            offset_wave,
            dq_wgts,
            kind="nearest",
            fill_value=0,
            bounds_error=False
        )

        # spec_data["wave_updated"]=ref_wave
        interp_flux=interp(common_grid_wave)
        interp_net=net_interp(common_grid_wave)
        interp_err=err_interp(common_grid_wave)
        interp_dq_wgt=dq_wgts_interp(common_grid_wave)

        # Completely throw out invalid pixels
        # Note: do I actually have to do this? It seems like setting the weight to zero should be enough
        interp_flux[interp_dq_wgt==0]=0
        interp_err[interp_dq_wgt==0]=0

        # For deweighted datapoints, the error is multiplied by sqrt(2)
        interp_err[interp_dq_wgt==0.5]*=(2**.5)

        if weighing=="exptime":
            weight=np.ones(len(interp_flux))*spec_data["exptime"]
        elif weighing=="modified_exptime":
            weight=(interp_net/interp_flux)*spec_data["exptime"]
            weight[interp_flux==0]=0.
        
        weight*=interp_dq_wgt

        mask=((interp_flux!=0) & ~np.isnan(interp_flux))
        coadded_flux[mask]+=interp_flux[mask]*weight[mask]

        error_accum[mask]+=(interp_err[mask]**2)*weight[mask]**2

        # error_accum[mask]+=(interp_err[mask]**2)*weight[mask]

        weights[mask]+=(mask.astype(np.float32)*weight)[mask]

    coadded_flux/=weights
    coadded_error=error_accum**.5/weights

    # S=sqrt(N*(s^2*w^2))/(N*w)

    # coadded_error=(error_accum/weights)**.5

    # s^2=SUM(s_i^2*(w_i/W)^2)
    # s^2=SUM(s_i^2*(w_i)^2)/W^2

    # s=sqrt(sum(s_i^2*w_i^2))/W

    # coadded_flux=coadded_error
    # Binning
    output_wave,output_flux,output_error=bin_spec(common_grid_wave,coadded_flux,coadded_error,bin)

    return {
        "wave":output_wave,
        "error":output_error,
        "flux":output_flux,
        "header":ref_spec_data["header"],
        "weighing_method":weighing,
        "binning":bin
    }

def bin_spec(x1, y1, error1, bins):
    if bins==1:
        return x1,y1,error1

    b1 = np.mgrid[0:len(x1):bins]
    dg1 = np.digitize(np.mgrid[0:len(x1.value):1], b1)

    dg_x1 = np.array([np.mean(x1.value[dg1==j]) for j in np.arange(len(b1)+1)[1:]])*x1.unit
    dg_y1 = np.array([np.mean(y1[dg1==j]) for j in np.arange(len(b1)+1)[1:]])
    dg_error1 = np.array([np.sum(error1[dg1==j]**2)**.5/bins for j in np.arange(len(b1)+1)[1:]])
    # dg_error1 = np.array([np.mean(error1[dg1==j]**2)**.5 for j in np.arange(len(b1)+1)[1:]])

    # sqrt(a^2+a^2+a^2)/sqrt(3)
    # sqrt(a^2+a^2+a^3)/sqrt(3)

    # sqrt((s1**2+s2**2+s3**2)/3)

    return dg_x1, dg_y1, dg_error1
