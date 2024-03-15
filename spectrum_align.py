import numpy as np
from scipy.interpolate import *
from astropy.stats import sigma_clip
import astropy.units as u
import astropy.constants as c
from astropy.modeling import *
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits

def genEmptySpecX1D():
    return {
        "wave":np.array([np.nan]),
        "flux":np.array([np.nan]),
        "error":np.array([np.nan]),
        "dq":np.array([np.nan]),
        "exptime":0,
        "lines":[],

        "flux_updated":np.array([np.nan]),
        "header":None,
        "grating":""
    }

def genEmptyLineData():
    return {
        "lambda":np.nan,
        "name":"",

        "line_wave":np.array([np.nan]),
        "line_vel":np.array([np.nan]),

        "line_flux":np.array([np.nan]),
        "line_error":np.array([np.nan]),

        "continuum_params":[],
        "flux_norm":np.array([np.nan]),
        "error_norm":np.array([np.nan]),

        "line_fit_params":[],
        "lambda_center":np.nan*u.AA,
        "dispersion":np.nan*u.AA,
        "chisq":-1,

        "has_cont":False,
        "bad_line":True,
        "valid":False
    }

def read_x1d_file(filename,wave_label="WAVELENGTH",flux_label="FLUX",error_label="ERROR",net_label="NET"):
    """
    Returns the wavelength, flux, error, and data HDU for a FITS file as a dictionary.
    
    Parameters:
    ----------
    filename:
        The filename to be read
    
    """
    data=fits.open(filename)
    spec_hdu=data[1]

    # print(("\n".join(sorted(data[0].header.keys()))).lower())
    # input()

    if spec_hdu.header["EXPTIME"]==0:
        print(f"Warning reading {filename}: EXPTIME=0, discarding")
        return {"exptime":0},spec_hdu

    # Cut off any parts of the low end of the spectrum that overlap the high end
    low_spectrum_mask=(spec_hdu.data[wave_label][1,:] < np.min(spec_hdu.data[wave_label][0,:]))

    net=np.concatenate([
        spec_hdu.data[net_label][1,:],#[low_spectrum_mask],
        spec_hdu.data[net_label][0,:]
    ])

    wave=np.concatenate([
        spec_hdu.data[wave_label][1,:],#[low_spectrum_mask],
        spec_hdu.data[wave_label][0,:]
    ])*u.AA

    flux=np.concatenate([
        spec_hdu.data[flux_label][1,:],#[low_spectrum_mask],
        spec_hdu.data[flux_label][0,:]
    ])

    error=np.concatenate([
        spec_hdu.data[error_label][1,:],#[low_spectrum_mask],
        spec_hdu.data[error_label][0,:]
    ])

    dq=np.concatenate([
        spec_hdu.data["dq"][1,:],#[low_spectrum_mask],
        spec_hdu.data["dq"][0,:]
    ])

    # Generate empty spectrum
    spec_data=genEmptySpecX1D()
    spec_data["net"]=net
    spec_data["wave"]=wave
    spec_data["flux"]=flux
    spec_data["error"]=error
    spec_data["dq"]=dq

    spec_data["exptime"]=spec_hdu.header["EXPTIME"]
    spec_data["header"]=data[0].header
    spec_data["header2"]=data[1].header
    spec_data["grating"]=data[0].header["OPT_ELEM"]

    spec_data["wavelim"]={
        "min":data[0].header["MINWAVE"]*u.AA,
        "max":data[0].header["MAXWAVE"]*u.AA
    }

    return spec_data,spec_hdu

"""Performs continuum fit and line fit on a given line for a given X1D.
Inputs:
- spec_data: single entry in spectrum table representing data for an x1d
- line: Data for spectral line being evaluated
Output:
- Stores the data inside the spec_data struct under the line name.
"""
def process_line(spec_data,line_params,quiet_mode=False):
    name=line_params["line_name"]
    line=line_params["lambda"]

    # Add line to X1D:
    line_data,result=cut_line_data(spec_data,line)
    line_data["name"]=name
    line_data["lambda"]=line
    # Because Python copies by reference, changing line_data changes the entry in here (I think)
    spec_data["lines"].append(line_data)
    
    if not result:
        if not quiet_mode:print(f"\tLine {name}:\n\t\t<outside wavelength range>\n")
        return

    # Estimate of how far around the line to cut when performing continuum fit
    line_threshold=0.6*u.AA

    # Continuum fit (adds data in struct for normalized flux and error, and stores fit params)
    continuum_fit_line_data(
        line_data,
        (np.abs(line_data["line_wave"]-line)>line_threshold) # Cut for continuum fit
    )
    line_data["has_cont"]=True

    flux_clipped,error_clipped=sigma_clip_data(line_data)

    prop_valid=prop_valid_data(line_data["line_vel"],flux_clipped)
    if prop_valid>0.16:
        if not quiet_mode:print(f"\tLine {name}:\n\t\t<not enough valid datapoints> ({prop_valid})\n")
        return
    line_data["bad_line"]=False

    curvefit_params,chisq=gaussian_curve_fit2(
        line_data["line_wave"],
        line,
        flux_clipped,
        error_clipped,
        True
    )

    line_data["line_fit_params"]=curvefit_params
    line_data["chisq"]=chisq

    line_data["lambda_center"]=curvefit_params[1]*u.AA
    line_data["dispersion"]=curvefit_params[2]*u.AA
    line_data["valid"]=True

    if not quiet_mode:
        print(f"""\
    Line {name}:
    \tA = {curvefit_params[0]: .4g}
    \tCenter wavelength = {curvefit_params[1]: .4g} Å
    \tWavelength dispersion = {curvefit_params[2]:.4g} Å
    \tχ²: {chisq:.4g}"""
        )


"""Returns a line data struct with spectrum data around a given line
Outputs:
    - line_data: The struct containing the line data
    - result (boolean): Returns true if the wavelength range is within the range of spectrum data"""
def cut_line_data(spec_data,line,delta_lambda=2*u.AA):
    # Generate an empty line data struct to populate
    line_data=genEmptyLineData()


    wave=spec_data["wave"]
    flux=spec_data["flux"]
    error=spec_data["error"]

    # Check if wavelength range for line isn't within wavelength range of data
    if (line-delta_lambda)<wave.min() or (line+delta_lambda)>wave.max():
        return line_data,False

    mask=(np.abs(wave-line)<delta_lambda)#(np.abs(vel)<(400*u.km/u.s))
    line_data["line_wave"]=wave[mask]
    line_data["line_flux"]=flux[mask]
    line_data["line_error"]=error[mask]
    line_data["line_vel"]=(c.c*(wave[mask]/line-1)).to(u.km/u.s)

    return line_data,True


linfit=fitting.LinearLSQFitter()
def continuum_fit_line_data(line_data,mask,curvefit_degree=2):
    if len(mask)==0:return
    wave=line_data["line_wave"]
    flux=line_data["line_flux"]
    error=line_data["line_error"]

    wave_masked=wave[mask]
    flux_masked=flux[mask]
    # Perform polynomial curve fit to approximate shape of continuum
    curvefit=linfit(
        polynomial.Polynomial1D(degree=curvefit_degree),
        wave_masked,
        flux_masked
    )
    line_data["flux_norm"]=flux/curvefit(wave)
    line_data["error_norm"]=error/curvefit(wave)
    line_data["continuum_params"]=curvefit

def sigma_clip_data(line_data, clipping_sigma=2.5, smoothing_sigma=6):
    # Smoothing the signal out by a lot
    flux_smoothed_1=gaussian_filter1d(flux_norm:=line_data["flux_norm"],smoothing_sigma)
    wave=line_data["line_wave"]
    
    # Comparing the actual flux to the smoothed version before running sigma clipping:
    # that way the code can remove random spikes in the data without also cutting out the absorption line.
    sigma_clip_mask=sigma_clip(
        flux_norm - flux_smoothed_1,
        sigma=clipping_sigma,
        masked=True
    ).mask # only need the mask, since it needs to filter the velocity and flux arrays

    sigma_clip_mask |= ~np.isfinite(flux_norm)

    # Filter data based on the mask 
    wave_clipped_0=wave[~sigma_clip_mask]
    flux_clipped_0=line_data["flux_norm"][~sigma_clip_mask]
    error_clipped_0=line_data["error_norm"][~sigma_clip_mask]

    # Re-interpolate flux (and error) to patch holes in data
    flux_out=interp1d(
        wave_clipped_0,
        flux_clipped_0,
        fill_value='extrapolate'
    )(wave)
    error_out=interp1d(
        wave_clipped_0,
        error_clipped_0,
        fill_value='extrapolate'
    )(wave)

    # Notably, we actually DON'T store this in the line data object,
    # because this isn't actually something we want to store permanently
    return flux_out,error_out

# Gaussian absorption line shape
def gaussian_line_shape(v,a,v0,sigma_v):
    return 1-a*np.exp(-0.5*(v-v0)**2/sigma_v**2)

def gaussian_curve_fit2(wave,line,flux,error,smooth_sigma=0):
    if smooth_sigma==0:
        flux_curvefit=flux
    else:
        # Smooth data slightly before running the curve fit
        flux_curvefit=gaussian_filter1d(flux,smooth_sigma)

    params,cov=curve_fit(
        gaussian_line_shape,
        wave.value,
        flux_curvefit,
        p0=(
            max(0,min(1,1-flux_curvefit.min())), #amplitude
            line.value, #center velocity
            0.07 #stddev (TODO: calculate this from FWHM (how to do that??))
        ),
        bounds=(
            # min
            (
                0,              # amplitude
                line.value-10.2, # center wavelength
                0.01            # wavelength dispersion
            ),
            # max
            (
                1,              # amplitude
                line.value+10.2, # center wavelength
                1               # wavelength dispersion
            ),
        )
    )

    # Reduced chi-squared value for fit (using unsmoothed data)
    # The model has 3 parameters, so there are N-3 degrees of freedom.
    chisq=np.sum(
        (flux-gaussian_line_shape(wave.value,*params))**2/error**2
    )/(len(flux)-3)

    return params,chisq

def prop_valid_data(wave,flux,elementWidth=0):
    # Boolean array (True if flux is nonpositive, NaN, or infinity)
    # (not sure which of these can even happen, just want to be safe)
    invalid=((flux<=0)|np.isnan(flux)|np.isinf(flux))

    # Convert the booleans to floats (0 or 1) and take the mean to find the proportion of invalid data points
    prop_invalid_flux=(invalid.astype(np.float32)).mean()

    # Look for weird behavior in the wavelength/velocity array:
    # the points should all be the same distance apart, and if they aren't there's something weird going on

    # Getting rid of NaNs early, because they're invalid anyway
    wave_diff=np.diff(wave[np.isfinite(wave)])

    #TODO: This is kind of janky. Maybe there's a better way of doing this?
    if elementWidth==0:
        elementWidth=np.median(np.diff(wave_diff))

    prop_invalid_wave=np.isclose(wave_diff,elementWidth).astype(np.float32).mean()

    return prop_invalid_flux+prop_invalid_wave
