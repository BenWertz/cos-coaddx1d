import numpy as np
import spectrum_align
import astropy.units as u
import astropy.stats as stats
import astropy.constants as c
from linetools.lists.linelist import LineList
import astropy.io.fits as fits
from astropy.table import *
import matplotlib.pyplot as plt
import os
import sys
import glob
from scipy.interpolate import *
import diagnostic_plots
import coaddition
import utils

# Command-line arguments
arg_TgtFolder=sys.argv[1]
arg_Grating=sys.argv[2]
arg_COADD_MODE=(sys.argv[3] if len(sys.argv)>=3 else "exptime")

wavelength_range={
    "g130m":[900,1450],
    "g160m":[1400,1775],
    "g130m_g160m":[900,1775],
}

def matchGrating(grating,mode):
    match mode.lower():
        case "g130m":
            return (grating.lower()=="g130m")
        case "g160m":
            return (grating.lower()=="g160m")
        case "g130m_g160m":
            return (grating.lower() in["g130m","g160m"])
        case _:
            raise NameError(f"Invalid grating '{mode.lower()}'")
            return False
# G130M_RANGE=[

# ]

llist_ism = LineList('ISM')._data.to_pandas()

    # "SII 1250","SII 1253"
target_lines=[
    "FeII 1144",
    "PII 1152",
    "SII 1253",
    "SII 1250",
    "AlII 1670",
    "FeII 1608",
    # "SiIV 1393",
    # "SiIV 1402"
    # "OI 1302",
]

full_line_list=[
    "PII 1152",
    "SII 1250",
    "SII 1253",
    "SiIV 1393",
    "SiIV 1402",
    "CII 1334",
    "SII 1190",
    "SII 1193",
    "SII 1260",
    "SIII 1206",
]

#TODO: Add actual validation lines for G160M grating
full_line_list+=target_lines
full_line_list=[line for i,line in enumerate(full_line_list) if (line not in full_line_list[:max(0,i-1)])]

llist_ism['species'] = llist_ism['name'].str.split().str[0]

def get_line_lib(line_list,wrange):
    sub_llist = llist_ism.query(f'(name in @line_list) and (wrest>{wrange[0]}) and (wrest<{wrange[1]})').sort_values("wrest") # from low to high

    return {"line_name": [line.replace(' ', '') for line in sub_llist['name']],
                "lambda": np.array(sub_llist['wrest'])*u.AA, 
                "f": list(sub_llist['f'])}

# Only lines that are suitable for line-fitting
line_lib=get_line_lib(target_lines,wavelength_range[arg_Grating.lower()])

# Includes lines that are too noisy to fit
full_line_lib=get_line_lib(full_line_list,wavelength_range[arg_Grating.lower()])

# Initial list of files to be read (some may be discarded if they don't have the right grating)
filenames_prelim = glob.glob(arg_TgtFolder+"/*x1d.fits")

filenames=[]
spec_table={}
for file in filenames_prelim:
    spec_data,spec_hdu=spectrum_align.read_x1d_file(file)
    # Only accept gratings that match the mode
    if matchGrating(spec_data["grating"],arg_Grating):
        # utils.viewX1DFile(file)
        print(spec_data["grating"],arg_Grating)

        filenames.append(file)
        spec_table[file]=spec_data

if len(filenames)==0:
    exit("\n[Error] No x1ds with the proper grating were found!")
    # exit(0)

output_table=QTable(
    [filenames]+[np.nan*np.ones(len(filenames))*u.AA for line in line_lib["lambda"]],
    names=["filename"]+line_lib["line_name"]
)

for i,filename in enumerate(filenames):
    print(f"File '{filename}':")
    spec_data=spec_table[filename]
    for idx,name,line in zip(np.arange(len(line_lib["f"])),line_lib["line_name"],line_lib["lambda"]):
        spectrum_align.process_line(
            spec_data,
            {
                "lambda":line,
                "line_name":name
            }
        )
        output_table[name][i]=spec_data["lines"][idx]["lambda_center"]
    print("\n========\n")

best_idx=-1
best_num_valid=0
for i in range(len(filenames)):
    num_valid=0
    for col in output_table.colnames[1:]:
        if not np.isnan(output_table[col][i]):
            num_valid+=1
    if num_valid>best_num_valid:
        best_idx=i
        best_num_valid=num_valid
if best_idx==-1:
    best_idx=0

ref_idx=best_idx

for col in output_table.colnames[1:]:
    ref_wavelength=output_table[col][ref_idx]
    for j in range(len(output_table[col])):
        if not np.isnan(ref_wavelength):
            output_table[col][j]-=ref_wavelength

# mean_output_table=QTable(
#     [filenames]+[np.nan*np.ones(len(filenames))*u.AA],
#     names=["filename","Dlambda_mean"]
# )

output_table.add_column(
    Column(
        data=np.array([np.mean([
            output_table[col1][j].value for col1 in output_table.colnames[1:] if not np.isnan(output_table[col1][j].value)
        ]) for j in range(len(output_table[output_table.colnames[-1]]))])*u.AA,
        name="mean_dlambda"
    )
)

print("Δλ table")
output_table.pprint(max_lines=-1,max_width=-1)

velocity_table=QTable(
    [filenames]+[np.nan*np.ones(len(filenames))*u.km/u.s for line in line_lib["lambda"]],
    names=["filename"]+line_lib["line_name"]
)

TGT_LINE_NAME_LIST=line_lib["line_name"]

os.chdir(sys.path[0])
if not os.path.exists("outputs"):
    os.mkdir("outputs")
    os.mkdir("outputs/figures")

for line,wlen in zip(full_line_lib["line_name"],full_line_lib["lambda"]):
    diagnostic_plots.diagnostic_plot(
        "outputs/figures",
        spec_table,line,wlen,output_table,
        filenames,
        line_lib,
        TGT_LINE_NAME_LIST
    )

coadd_data_unbinned=coaddition.coadd_spec(
    spec_table,
    filenames,
    output_table,
    arg_COADD_MODE,
    1
)

coadd_data=coaddition.coadd_spec(
    spec_table,
    filenames,
    output_table,
    arg_COADD_MODE,
    3
)
ref_wave,coadded_flux,coadded_error=coadd_data["wave"],coadd_data["flux"],coadd_data["error"]

# print(ref_wave[np.isnan(coadded_error)])
# print("Min:")
# print([spec_table[f]["header"]["MINWAVE"] for f in filenames])
# print("Max:")
# print([spec_table[f]["header"]["MAXWAVE"] for f in filenames])

# print("Wave array min")
# print([spec_table[f]["wave"].min().value for f in filenames])
# print("Wave array max")
# print([spec_table[f]["wave"].max().value for f in filenames])


plt.figure(figsize=(12,6))
lim=diagnostic_plots.plot_clip_lim(coadded_flux,5)
plt.ylim(lim[0],lim[1])

# plt.step(coadd_data_unbinned["wave"],coadd_data_unbinned["flux"],color="b",lw=2,label=f"unbinned")

plt.step(ref_wave,coadded_flux,color="k",lw=2,label=f"Coadded spectrum ({coadd_data['weighing_method']})")

plt.step(ref_wave,coadded_error,color="blue",lw=1,alpha=0.5,label="Error")

ref_coadd_data=fits.open("PG1011-040_G130M+G160M_coadd_x1d_bin3_m1.fits")[1]

# coadd_data=spectrum_align.read_x1d_file("PG1011-040_G130M+G160M_coadd_x1d_bin3_m1.fits",wave_label="WAVE")
plt.step(ref_coadd_data.data["WAVE"][0,:],ref_coadd_data.data["FLUX"][0,:],color="orange",lw=1.5,label="Actual coadded spectrum")

plt.step(ref_coadd_data.data["WAVE"][0,:],ref_coadd_data.data["ERROR"][0,:],color="red",lw=1,alpha=0.5,label="(actual) Error")


plt.legend(loc="upper left",fontsize="x-small")
plt.xlabel("$\lambda (\AA)$")
plt.ylabel("Flux (ergs s^-1 m^-2 sr^-1)")
plt.minorticks_on()
plt.savefig("outputs/figures/coadded.png")
plt.show()

coaddition.save_spec_data(
    "outputs/coadd",
    coadd_data
)
coaddition.save_spec_data(
    "outputs/coadd_unbinned",
    coadd_data_unbinned
)