import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import astropy.stats as stats
import spectrum_align

def diagnostic_plot(target_dir,spec_table,line,wlen,output_table,filenames,line_lib,TGT_LINE_NAME_LIST):
    fig=plt.figure(figsize=(10,8))
    ax=fig.subplots(3,1)

    j=-1
    if line not in TGT_LINE_NAME_LIST:
        for f in filenames:
            spec_data=spec_table[f]
            line_data,valid=spectrum_align.cut_line_data(
                spec_data,
                wlen,
            )
            line_data["name"]=line
            spec_data["lines"].append(line_data)

            if not valid:
                continue

            spectrum_align.continuum_fit_line_data(
                line_data,
                (np.abs(line_data["line_wave"]-wlen)>(0.6*u.AA))
            )
            line_data["has_cont"]=True

            flux_clipped,error_clipped=spectrum_align.sigma_clip_data(line_data)
            if spectrum_align.prop_valid_data(line_data["line_vel"],flux_clipped)>0.16:
                continue
            line_data["bad_line"]=False

    else:        
        j=[l["name"] for l in spec_table[filenames[0]]["lines"]].index(line)

    for i,f in enumerate(filenames):
        wave=spec_table[f]["lines"][j]["line_wave"].copy()
        vel=spec_table[f]["lines"][j]["line_vel"].copy()
        dL=0*u.AA
        dv=0*u.km/u.s
        # if line in output_table.colnames:
        dL=output_table["mean_dlambda"][i]
        dv=output_table["mean_dlambda"][i]*c.c/wlen

        lbl=f.split("\\")[-1].removesuffix(".fits")
        if spec_table[f]["lines"][j]["bad_line"]:
            try:
                ax[0].step(
                    wave,
                    spec_table[f]["lines"][j]["line_flux"],
                    "--",
                    lw=0.5,
                    alpha=0.2,
                    label=lbl+" (bad)",
                )
                ax[1].step(
                    wave,
                    spec_table[f]["lines"][j]["flux_norm"],
                    "--",
                    lw=0.5,
                    alpha=0.2,
                    label=lbl+" (bad)",
                )
                ax[2].step(
                    vel,
                    spec_table[f]["lines"][j]["flux_norm"],
                    "--",
                    lw=0.5,
                    alpha=0.2,
                    label=lbl+" (bad)",
                )
            except ValueError:pass
        elif spec_table[f]["lines"][j]["has_cont"]:
            wave_plot=ax[0].step(
                wave,
                spec_table[f]["lines"][j]["line_flux"],
                lw=1,
                label=lbl
            )
            wave_plot_norm=ax[1].step(
                wave,
                spec_table[f]["lines"][j]["flux_norm"],
                lw=0.5
            )
            wave_plot_norm=ax[1].step(
                wave-dL,
                spec_table[f]["lines"][j]["flux_norm"],
                color=wave_plot_norm[0].get_color(),
                lw=1,
                label=lbl,
            )
            vel_plot=ax[2].step(
                vel,
                spec_table[f]["lines"][j]["flux_norm"],
                lw=0.5,
            )
            vel_plot=ax[2].step(
                vel-dv,
                spec_table[f]["lines"][j]["flux_norm"],
                color=vel_plot[0].get_color(),
                lw=1,
                label=lbl
            )
        if spec_table[f]["lines"][j]["valid"]:
            ax[1].plot(
                wave,
                spectrum_align.gaussian_line_shape(wave.value,*(spec_table[f]["lines"][j]["line_fit_params"])),
                "--",
                color=wave_plot_norm[0].get_color(),
                label=lbl+" curve fit"
            )

            ax[0].axvline(
                x=spec_table[f]["lines"][j]["lambda_center"].value,
                c=wave_plot[0].get_color(),
                lw=0.5
            )
            ax[1].axvline(
                x=spec_table[f]["lines"][j]["lambda_center"].value,
                c=wave_plot_norm[0].get_color()
            )
            ax[2].axvline(
                x=((spec_table[f]["lines"][j]["lambda_center"]/line_lib["lambda"][j]-1)*c.c).to(u.km/u.s).value,
                c=vel_plot[0].get_color(),
                lw=0.5
            )

            ax[1].text(
                0.01, 0.4-0.10*i,
                "${\\delta\\lambda}=$ "+"{:.4g}".format(output_table[line][i].to(u.mAA).value)+" mÅ",
                horizontalalignment='left',
                verticalalignment='top',
                transform = ax[1].transAxes,
                color=wave_plot_norm[0].get_color()
            )
            ax[2].text(
                0.01, 0.4-0.10*i,
                "${\\delta v}=$ "+"{:.4g}".format((output_table[line][i]*c.c/wlen).to(u.km/u.s).value)+" km/s",
                horizontalalignment='left',
                verticalalignment='top',
                transform = ax[2].transAxes,
                color=vel_plot[0].get_color()
            )
    ax[0].set_xlabel("$\lambda (\AA)$")
    ax[0].set_ylabel("Flux (ergs s^-1 m^-2 sr^-1)")

    ax[1].set_xlabel("$\lambda (\AA)$")
    ax[1].set_ylabel("Normalized flux")

    ax[2].set_xlabel("Heliocentric velocity (km/s)")
    ax[2].set_ylabel("Normalized flux")

    low,high=plot_clip_lim(
        np.concatenate(
            [
                spec_table[f]["lines"][j]["line_flux"]
                for f in filenames
                if spec_table[f]["lines"][j]["has_cont"]
            ]
        )
    )
    ax[0].set_ylim(
        low,high
    )
    ax[1].set_ylim(-0.1,1.8)
    ax[2].set_ylim(-0.1,1.8)

    for a in ax:
        a.text(
            0.99, 0.1,
            line,
            horizontalalignment='right',
            verticalalignment='center',
            transform = a.transAxes
        )

        a.legend(loc="upper right",fontsize="x-small")
        a.minorticks_on()
        a.grid(which="minor",alpha=0.1)
        a.grid(which="major",alpha=0.3)
    fig.suptitle(line)
    # fig.canvas.manager.full_screen_toggle()
    fig.tight_layout()
    fig.savefig(
        f"{target_dir}/diagnostic_{line}.pdf"
    )
    fig.savefig(
        f"{target_dir}/diagnostic_{line}.png"
    )
    # fig.canvas.manager.full_screen_toggle()

def plot_clip_lim(data,sigma=3):
    clip_data=stats.sigma_clip(
        data,
        sigma=sigma
    )
    return clip_data.min()-0.2*(clip_data.max()-clip_data.min()),\
            clip_data.max()+0.2*(clip_data.max()-clip_data.min())

def print_spec_table(obj,depth=0):
    ind="\t"*depth
    if type(obj)==dict:
        print(ind+"{")
        for key in sorted(obj.keys()):
            print(ind+"    "+key+":")
            print_spec_table(obj[key],depth+1)
        print(ind+"}")

    elif type(obj)==list:
        print(ind+"[")
        for l in obj:
            print_spec_table(l,depth+1)
        print(ind+"]")
    elif ((type(obj)==np.ndarray) or (type(obj)==u.Quantity and (type(obj.value)==np.ndarray))) and len(obj)>10:
        print(ind+"[...]")
    else:
        print(ind+repr(obj))
