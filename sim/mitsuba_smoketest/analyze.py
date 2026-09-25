"""Sanity-check the clear and foggy transient renders against scene ground truth.

Checks:
  clear: one peak per object pixel at the ground-truth time of flight (ToF)
  fog:   same peak (attenuated), plus scattered light before it (backscatter)
         and after it (multiple-scattering tail); fog-only pixels compared with
         an analytic single-scattering prediction.
Writes outputs/results.json and figures/*.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "outputs", HERE / "figures"
FIG.mkdir(exist_ok=True)

C = 299792458.0
BIN_OPL = 0.0299792458          # m of optical path per bin (100 ps)
BIN_NS = BIN_OPL / C * 1e9
PULSE_FWHM_NS = 1.0             # placeholder pulse width
LASER = np.array([0.0, 1.0, -0.001])
LASER_RADIUS, LASER_RADIANCE, LASER_HALF_ANGLE = 0.01, 1e4, 30.0
FOG_FRONT_Z = 0.05
SUB = 4                          # sub-rays per pixel axis for ground truth


def convolve_pulse(h, fwhm_ns=PULSE_FWHM_NS):
    """Linear (non-circular) convolution of the last axis with a Gaussian pulse."""
    sigma = fwhm_ns / BIN_NS / (2 * np.sqrt(2 * np.log(2)))
    half = int(np.ceil(4 * sigma))
    k = np.arange(-half, half + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    out = np.zeros_like(h)
    T = h.shape[-1]
    for kk, ww in zip(k, w):
        if kk >= 0:
            out[..., kk:] += ww * h[..., :T - kk]
        else:
            out[..., :T + kk] += ww * h[..., -kk:]
    return out


def ground_truth(scene_path, W, H):
    import mitsuba as mi
    import drjit as dr
    mi.set_variant("llvm_ad_mono")
    import mitransient  # noqa: F401
    scene = mi.load_file(str(scene_path), sigma_t=0.0, spp=1)
    sensor = scene.sensors()[0]

    ys, xs, jy, jx = np.meshgrid(np.arange(H), np.arange(W), np.arange(SUB), np.arange(SUB),
                                 indexing="ij")
    px = ((xs + (jx + 0.5) / SUB) / W).ravel()
    py = ((ys + (jy + 0.5) / SUB) / H).ravel()
    ray, _ = sensor.sample_ray(0.0, 0.5, mi.Point2f(px, py), mi.Point2f(0.5, 0.5))
    origin = np.array(ray.o).T.copy()
    direction = np.array(ray.d).T.copy()

    t_acc = np.zeros(len(px))
    hit_p = np.full((len(px), 3), np.nan)
    active = np.ones(len(px), bool)
    r = mi.Ray3f(ray)
    for _ in range(6):  # step through null (fog-box) faces to the first opaque hit
        si = scene.ray_intersect(r, mi.Bool(active))
        valid = np.array(si.is_valid())
        is_null = np.array(mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.Null)) & valid
        t = np.array(si.t)
        t_acc[active & valid] += t[active & valid]
        opaque = active & valid & ~is_null
        hit_p[opaque] = np.array(si.p).T[opaque]
        active = active & is_null
        if not active.any():
            break
        r = si.spawn_ray(r.d)

    # Classify by geometry, not shape id: Mitsuba merges same-BSDF meshes and
    # the first render pair had both boxes as one unnamed mesh. AABBs below
    # match scene.xml (sphere A / cube B / cube C / ground y=0).
    names = np.full(len(px), "sky", dtype=object)
    hit = np.isfinite(hit_p).all(-1)
    p = hit_p
    names[hit & (np.linalg.norm(p - np.array([-1.4, 0.5, 3.8]), axis=1) <= 0.51)] = "obj_A_sphere"
    names[hit & (p[:, 0] >= 0.99) & (p[:, 0] <= 2.01) & (p[:, 1] >= -0.01) & (p[:, 1] <= 1.01)
          & (p[:, 2] >= 4.99) & (p[:, 2] <= 6.01)] = "obj_B_box"
    names[hit & (p[:, 0] >= -0.81) & (p[:, 0] <= 0.81) & (p[:, 1] >= -0.01) & (p[:, 1] <= 1.51)
          & (p[:, 2] >= 6.49) & (p[:, 2] <= 7.51)] = "obj_C_box"
    still = hit & (names == "sky")
    names[still & (np.abs(p[:, 1]) < 0.02)] = "ground"
    names[still & (names == "sky")] = "other"

    opl = np.full(len(px), np.nan)
    opl[hit] = t_acc[hit] + np.linalg.norm(hit_p[hit] - LASER, axis=1)
    reshape = lambda a: a.reshape(H, W, SUB * SUB)
    names_p, opl_p = reshape(names), reshape(opl)
    finite = np.isfinite(opl_p)
    opl_mean = np.divide(np.nansum(np.where(finite, opl_p, 0), -1),
                         np.maximum(finite.sum(-1), 1), where=finite.any(-1))
    opl_mean = np.where(finite.any(-1), opl_mean, np.nan)
    return {
        "names": names_p,
        "label": names_p[..., 0],
        "pure": (names_p == names_p[..., :1]).all(-1),
        "opl_mean": opl_mean,
        "opl_min": np.where(finite.any(-1), np.nanmin(np.where(finite, opl_p, np.inf), -1), np.nan),
        "opl_max": np.where(finite.any(-1), np.nanmax(np.where(finite, opl_p, -np.inf), -1), np.nan),
        "t_cam": reshape(t_acc),
        "origin": origin.reshape(H, W, SUB * SUB, 3),
        "dir": direction.reshape(H, W, SUB * SUB, 3),
    }


def single_scatter_prediction(gt, sigma_t, albedo, g, nbins, mask=None):
    """Analytic single-scattering transient per pixel, point-emitter approximation:
    dL/dr = sigma_s p_HG(mu) I(theta_e) / R^2 * exp(-sigma_t * path_in_fog), deposited at
    OPL = r + R. Averaged over the SUB x SUB sub-rays like the film's box filter."""
    H, W, S, _ = gt["dir"].shape
    sigma_s = albedo * sigma_t
    area = np.pi * LASER_RADIUS ** 2
    r = np.arange(1e-4, nbins * BIN_OPL / 2 + 0.05, 1e-3)
    dr_ = np.diff(r, append=r[-1] + 1e-3)
    pred = np.zeros((H, W, nbins))
    t_hit = np.where(np.isfinite(gt["t_cam"]) & (gt["names"] != "sky"), gt["t_cam"], np.inf)
    if mask is None:
        ys, xs = np.indices((H, W)).reshape(2, -1)
    else:
        ys, xs = np.nonzero(mask)
    for yy, xx in zip(ys, xs):
            acc = np.zeros(nbins)
            for s in range(S):
                o, d = gt["origin"][yy, xx, s], gt["dir"][yy, xx, s]
                rr, ww = r[r < t_hit[yy, xx, s]], dr_[r < t_hit[yy, xx, s]]
                x = o[None] + rr[:, None] * d[None]
                v = x - LASER
                R = np.linalg.norm(v, axis=1)
                u = v / R[:, None]                      # propagation dir laser -> x
                cos_e = u[:, 2]                         # disk normal is +z
                lit = cos_e > np.cos(np.radians(LASER_HALF_ANGLE))
                mu = np.einsum("ij,j->i", u, -d)        # cos(incident prop, outgoing prop)
                p = (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * mu) ** 1.5)
                in_fog = x[:, 2] > FOG_FRONT_Z
                r0 = (FOG_FRONT_Z - o[2]) / d[2]
                len_cam = np.clip(rr - r0, 0, None)
                len_em = R * np.clip((x[:, 2] - FOG_FRONT_Z) / (x[:, 2] - LASER[2]), 0, None)
                f = sigma_s * p * LASER_RADIANCE * area * cos_e / R ** 2 \
                    * np.exp(-sigma_t * (len_cam + len_em)) * lit * in_fog
                b = ((rr + R) / BIN_OPL).astype(int)
                ok = b < nbins
                acc += np.bincount(b[ok], weights=(f * ww)[ok], minlength=nbins)[:nbins]
            pred[yy, xx] = acc / S
    return pred


def local_peaks(h, frac=0.05):
    thr = frac * h.max()
    m = (h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:]) & (h[1:-1] > thr)
    return np.nonzero(m)[0] + 1


def main():
    clear, fog = np.load(OUT / "clear.npz"), np.load(OUT / "fog.npz")
    mc, mf = (json.loads((OUT / f"{n}.json").read_text()) for n in ("clear", "fog"))
    hc, hf = clear["transient"].astype(np.float64), fog["transient"].astype(np.float64)
    H, W, T = hc.shape
    sigma_t, albedo, g = (mf["params"][k] for k in ("sigma_t", "albedo", "g"))
    gt = ground_truth(HERE / "scene.xml", W, H)
    cc, cf = convolve_pulse(hc), convolve_pulse(hf)

    label, pure = gt["label"], gt["pure"]
    with np.errstate(invalid="ignore"):
        b_lo = np.floor(gt["opl_min"] / BIN_OPL)
        b_hi = np.floor(gt["opl_max"] / BIN_OPL)
    b_lo = np.where(np.isfinite(b_lo), b_lo, -1).astype(int) - 1
    b_hi = np.where(np.isfinite(b_hi), b_hi, -1).astype(int) + 1
    b_mean = gt["opl_mean"] / BIN_OPL
    tidx = np.arange(T)

    res = {"sigma_t": sigma_t, "albedo": albedo, "g": g,
           "pixel_counts": {n: int((label == n).sum()) for n in np.unique(label)},
           "objects": {}}
    objects = ["obj_A_sphere", "obj_B_box", "obj_C_box"]
    rep_pixel = {}
    for name in objects:
        m = (label == name) & pure & (b_lo >= 0) & (b_hi < T)
        if not m.any():
            res["objects"][name] = {"n_pure_pixels": 0, "skipped": True}
            continue
        ys, xs = np.nonzero(m)
        o = {"n_pure_pixels": int(m.sum())}
        in_win = (tidx[None] >= b_lo[m][:, None]) & (tidx[None] <= b_hi[m][:, None])
        before = tidx[None] < b_lo[m][:, None]
        after = tidx[None] > b_hi[m][:, None]
        for tag, h, hcv in (("clear", hc[m], cc[m]), ("fog", hf[m], cf[m])):
            am_raw = h.argmax(-1)
            # object peak searched in its expected window (+-3 bins), raw delta-pulse response
            win3 = (tidx[None] >= b_lo[m][:, None] - 3) & (tidx[None] <= b_hi[m][:, None] + 3)
            am_obj = np.where(win3, h, -1).argmax(-1)
            ok_obj = (am_obj >= b_lo[m]) & (am_obj <= b_hi[m])
            e_tot = h.sum(-1)
            e_win = (h * in_win).sum(-1)
            e_pre = (h * before).sum(-1)
            e_post = (h * after).sum(-1)
            am_conv = hcv.argmax(-1)
            strongest_is_obj = (am_conv >= b_lo[m] - 5) & (am_conv <= b_hi[m] + 5)
            n_peaks = np.array([len(local_peaks(x)) for x in hcv])
            o[tag] = {
                "global_raw_argmax_at_gt_frac": float(((am_raw >= b_lo[m]) & (am_raw <= b_hi[m])).mean()),
                "object_peak_at_gt_frac": float(ok_obj.mean()),
                "object_peak_err_bins_vs_mean_median": float(np.median(np.abs(am_obj + 0.5 - b_mean[m]))),
                "object_peak_err_bins_vs_mean_max": float(np.max(np.abs(am_obj + 0.5 - b_mean[m]))),
                "energy_frac_in_gt_window_median": float(np.median(e_win / e_tot)),
                "energy_frac_before_median": float(np.median(e_pre / e_tot)),
                "energy_frac_after_median": float(np.median(e_post / e_tot)),
                "pre_over_peak_energy_median": float(np.median(e_pre / e_win)),
                "post_over_peak_energy_median": float(np.median(e_post / e_win)),
                "strongest_pulsed_return_is_object_frac": float(strongest_is_obj.mean()),
                "pulsed_local_peaks_gt5pct_median": float(np.median(n_peaks)),
                "pulsed_single_peak_frac": float((n_peaks == 1).mean()),
                "e_win_median": float(np.median(e_win)),
            }
        # Beer-Lambert check on the direct return: fog/clear vs exp(-sigma_t * path in fog)
        e_win_c = (hc[m] * in_win).sum(-1)
        e_win_f = (hf[m] * in_win).sum(-1)
        # camera leg + return leg, each minus the 5 cm of vacuum in front of the fog box
        opl = gt["opl_mean"][m]
        path_fog = opl - 2 * FOG_FRONT_Z  # small-angle approximation of the vacuum legs
        expect = np.exp(-sigma_t * path_fog)
        o["gt_one_way_range_m"] = {"min": float(gt["opl_min"][m].min() / 2),
                                   "median": float(np.median(opl) / 2),
                                   "max": float(gt["opl_max"][m].max() / 2)}
        o["gt_tof_ns_median"] = float(np.median(opl) / C * 1e9)
        o["attenuation_measured_median"] = float(np.median(e_win_f / e_win_c))
        o["attenuation_beer_lambert_median"] = float(np.median(expect))
        o["attenuation_ratio_measured_over_bl_median"] = float(np.median(e_win_f / e_win_c / expect))
        # representative pixel: pure pixel with the tightest GT spread, nearest the centroid
        spread = (b_hi - b_lo)[m]
        d2 = (ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2
        k = np.lexsort((d2, spread))[0]
        rep_pixel[name] = (int(ys[k]), int(xs[k]))
        o["rep_pixel_yx"] = rep_pixel[name]
        res["objects"][name] = o

    # fog-only pixels: no opaque surface anywhere along the ray, well inside the laser cone
    theta = np.degrees(np.arccos(np.clip(gt["dir"][..., 2], -1, 1))).max(-1)
    sky = (label == "sky") & pure & (theta < LASER_HALF_ANGLE - 2)
    res["sky"] = {"n_pixels": int(sky.sum()),
                  "clear_total_energy_max": float(hc[sky].sum(-1).max()),
                  "fog_total_energy_median": float(np.median(hf[sky].sum(-1)))}
    print("computing analytic single-scattering prediction ...", file=sys.stderr)
    need = sky.copy()
    for name in objects:
        if not res["objects"][name].get("skipped"):
            need |= (label == name) & pure
    ss = single_scatter_prediction(gt, sigma_t, albedo, g, T, mask=need)
    if not sky.any():
        raise SystemExit("no fog-only pixels inside the laser cone; cannot test backscatter")
    mc_sky, ss_sky = hf[sky].mean(0), ss[sky].mean(0)
    rng_m = (tidx + 0.5) * BIN_OPL / 2
    ratios = {}
    for r_q in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        sel = np.abs(rng_m - r_q) <= 0.15
        ratios[f"{r_q:.2f}m"] = float(mc_sky[sel].sum() / ss_sky[sel].sum())
    res["sky"]["mc_over_single_scatter_by_range"] = ratios
    cs = convolve_pulse(mc_sky)
    res["sky"]["pulsed_backscatter_peak_range_m"] = float(rng_m[cs.argmax()])
    res["sky"]["pulsed_backscatter_peak_tof_ns"] = float((cs.argmax() + 0.5) * BIN_NS)
    res["sky"]["pulsed_backscatter_onset_range_m"] = float(rng_m[np.nonzero(cs > 0.1 * cs.max())[0][0]])
    for r_q in (1.0, 3.0, 6.0):
        res["sky"][f"pulsed_backscatter_at_{r_q:.0f}m_over_peak"] = float(
            cs[np.argmin(np.abs(rng_m - r_q))] / cs.max())
    # object pixels in fog: pre-object backscatter vs single-scatter prediction
    for name in objects:
        if res["objects"][name].get("skipped"):
            continue
        m = (label == name) & pure & (b_lo >= 0) & (b_hi < T)
        pre = tidx[None] < b_lo[m][:, None] - 3
        den = (ss[m] * pre).sum()
        res["objects"][name]["fog_pre_mc_over_single_scatter"] = (
            float((hf[m] * pre).sum() / den) if den > 0 else None)

    # ---------------- figures ----------------
    x_m = rng_m
    rng_map = lambda h: np.where(h.max(-1) > 0, (h.argmax(-1) + 0.5) * BIN_OPL / 2, np.nan)
    lab_ids = {n: i for i, n in enumerate(["sky", "ground", *objects, "fog_box", "laser"])}
    lab_img = np.vectorize(lambda s: lab_ids.get(s, -1))(label).astype(float)
    gt_rng = np.where(label != "sky", gt["opl_mean"] / 2, np.nan)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.2))
    panels = [
        (lab_img, "Ground-truth first surface\n0 sky, 1 ground, 2 A, 3 B, 4 C", "tab10", None),
        (gt_rng, "Ground-truth one-way range (m)", "viridis", (0, 7.7)),
        (clear["steady"], "Steady (time-summed), clear", "gray", (0, np.percentile(clear["steady"], 99.5))),
        (fog["steady"], f"Steady, fog $\\sigma_t$={sigma_t}/m (same scale)", "gray",
         (0, np.percentile(clear["steady"], 99.5))),
        (rng_map(cc), "Strongest pulsed return range (m), clear", "viridis", (0, 7.7)),
        (rng_map(cf), "Strongest pulsed return range (m), fog", "viridis", (0, 7.7)),
    ]
    for a, (img, title, cmap, lim) in zip(ax.ravel(), panels):
        im = a.imshow(img, cmap=cmap, vmin=None if lim is None else lim[0],
                      vmax=None if lim is None else lim[1], interpolation="nearest")
        a.set_title(title, fontsize=10)
        a.set_xticks([]), a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046)
    for name, (yy, xx) in rep_pixel.items():
        for a in (ax[0, 0], ax[1, 1]):
            a.plot(xx, yy, "r+", ms=9)
    fig.suptitle("mitransient smoke test: 64x64 px, 512 x 100 ps bins (placeholders)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "overview.png", dpi=110)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.2))
    titles = {"obj_A_sphere": "A: sphere", "obj_B_box": "B: 1 m cube", "obj_C_box": "C: far box"}
    for a, name in zip(ax.ravel()[:3], objects):
        if name not in rep_pixel:
            a.set_title(f"{titles[name]}: no pure pixels")
            continue
        yy, xx = rep_pixel[name]
        a.semilogy(x_m, np.maximum(cc[yy, xx], 1e-14), "C0", lw=1.4, label="clear")
        a.semilogy(x_m, np.maximum(cf[yy, xx], 1e-14), "C3", lw=1.4, label=f"fog $\\sigma_t$={sigma_t}/m")
        a.axvline(gt["opl_mean"][yy, xx] / 2, color="k", ls="--", lw=1, label="ground-truth ToF")
        peak = max(cc[yy, xx].max(), cf[yy, xx].max())
        a.set_ylim(peak * 1e-5, peak * 3)
        a.set_title(f"{titles[name]}, pixel (y={yy}, x={xx}), 1 ns pulse applied", fontsize=10)
        a.set_xlabel("one-way range (m) = OPL / 2"), a.set_ylabel("radiance per 100 ps bin (a.u.)")
        a.legend(fontsize=8)
        sec = a.secondary_xaxis("top", functions=(lambda r: 2 * r / C * 1e9, lambda t: t * C / 2e9))
        sec.set_xlabel("time of flight (ns)", fontsize=8)
    a = ax[1, 1]
    a.semilogy(x_m, np.maximum(convolve_pulse(hc[sky].mean(0)), 1e-14), "C0", lw=1.4, label="clear (all zero)")
    a.semilogy(x_m, np.maximum(cs, 1e-14), "C3", lw=1.4, label="fog, Monte Carlo")
    a.semilogy(x_m, np.maximum(convolve_pulse(ss_sky), 1e-14), "k:", lw=1.4, label="fog, analytic single scattering")
    a.set_ylim(cs.max() * 1e-4, cs.max() * 3)
    a.set_title(f"Fog-only pixels (no surface), mean of {int(sky.sum())} px, 1 ns pulse applied", fontsize=10)
    a.set_xlabel("one-way range (m) = OPL / 2"), a.set_ylabel("radiance per 100 ps bin (a.u.)")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "histograms.png", dpi=110)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax[0].loglog(x_m, mc_sky, "C3", lw=1, label="Monte Carlo (raw, delta pulse)")
    ax[0].loglog(x_m, ss_sky, "k:", lw=1.4, label="analytic single scattering")
    ax[0].set_ylabel("radiance per bin (a.u.)"), ax[0].legend(fontsize=8)
    ax[0].set_title(f"Fog-only pixels: backscatter vs single-scattering model "
                    f"($\\sigma_t$={sigma_t}, albedo={albedo}, g={g})", fontsize=10)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[1].semilogx(x_m, convolve_pulse(mc_sky) / convolve_pulse(ss_sky), "C3", lw=1)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_ylim(0, 3), ax[1].set_ylabel("MC / single scatter (pulsed)")
    ax[1].set_xlabel("one-way range (m)")
    fig.tight_layout()
    fig.savefig(FIG / "backscatter_vs_single_scatter.png", dpi=110)

    res["render_meta"] = {"clear": mc, "fog": mf}
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "render_meta"}, indent=2))


if __name__ == "__main__":
    main()
