"""Small helper module shared by the C1 practical notebooks.

It fetches the data files (from ../data when run inside the repository, otherwise from GitHub),
draws global and regional maps with pyslfp and cartopy, and provides a few numerical utilities. Nothing here is
examinable; it exists so that the notebook cells stay short.
"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE = os.environ.get("C1PRAC_BASE", "https://raw.githubusercontent.com/da380/c1-practicals/main/")
_HERE = pathlib.Path(__file__).resolve().parent


def get(name: str) -> str:
    """Return a local path to a data file, downloading it if necessary."""
    for cand in (_HERE / name, pathlib.Path("../data") / name, pathlib.Path("data") / name):
        if cand.exists():
            return str(cand)
    d = pathlib.Path("data")
    d.mkdir(exist_ok=True)
    p = d / name
    urllib.request.urlretrieve(BASE + "data/" + name, p)
    return str(p)


def load_json(name: str):
    with open(get(name)) as f:
        return json.load(f)


plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "figure.dpi": 110, "figure.figsize": (10, 6), "axes.spines.top": False, "axes.spines.right": False})

def _wrap_grid(lon, field):
    """Rearrange a grid with longitudes in [0, 360] to longitudes in [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    field = np.asarray(field)
    if lon.max() > 180.0:
        if np.isclose(lon[-1], 360.0):
            lon, field = lon[:-1], field[..., :-1]
        lon = np.where(lon >= 180.0, lon - 360.0, lon)
        order = np.argsort(lon)
        lon, field = lon[order], field[..., order]
    return lon, field


def _as_shgrid(field, lon, lat):
    """Wrap a (lat, lon) array on a pyshtools Driscoll-Healy grid as an SHGrid, or return None."""
    import pyshtools as pysh
    f = np.asarray(field, dtype=float)
    nlat, nlon = f.shape
    if lon[0] != 0.0 or lat[0] != 90.0 or nlon not in (nlat, 2 * nlat - 1):
        return None
    return pysh.SHGrid.from_array(f, grid="DH")


def graticule(ax, step=30, fontsize=8):
    """Lines of latitude and longitude every `step` degrees on a global cartopy map, with latitude labels
    outside the left edge and longitude labels below the bottom edge (cartopy's own labels upset the
    figure layout on a Robinson map)."""
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.5, color="0.35", alpha=0.7,
                 xlocs=range(-180, 181, step), ylocs=range(-90 + step, 90, step))
    try:
        c0 = ax.projection.proj4_params.get("lon_0", 0.0)
    except AttributeError:
        c0 = 0.0
    for la in range(-90 + step, 90, step):
        x, y = ax.projection.transform_point(c0 - 180 + 1e-6, la, pc)
        ax.annotate(f"{abs(la)}°{'N' if la > 0 else 'S' if la < 0 else ''}", (x, y), xytext=(-3, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=fontsize, annotation_clip=False)
    for lo in range(-180 + step, 180, step):
        x, _ = ax.projection.transform_point(c0 + lo, -89.5, pc)     # the pole itself does not transform
        y = ax.projection.y_limits[0]
        ax.annotate(f"{abs(lo)}°{'E' if lo > 0 else 'W' if lo < 0 else ''}", (x, y), xytext=(0, -3), textcoords="offset points",
                    ha="center", va="top", fontsize=fontsize, annotation_clip=False)


def global_map(field, lon, lat, *, ax=None, vmin=None, vmax=None, center=None, cmap="RdBu_r", label="", title="",
               contours=(), ocean=None, sites=None, figsize=(12, 6.5), colorbar=True, grid=True):
    """Filled map of a field on a Robinson projection, drawn with pyslfp and cartopy. Longitudes may run 0-360.

    field: 2-D array (lat, lon). ocean: optional mask (1 over ocean) that hides land. contours: list
    of levels drawn as black lines (the first solid, the second dashed). sites: dict name -> (lat, lon).
    center: if given, use a two-slope colour scale centred there (for fingerprints).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import pyslfp
    f = np.where(np.asarray(ocean) > 0, field, np.nan) if ocean is not None else np.asarray(field, dtype=float)
    if ax is None:
        fig, ax = pyslfp.create_map_figure(figsize=figsize)
    else:
        fig = ax.figure
    norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax) if center is not None else None
    scale = {"norm": norm} if norm else {"vmin": vmin, "vmax": vmax}
    styles = {"linestyles": ["-", "--"][: len(contours)], "linewidths": [1.2, 0.8][: len(contours)]}
    g = _as_shgrid(f, lon, lat)
    if g is not None:
        _, pm = pyslfp.plot(g, ax=ax, cmap=cmap, gridlines=False, colorbar=False, contour_lines=bool(contours),
                            levels=list(contours), contour_lines_kwargs=styles, rasterized=True, **scale)
    else:
        pm = ax.pcolormesh(lon, lat, f, transform=ccrs.PlateCarree(), cmap=cmap, rasterized=True, **scale)
        if contours:
            ax.contour(lon, lat, f, transform=ccrs.PlateCarree(), levels=list(contours), colors="k", **styles)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.set_global()
    if grid:
        graticule(ax)
    if sites:
        for name, (la, lo) in sites.items():
            ax.plot(lo, la, "k^", ms=6, mfc="yellow", transform=ccrs.PlateCarree())
            ax.annotate(name, ax.projection.transform_point(lo, la, ccrs.PlateCarree()), xytext=(4, 4), textcoords="offset points", fontsize=11)
    if colorbar:
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", shrink=0.6, pad=0.06)
        cb.set_label(label)
    if title:
        ax.set_title(title, fontsize=14)
    return fig, ax


def regional_map(field, lon, lat, *, lon_range, lat_range, ax=None, vmin=None, vmax=None, cmap="RdBu_r", label="", title="",
                 contours=(), ocean=None, sites=None, figsize=(9, 6.5)):
    """Filled map of a region in a plate carree projection with cartopy coastlines."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    f = np.where(np.asarray(ocean) > 0, field, np.nan) if ocean is not None else np.asarray(field, dtype=float)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    else:
        fig = ax.figure
    pm = ax.pcolormesh(lon, lat, f, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    if contours:
        ax.contour(lon, lat, f, transform=ccrs.PlateCarree(), levels=list(contours), colors="k",
                   linestyles=["-", "--"][: len(contours)], linewidths=[1.2, 0.8][: len(contours)])
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    if sites:
        for name, (la, lo) in sites.items():
            ax.plot(lo, la, "k^", ms=6, mfc="yellow", transform=ccrs.PlateCarree())
            ax.annotate(name, (lo, la), xytext=(4, 4), textcoords="offset points", fontsize=11)
    ax.set_extent([*lon_range, *lat_range], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, linestyle="--", alpha=0.4)   # cartopy's own labels break the inline figure bounding box
    xt = [v for v in range(-180, 181, 10) if lon_range[0] < v < lon_range[1]]
    yt = [v for v in range(-80, 81, 10) if lat_range[0] < v < lat_range[1]]
    ax.set_xticks(xt, crs=ccrs.PlateCarree()); ax.set_yticks(yt, crs=ccrs.PlateCarree())
    ax.set_xticklabels([f"{abs(v)}°{'W' if v < 0 else 'E' if v > 0 else ''}" for v in xt])
    ax.set_yticklabels([f"{abs(v)}°{'S' if v < 0 else 'N' if v > 0 else ''}" for v in yt])
    fig.colorbar(pm, ax=ax, orientation="horizontal", shrink=0.9, pad=0.08, label=label)
    if title:
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=13)
    return fig, ax


_STATE = {}


def earth_state(lmax: int = 128):
    """A pyslfp EarthState (present-day, PREM Love numbers), cached so that later cells reuse it."""
    from pyslfp import EarthState
    if lmax not in _STATE:
        _STATE[lmax] = EarthState.from_defaults(lmax=lmax)
    return _STATE[lmax]


def fingerprint(state, load, *, rotation: bool = True) -> dict:
    """Solve the sea level equation for a load and return the fields divided by the bathtub value.

    Returns lon, lat and the ocean mask, plus 'sl' (relative sea level change), 'uplift' (vertical
    displacement of the solid surface) and 'geoid' (geoid height change), all normalised so that the
    ocean mean of 'sl' is one, and 'mean', the bathtub value itself (in metres for the load given).
    """
    from pyslfp import LinearSeaLevelEquation
    sl, u, phi, _ = LinearSeaLevelEquation(state).solve_sea_level_equation(load, rotational_feedbacks=rotation)
    lon, lat, ocean = state.lons(), state.lats(), state.ocean_projection(value=0.0).data
    mean = ocean_mean(sl.data, lon, lat, ocean)
    g = state.model.parameters.gravitational_acceleration
    return {"lon": lon, "lat": lat, "ocean": ocean, "mean": mean, "sl": sl.data / mean,
            "uplift": u.data / mean, "geoid": -phi.data / (g * mean)}


def point_value(field, lon, lat, plat, plon):
    """Bilinear interpolation of a (lat, lon) grid at a point. Longitudes may run 0-360 or -180-180."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    field = np.asarray(field, dtype=float)
    if lon.max() > 180.0:
        plon = plon % 360.0
    # latitude axis may run north to south
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[::-1, :]
    i = np.clip(np.searchsorted(lat, plat) - 1, 0, lat.size - 2)
    j = np.clip(np.searchsorted(lon, plon) - 1, 0, lon.size - 2)
    ty = (plat - lat[i]) / (lat[i + 1] - lat[i])
    tx = (plon - lon[j]) / (lon[j + 1] - lon[j])
    return ((1 - ty) * (1 - tx) * field[i, j] + (1 - ty) * tx * field[i, j + 1] + ty * (1 - tx) * field[i + 1, j] + ty * tx * field[i + 1, j + 1])


def ocean_mean(field, lon, lat, ocean):
    """Area-weighted mean over the ocean of a (lat, lon) grid."""
    w = np.cos(np.radians(lat))[:, None] * (np.asarray(ocean) > 0)
    return float(np.nansum(field * w) / np.sum(w))


def weighted_least_squares(A, d, sigma):
    """Solve min sum_i ((d_i - (A m)_i)/sigma_i)^2. Returns the estimate, its covariance, and residuals."""
    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float)
    W = 1.0 / np.asarray(sigma, dtype=float) ** 2
    N = A.T @ (W[:, None] * A)
    cov = np.linalg.inv(N)
    m = cov @ (A.T @ (W * d))
    return m, cov, d - A @ m


def synthetic_record(t, rsl, *, sigma_t=0.15, sigma_h=3.0, seed=1):
    """Perturb a modelled record with errors of the kind real records carry: sigma_t (ka) in age and
    sigma_h (m) in height, Gaussian, fixed seed. The present-day point (age 0) is left exact."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    y = np.asarray(rsl, dtype=float)
    past = t > 0
    t_obs = np.clip(t + np.where(past, rng.normal(0.0, sigma_t, t.size), 0.0), 0.0, None)
    y_obs = y + np.where(past, rng.normal(0.0, sigma_h, t.size), 0.0)
    return t_obs, y_obs


def fit_amplitude(t, rsl, tau, *, sigma_t=0.15, sigma_h=3.0):
    """Fit A in RSL = A (exp(t/tau) - 1) for a fixed decay time tau (offset B = 0).

    Returns A, the rms misfit, chi-squared and the effective error of each point. The timing error is
    converted to an equivalent height error through the slope of the curve, and the fit is weighted.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(rsl, dtype=float)
    g = np.exp(t / tau) - 1.0
    A = float(g @ y / (g @ g))
    slope = (A / tau) * np.exp(t / tau)
    sig = np.sqrt(sigma_h**2 + (slope * sigma_t) ** 2)
    w = 1.0 / sig**2
    A = float((w * g) @ y / ((w * g) @ g))
    r = y - A * g
    return A, float(np.sqrt(np.mean(r**2))), float(np.sum((r / sig) ** 2)), sig


def fit_decay(t, rsl, B, taus=None):
    """Fit RSL(t) = A (exp(t/tau) - 1) + B, with t the age, for fixed B: scan tau and solve for A.

    Returns tau, A, the tau values scanned and the rms misfit for each. The remaining uplift at the
    present day decays as exp(-time/tau), so as a function of age it grows as exp(age/tau).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(rsl, dtype=float) - B
    if taus is None:
        taus = np.linspace(0.5, 20.0, 391)
    best = None
    mis = []
    for tau in taus:
        g = np.exp(t / tau) - 1.0
        A = float(g @ y / (g @ g)) if g @ g > 0 else 0.0
        r = float(np.sqrt(np.mean((y - A * g) ** 2)))
        mis.append(r)
        if best is None or r < best[2]:
            best = (tau, A, r)
    return best[0], best[1], np.array(taus), np.array(mis)
