"""
Etosha National Park — Risk Computation & Protectedness Model
IMMC International Round

Consumes preprocessed data from preprocess.py and computes per-cell
fire risk, poaching risk, combined risk, and protectedness scores.

Formulas:
  Fire risk (climate-driven):
    r_f = (f_veg) / (w1 * f_precip) * (1 - sech(w2 * f_wind)) * exp(w3 * f_temp)

  Poaching risk (exponential accessibility):
    r_p = exp(-a1 * d_road - a2 * d_waterhole + a3 * d_ranger)

  Combined risk = sqrt(r_f * r_p)   (geometric mean)
  Protectedness = allocated_resources / total_risk, normalized via x/(x+c)
"""

import numpy as np
import csv
from math import sqrt, exp, cosh

from preprocess import (
    lat_centers, lon_centers, n_rows, n_cols,
    CELL_AREA_KM2,
    min_dist, sat_norm, is_pan, compute_cell_ndvi,
    RANGER_STATIONS, WATERHOLES, ROAD_POINTS,
    preprocess,
)

# =============================================================================
# 1. FIRE RISK WEIGHTS
# =============================================================================

W_F_PRECIP = 1.0    # w1: precipitation suppression
W_F_WIND = 0.5      # w2: wind spread (sech activation)
W_F_TEMP = 0.05     # w3: temperature diffusion (exponential)


# =============================================================================
# 2. RISK FUNCTIONS
# =============================================================================

def sech(x):
    """Hyperbolic secant: 1/cosh(x)."""
    return 1.0 / cosh(min(x, 50))  # clamp to avoid overflow

def compute_raw_fire_risk(fire_count, time_years, veg_density,
                          temp_C=None, precip_mm=None, wind_ms=None):
    """
    Climate-driven fire risk, modulated by observed fire history:

    r_f = climate_susceptibility × (1 + β × fire_density)

    climate_susceptibility = (f_veg / (w1*f_precip)) * (1 - sech(w2*f_wind)) * exp(w3*f_temp)
    fire_density = fire_count / (area × time)

    When climate data unavailable, falls back to original formula.
    """
    fire_density = fire_count / (CELL_AREA_KM2 * time_years)
    BETA = 10.0  # fire history amplification factor

    if temp_C is not None and precip_mm is not None and wind_ms is not None:
        # Climate-based susceptibility
        precip_factor = 1.0 / (W_F_PRECIP * max(precip_mm, 0.01))
        wind_factor = 1.0 - sech(W_F_WIND * wind_ms)
        temp_factor = exp(W_F_TEMP * temp_C)

        climate_susc = veg_density * precip_factor * wind_factor * temp_factor

        # Combine: baseline climate risk amplified by empirical fire evidence
        return climate_susc * (1.0 + BETA * fire_density)

    # Fallback: purely empirical
    return fire_density * veg_density


def compute_raw_poaching_risk(lat, lon):
    """
    Exponential poaching risk:

    r_p = exp(-a1 * d_road - a2 * d_waterhole + a3 * d_ranger)

    Three distinct mechanisms:
      d_road:      accessibility — closer to roads = easier poacher movement
      d_waterhole: target value — rhinos concentrate at waterholes daily
      d_ranger:    deterrence — farther from rangers = less patrol coverage

    Signs: negative on d_road, d_waterhole (closer = higher risk)
           positive on d_ranger (farther = higher risk)

    Weights a1, a2, a3 set characteristic decay distances:
      a = 1/L where L is the distance at which the factor decays by ~63%.
    """
    # Characteristic distances (km)
    L_ROAD = 10.0       # road influence decays over ~10 km
    L_WATERHOLE = 15.0  # waterhole influence decays over ~15 km
    L_RANGER = 20.0     # ranger deterrence decays over ~20 km

    a1 = 1.0 / L_ROAD
    a2 = 1.0 / L_WATERHOLE
    a3 = 1.0 / L_RANGER

    d_road = min_dist(lat, lon, ROAD_POINTS)
    d_waterhole = min_dist(lat, lon, WATERHOLES)
    d_ranger = min_dist(lat, lon, RANGER_STATIONS)

    return exp(-a1 * d_road - a2 * d_waterhole + a3 * d_ranger)


# =============================================================================
# 3. RESPONSE CAPACITY
# =============================================================================

def compute_response_capacity(lat, lon):
    """
    Response capacity = 1 / (effective_distance_to_ranger)
    Terrain penalty: pan cells 3× harder to traverse.
    """
    d_ranger = min_dist(lat, lon, RANGER_STATIONS)
    terrain = 3.0 if is_pan(lat, lon) else 1.0
    effective = d_ranger * terrain
    return 1.0 / (effective + 0.5)


# =============================================================================
# 4. MODEL RUNNER
# =============================================================================

def run_model(data):
    """
    Compute per-cell risks, normalize, calculate protectedness,
    save CSV, and print summary.
    """
    counts = data['counts']
    T = data['T']
    get_ndvi = data['get_ndvi']
    use_ndvi = data['use_ndvi']
    climate = data['climate']

    print("\nComputing per-cell risk values...")
    raw_f_list = []
    raw_p_list = []
    cells = []

    ndvi_values = []
    ndvi_missing = 0

    for i, lat in enumerate(lat_centers):
        for j, lon in enumerate(lon_centers):
            cell = (i, j)
            fc = counts.get(cell, 0)

            # Vegetation density
            if use_ndvi:
                ndvi_val = compute_cell_ndvi(get_ndvi, lat, lon)
                if ndvi_val is not None:
                    veg = ndvi_val
                    ndvi_values.append(ndvi_val)
                else:
                    # Fallback for cells outside NDVI raster
                    veg = 0.05 if is_pan(lat, lon) else 0.3
                    ndvi_missing += 1
            else:
                # No NDVI data: use pan approximation
                veg = 0.05 if is_pan(lat, lon) else 1.0
                ndvi_val = veg

            # Climate data for fire risk
            temp_C, precip_mm, wind_ms = (None, None, None)
            if climate.available:
                temp_C, precip_mm, wind_ms = climate.get(lat, lon)

            rf = compute_raw_fire_risk(fc, T, veg, temp_C, precip_mm, wind_ms)
            rp = compute_raw_poaching_risk(lat, lon)

            raw_f_list.append(rf)
            raw_p_list.append(rp)

            cells.append({
                'row': i, 'col': j,
                'lat': round(lat, 4), 'lon': round(lon, 4),
                'fire_count': round(fc, 2),
                'ndvi': round(ndvi_val if ndvi_val else 0, 4),
                'raw_f': rf, 'raw_p': rp,
                'in_pan': is_pan(lat, lon),
            })

    if use_ndvi:
        print(f"  NDVI stats: mean={np.mean(ndvi_values):.4f}, "
              f"min={np.min(ndvi_values):.4f}, max={np.max(ndvi_values):.4f}")
        if ndvi_missing > 0:
            print(f"  {ndvi_missing} cells had no NDVI coverage (used fallback)")

    # --- Normalize with x/(x+c), c = median of nonzero ---
    nz_f = [v for v in raw_f_list if v > 0]
    nz_p = [v for v in raw_p_list if v > 0]
    c_f = float(np.median(nz_f)) if nz_f else 1.0
    c_p = float(np.median(nz_p)) if nz_p else 1.0
    print(f"\nNormalization constants: c_fire={c_f:.6f}, c_poach={c_p:.6f}")

    for d in cells:
        d['r_f'] = round(sat_norm(d['raw_f'], c_f), 4)
        d['r_p'] = round(sat_norm(d['raw_p'], c_p), 4)

        # Combined risk: geometric mean
        d['risk'] = round(sqrt(d['r_f'] * d['r_p']), 4)

        # Response capacity
        d['response'] = round(compute_response_capacity(d['lat'], d['lon']), 6)

    # --- Protectedness = resources / risk ---
    # Uniform allocation for Part 1; optimization in Part 2
    TOTAL_RANGERS = 30
    n = len(cells)
    uniform_alloc = TOTAL_RANGERS / n

    for d in cells:
        if d['risk'] > 0:
            d['protectedness'] = round(uniform_alloc / d['risk'], 6)
        else:
            d['protectedness'] = 999.0  # sentinel: no risk

    # Normalize protectedness for display [0,1]
    finite = [d['protectedness'] for d in cells if d['protectedness'] < 999]
    pmax = max(finite) if finite else 1.0
    for d in cells:
        d['P_norm'] = round(min(d['protectedness'] / pmax, 1.0), 4)

    # --- Save results ---
    out = "etosha_protectedness_grid.csv"
    fields = ['row', 'col', 'lat', 'lon', 'fire_count', 'ndvi',
              'r_f', 'r_p', 'risk', 'response', 'protectedness', 'P_norm', 'in_pan']

    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(cells)
    print(f"\n✓ Saved {len(cells)} cells to {out}")

    # --- Summary Statistics ---
    risks = [d['risk'] for d in cells]
    r_fs = [d['r_f'] for d in cells]
    r_ps = [d['r_p'] for d in cells]

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Fire risk (r_f):     [{min(r_fs):.4f}, {max(r_fs):.4f}], mean={np.mean(r_fs):.4f}")
    print(f"Poaching risk (r_p): [{min(r_ps):.4f}, {max(r_ps):.4f}], mean={np.mean(r_ps):.4f}")
    print(f"Combined risk:       [{min(risks):.4f}, {max(risks):.4f}], mean={np.mean(risks):.4f}")

    # Cells with fire detections
    fire_cells = sum(1 for d in cells if d['fire_count'] > 0)
    print(f"\nCells with fire detections: {fire_cells}/{len(cells)} ({100*fire_cells/len(cells):.1f}%)")

    # Top 10 highest risk
    top = sorted(cells, key=lambda d: d['risk'], reverse=True)
    print(f"\nTop 10 highest risk cells:")
    print(f"  {'Lat':>8} {'Lon':>8} {'risk':>6} {'r_f':>6} {'r_p':>6} {'NDVI':>6} {'fires':>6}")
    for c in top[:10]:
        print(f"  {c['lat']:8.4f} {c['lon']:8.4f} {c['risk']:6.4f} "
              f"{c['r_f']:6.4f} {c['r_p']:6.4f} {c['ndvi']:6.4f} {c['fire_count']:6.1f}")

    # Top 10 most vulnerable
    vuln = sorted(cells, key=lambda d: d['protectedness'])
    print(f"\nTop 10 most vulnerable (lowest protectedness):")
    print(f"  {'Lat':>8} {'Lon':>8} {'P':>10} {'risk':>6} {'response':>8}")
    for c in vuln[:10]:
        print(f"  {c['lat']:8.4f} {c['lon']:8.4f} {c['protectedness']:10.4f} "
              f"{c['risk']:6.4f} {c['response']:8.6f}")

    # Pan vs non-pan comparison
    pan_cells = [d for d in cells if d['in_pan']]
    nonpan_cells = [d for d in cells if not d['in_pan']]
    if pan_cells and nonpan_cells:
        print(f"\nPan vs Non-Pan:")
        print(f"  Pan cells: {len(pan_cells)}, mean risk={np.mean([d['risk'] for d in pan_cells]):.4f}")
        print(f"  Non-pan:   {len(nonpan_cells)}, mean risk={np.mean([d['risk'] for d in nonpan_cells]):.4f}")


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    data = preprocess()
    run_model(data)


if __name__ == "__main__":
    main()
