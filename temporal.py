"""
Etosha National Park — Temporal Analysis & Staffing Requirements
IMMC International Round — Requirement 3

Runs the protection model year-by-year (2015–2024) using:
  - Annual September NDVI (vegetation/fuel load proxy)
  - FIRMS fire data filtered to each year
  - Fixed climate averages (slow-varying)

Then sweeps ranger staffing levels to find the minimum
personnel required to achieve target protection thresholds.

Usage:
    conda install numpy rasterio
    python temporal.py
"""

import numpy as np
import csv
import os
from math import sqrt, exp, cosh
from collections import defaultdict
from datetime import datetime

from preprocess import (
    lat_centers, lon_centers, n_rows, n_cols,
    CELL_AREA_KM2, CELL_SIZE_LAT, CELL_SIZE_LON,
    min_dist, sat_norm, is_pan, compute_cell_ndvi,
    RANGER_STATIONS, WATERHOLES, ROAD_POINTS,
    load_firms, bin_fires, load_ndvi_multi,
    ClimateData,
)

from model import (
    W_F_PRECIP, W_F_WIND, W_F_TEMP, BETA_FIRE,
    L_ROAD, L_WATERHOLE, L_RANGER,
    E, RESOURCE_NAMES, N_RESOURCES, BUDGETS, ALPHA,
    sech, compute_raw_fire_risk, compute_raw_poaching_risk,
    compute_demand_vector, compute_coverage, compute_protection,
    optimize_allocation,
)


# =============================================================================
# 1. SEPTEMBER NDVI FILE MAPPING (year → filename)
# =============================================================================

SEPT_NDVI_FILES = {
    2015: "MOD13A2_NDVI_16_Days_Sept_14_2015.tif",
    2016: "MOD13A2_NDVI_16_Days_Sept_13_2016.tif",
    2017: "MOD13A2_NDVI_16_Days_Sept_30_2017.tif",
    2018: "MOD13A2_NDVI_16_Days_Sept_14_2018.tif",
    2019: "MOD13A2_NDVI_16_Days_Sept_14.tif",   # assumed 2019
    2020: "MOD13A2_NDVI_16_Days_Sept_2020.tif",
    2021: "MOD13A2_NDVI_16_Days_Sept_14_2021.tif",
    2022: "NDVI_16_Days_2022-09-14.tif",
    2024: "MOD13A2_NDVI_16_Days_Sept_2024.tif",
}


# =============================================================================
# 2. YEAR-SPECIFIC DATA LOADING
# =============================================================================

def filter_firms_by_year(all_fires, year):
    """Filter FIRMS fire detections to a single calendar year."""
    return [f for f in all_fires if f['date'].startswith(str(year))]


def load_climate_once():
    """
    Load NASA POWER climate data once (averaged across all years).
    Climate varies slowly; temporal signal comes from NDVI and fires.
    """
    def find_power_csv(keyword):
        for d in ['.', './data', '../data']:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith('.csv') and 'POWER' in f.upper():
                    path = os.path.join(d, f)
                    try:
                        with open(path, 'r') as fh:
                            head = fh.read(500)
                            if keyword in head:
                                return path
                    except:
                        pass
        return None

    temp_path = find_power_csv('T2M')
    precip_path = find_power_csv('PRECTOTCORR')
    wind_path = find_power_csv('WS2M')

    climate = ClimateData(temp_path, precip_path, wind_path)
    return climate


# =============================================================================
# 3. CORE MODEL RUN (single year, given budgets)
# =============================================================================

def run_single_year(year_fires, get_ndvi, use_ndvi, climate, budgets,
                    alpha=ALPHA, quiet=False):
    """
    Run the full risk + optimization pipeline for one configuration.

    Args:
        year_fires: list of fire dicts (already filtered)
        get_ndvi: NDVI lookup function (or None)
        use_ndvi: bool
        climate: ClimateData instance
        budgets: np.array([drones, rangers])
        alpha: resource efficiency parameter
        quiet: suppress print output

    Returns:
        dict with summary stats:
            mean_P, total_P, mean_R, n_active,
            prot_quartiles, cells (full list if needed)
    """
    # Bin fires
    counts, frps = bin_fires(year_fires)

    # Time span for fire density: 1 year (single-year slice)
    T = 1.0

    # Compute raw risks
    cells = []
    raw_f_list = []
    raw_p_list = []

    for i, lat in enumerate(lat_centers):
        for j, lon in enumerate(lon_centers):
            cell = (i, j)
            fc = counts.get(cell, 0)

            # Vegetation
            if use_ndvi:
                ndvi_val = compute_cell_ndvi(get_ndvi, lat, lon)
                veg = ndvi_val if ndvi_val is not None else (0.05 if is_pan(lat, lon) else 0.3)
            else:
                veg = 0.05 if is_pan(lat, lon) else 1.0

            # Climate
            temp_C, precip_mm, wind_ms = (None, None, None)
            if climate.available:
                temp_C, precip_mm, wind_ms = climate.get(lat, lon)

            rf = compute_raw_fire_risk(fc, T, veg, temp_C, precip_mm, wind_ms)
            rp = compute_raw_poaching_risk(lat, lon)

            raw_f_list.append(rf)
            raw_p_list.append(rp)
            cells.append({
                'row': i, 'col': j, 'lat': lat, 'lon': lon,
                'raw_f': rf, 'raw_p': rp,
                'in_pan': is_pan(lat, lon),
            })

    # Normalize
    nz_f = [v for v in raw_f_list if v > 0]
    nz_p = [v for v in raw_p_list if v > 0]
    c_f = float(np.median(nz_f)) if nz_f else 1.0
    c_p = float(np.median(nz_p)) if nz_p else 1.0

    for d in cells:
        d['r_f'] = sat_norm(d['raw_f'], c_f)
        d['r_p'] = sat_norm(d['raw_p'], c_p)
        d['risk'] = sqrt(d['r_f'] * d['r_p'])
        d['lambda'] = compute_demand_vector(d['r_f'], d['r_p'])

    # Run optimization with given budgets
    # Temporarily override BUDGETS for the optimizer
    import model as model_mod
    orig_budgets = model_mod.BUDGETS.copy()
    model_mod.BUDGETS = budgets.copy()

    # Suppress output if quiet
    if quiet:
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        c_opt, total_P = optimize_allocation(cells, alpha=alpha)
    finally:
        if quiet:
            sys.stdout = old_stdout
        model_mod.BUDGETS = orig_budgets

    # Compute per-cell protection
    n_active = 0
    protections = []
    risks = []
    for k, d in enumerate(cells):
        c_vec = c_opt[:, k]
        C_k = np.dot(d['lambda'], c_vec)
        P_k = compute_protection(C_k, d['risk'], alpha)
        d['protection'] = P_k
        d['coverage'] = C_k
        for n in range(N_RESOURCES):
            d[f'c_{RESOURCE_NAMES[n].lower()}'] = c_vec[n]

        if d['risk'] > 1e-8:
            n_active += 1
            protections.append(P_k)
            risks.append(d['risk'])

    mean_P = np.mean(protections) if protections else 0
    mean_R = np.mean(risks) if risks else 0

    # Protection by risk quartile
    active_sorted = sorted(
        [d for d in cells if d['risk'] > 1e-8],
        key=lambda d: d['risk']
    )
    Q = len(active_sorted) // 4
    quartile_means = {}
    if Q > 0:
        labels = ['Q1_low', 'Q2_medlow', 'Q3_medhigh', 'Q4_high']
        slices = [active_sorted[:Q], active_sorted[Q:2*Q],
                  active_sorted[2*Q:3*Q], active_sorted[3*Q:]]
        for lbl, sl in zip(labels, slices):
            quartile_means[lbl] = np.mean([d['protection'] for d in sl])

    # Minimum protection among active cells
    min_P = min(protections) if protections else 0
    # Fraction of cells below thresholds
    frac_below_30 = sum(1 for p in protections if p < 0.3) / max(len(protections), 1)
    frac_below_50 = sum(1 for p in protections if p < 0.5) / max(len(protections), 1)

    return {
        'mean_P': mean_P,
        'total_P': total_P,
        'mean_R': mean_R,
        'min_P': min_P,
        'n_active': n_active,
        'frac_below_30': frac_below_30,
        'frac_below_50': frac_below_50,
        'quartile_means': quartile_means,
        'n_fires': len(year_fires),
        'cells': cells,
    }


# =============================================================================
# 4. TEMPORAL ANALYSIS: Protection over years
# =============================================================================

def temporal_analysis(all_fires, climate):
    """
    Run the model for each year with annual September NDVI.
    Returns a list of {year, ...summary stats...}.
    """
    print("\n" + "=" * 60)
    print("TEMPORAL ANALYSIS: Protection over time (2015–2024)")
    print("=" * 60)

    results = []
    years = sorted(SEPT_NDVI_FILES.keys())

    for year in years:
        ndvi_file = SEPT_NDVI_FILES[year]
        if not os.path.isfile(ndvi_file):
            print(f"\n  {year}: NDVI file '{ndvi_file}' not found — skipping")
            continue

        print(f"\n--- {year} ---")

        # Load this year's NDVI
        _, get_ndvi = load_ndvi_multi([ndvi_file], label=f"Sept {year}")
        use_ndvi = get_ndvi is not None

        # Filter fires to this year
        year_fires = filter_firms_by_year(all_fires, year)
        print(f"  Fire detections in {year}: {len(year_fires)}")

        # Run model with default budgets
        stats = run_single_year(
            year_fires, get_ndvi, use_ndvi, climate,
            budgets=BUDGETS, quiet=True
        )
        stats['year'] = year
        results.append(stats)

        print(f"  Mean risk:       {stats['mean_R']:.4f}")
        print(f"  Mean protection: {stats['mean_P']:.4f}")
        print(f"  Min protection:  {stats['min_P']:.4f}")
        print(f"  Active cells:    {stats['n_active']}")
        if stats['quartile_means']:
            qm = stats['quartile_means']
            print(f"  Quartile P:  Q1={qm.get('Q1_low',0):.3f}  "
                  f"Q2={qm.get('Q2_medlow',0):.3f}  "
                  f"Q3={qm.get('Q3_medhigh',0):.3f}  "
                  f"Q4={qm.get('Q4_high',0):.3f}")

    return results


# =============================================================================
# 5. STAFFING SWEEP: Minimum rangers for target protection
# =============================================================================

def staffing_sweep(all_fires, climate, target_year=2024,
                   drone_budget=20, sensor_budget=50, ranger_range=None):
    """
    For a fixed year and drone/sensor count, sweep ranger staffing levels
    and find the minimum needed for various protection thresholds.

    Returns list of {rangers, mean_P, min_P, frac_below_30, ...}
    """
    if ranger_range is None:
        ranger_range = [50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 700, 1000]

    print(f"\n{'='*60}")
    print(f"STAFFING SWEEP: Year {target_year}, {drone_budget} drones, {sensor_budget} sensors")
    print(f"{'='*60}")

    # Load NDVI for target year
    ndvi_file = SEPT_NDVI_FILES.get(target_year)
    if ndvi_file and os.path.isfile(ndvi_file):
        _, get_ndvi = load_ndvi_multi([ndvi_file], label=f"Sept {target_year}")
        use_ndvi = get_ndvi is not None
    else:
        # Fallback to most recent available
        print(f"  Warning: no NDVI for {target_year}, using fallback")
        get_ndvi = None
        use_ndvi = False

    year_fires = filter_firms_by_year(all_fires, target_year)
    print(f"  Fire detections in {target_year}: {len(year_fires)}")

    results = []
    print(f"\n  {'Rangers':>8} {'Mean P':>8} {'Min P':>8} "
          f"{'<0.3':>6} {'<0.5':>6} {'Q4 P':>6}")
    print(f"  {'-'*50}")

    for n_rangers in ranger_range:
        budgets = np.array([float(drone_budget), float(n_rangers), float(sensor_budget)])
        stats = run_single_year(
            year_fires, get_ndvi, use_ndvi, climate,
            budgets=budgets, quiet=True
        )
        stats['rangers'] = n_rangers
        stats['drones'] = drone_budget
        results.append(stats)

        q4 = stats['quartile_means'].get('Q4_high', 0)
        print(f"  {n_rangers:>8} {stats['mean_P']:>8.4f} {stats['min_P']:>8.4f} "
              f"{stats['frac_below_30']:>6.1%} {stats['frac_below_50']:>6.1%} "
              f"{q4:>6.3f}")

    # Find minimum rangers for thresholds
    print(f"\n  Staffing recommendations:")
    for threshold, label in [(0.3, "Mean P ≥ 0.30"),
                              (0.4, "Mean P ≥ 0.40"),
                              (0.5, "Mean P ≥ 0.50"),
                              (0.6, "Mean P ≥ 0.60")]:
        sufficient = [r for r in results if r['mean_P'] >= threshold]
        if sufficient:
            min_r = min(r['rangers'] for r in sufficient)
            print(f"    {label}: minimum ~{min_r} rangers")
        else:
            print(f"    {label}: not achieved with ≤{ranger_range[-1]} rangers")

    return results


# =============================================================================
# 6. SAVE RESULTS
# =============================================================================

def save_temporal_csv(results, filename="temporal_protection.csv"):
    """Save year-by-year results to CSV."""
    fields = ['year', 'n_fires', 'mean_R', 'mean_P', 'min_P',
              'n_active', 'frac_below_30', 'frac_below_50',
              'Q1_low', 'Q2_medlow', 'Q3_medhigh', 'Q4_high']
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = {k: round(v, 6) if isinstance(v, float) else v
                   for k, v in r.items() if k in fields}
            # Flatten quartiles
            for qk, qv in r.get('quartile_means', {}).items():
                row[qk] = round(qv, 6)
            w.writerow(row)
    print(f"\n✓ Saved temporal results to {filename}")


def save_staffing_csv(results, filename="staffing_sweep.csv"):
    """Save staffing sweep results to CSV."""
    fields = ['drones', 'rangers', 'mean_P', 'min_P',
              'frac_below_30', 'frac_below_50',
              'Q1_low', 'Q2_medlow', 'Q3_medhigh', 'Q4_high']
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = {k: round(v, 6) if isinstance(v, float) else v
                   for k, v in r.items() if k in fields}
            for qk, qv in r.get('quartile_means', {}).items():
                row[qk] = round(qv, 6)
            w.writerow(row)
    print(f"✓ Saved staffing sweep to {filename}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ETOSHA — TEMPORAL ANALYSIS & STAFFING (Requirement 3)")
    print("=" * 60)
    print(f"Grid: {n_rows}×{n_cols} = {n_rows * n_cols} cells")
    print(f"Default budgets: {BUDGETS[0]:.0f} drones, {BUDGETS[1]:.0f} rangers")

    # Load all fires once
    all_fires = load_firms("FIRMS_dataset.csv")

    # Load climate once (averaged across years)
    climate = load_climate_once()

    # --- Part A: Year-by-year protection ---
    temporal_results = temporal_analysis(all_fires, climate)
    if temporal_results:
        save_temporal_csv(temporal_results)

        # Summary table
        print(f"\n{'='*60}")
        print("TEMPORAL SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Year':>6} {'Fires':>6} {'Mean R':>8} {'Mean P':>8} {'Min P':>8}")
        for r in temporal_results:
            print(f"  {r['year']:>6} {r['n_fires']:>6} "
                  f"{r['mean_R']:>8.4f} {r['mean_P']:>8.4f} {r['min_P']:>8.4f}")

    # --- Part B: Staffing sweep ---
    staffing_results = staffing_sweep(all_fires, climate, target_year=2024)
    if staffing_results:
        save_staffing_csv(staffing_results)

    # --- Part C: Staffing sweep for an earlier year (comparison) ---
    staffing_2015 = staffing_sweep(all_fires, climate, target_year=2015)
    if staffing_2015:
        save_staffing_csv(staffing_2015, "staffing_sweep_2015.csv")

    print(f"\n{'='*60}")
    print("DONE — Requirement 3 analysis complete")
    print(f"{'='*60}")
    print("Output files:")
    print("  temporal_protection.csv  — protection metrics by year")
    print("  staffing_sweep.csv       — ranger sweep for 2024")
    print("  staffing_sweep_2015.csv  — ranger sweep for 2015 (comparison)")


if __name__ == "__main__":
    main()