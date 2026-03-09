"""
Etosha National Park — Risk Computation & Protectedness Model
IMMC International Round

Consumes preprocessed data from preprocess.py and computes per-cell
fire risk, poaching risk, combined risk, and optimal resource allocation.

Risk Formulas:
  Fire risk (climate-driven):
    r_f = (f_veg / (w1 * f_precip)) * (1 - sech(w2 * f_wind)) * exp(w3 * f_temp)
    r_f *= (1 + β * fire_density)

  Poaching risk (exponential accessibility):
    r_p = exp(-a1 * d_road - a2 * d_waterhole + a3 * d_ranger)

  Combined risk R_k = sqrt(r̃_f * r̃_p)  where r̃ = x/(x+c) normalization

Protection Model (Koopman search theory):
  Demand vector:    λ(k) = E · r(k) / ‖E · r(k)‖
  Coverage:         C_k = λ(k) · c(k)
  Protection:       P_k = 1 - exp(-α · C_k / R_k)

Resources (3 types):
  Drones  — aerial thermal/optical; primary fire detection, some poaching
  Rangers — ground patrols; primary anti-poaching, some fire suppression
  Sensors — fixed camera traps + IR + acoustic; 24/7 poaching surveillance

Optimization:
  Maximize Σ_k P_k  subject to  Σ_k c_n(k) ≤ B_n  for each resource n
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
# 1. TUNABLE PARAMETERS
# =============================================================================

# --- Fire risk weights ---
W_F_PRECIP = 1.0    # w1: precipitation suppression
W_F_WIND   = 0.5    # w2: wind spread (sech activation)
W_F_TEMP   = 0.05   # w3: temperature diffusion (exponential)
BETA_FIRE  = 10.0   # fire history amplification factor

# --- Poaching risk characteristic distances (km) ---
L_ROAD       = 10.0   # road influence decays over ~10 km
L_WATERHOLE  = 15.0   # waterhole influence decays over ~15 km
L_RANGER     = 20.0   # ranger deterrence decays over ~20 km

# --- Resource model ---
# Effectiveness matrix E: rows = resources, cols = risk types [fire, poaching]
#
# Literature basis (see report for citations):
#   Drones:  thermal imaging is primary fire detection tool (0.8); aerial
#            surveillance aids poaching detection but cannot arrest (0.3)
#   Rangers: limited fire suppression & prescribed burns (0.2); only resource
#            that deters, arrests, and removes snares (0.9)
#   Sensors: fixed camera traps + IR + acoustic; poor spatial fire coverage
#            (0.15); strong 24/7 poaching surveillance at choke points (0.6)
#            — AI-powered detection is 17× faster than conventional methods
#            (Hwange NP acoustic study); WPS wpsWatch platform covers 3500+
#            connected traps globally with real-time alerts.
#
E = np.array([
    [0.80, 0.30],   # Drones:  strong fire detection, some poaching surveillance
    [0.20, 0.90],   # Rangers: some fire suppression, strong anti-poaching
    [0.15, 0.60],   # Sensors: weak fire detection, strong poaching surveillance
])
RESOURCE_NAMES = ["Drones", "Rangers", "Sensors"]
N_RESOURCES = len(RESOURCE_NAMES)

# Budgets: total units available for each resource
BUDGETS = np.array([20.0, 200.0, 50.0])   # 20 drones, 200 rangers, 50 sensor stations

# Resource efficiency (α in the exponential detection model)
ALPHA = 1.0


# =============================================================================
# 2. RISK FUNCTIONS
# =============================================================================

def sech(x):
    """Hyperbolic secant: 1/cosh(x)."""
    return 1.0 / cosh(min(x, 50))


def compute_raw_fire_risk(fire_count, time_years, veg_density,
                          temp_C=None, precip_mm=None, wind_ms=None):
    """
    Climate-driven fire risk, modulated by observed fire history.
    r_f = climate_susceptibility × (1 + β × fire_density)
    """
    fire_density = fire_count / (CELL_AREA_KM2 * time_years)

    if temp_C is not None and precip_mm is not None and wind_ms is not None:
        precip_factor = 1.0 / (W_F_PRECIP * max(precip_mm, 0.01))
        wind_factor = 1.0 - sech(W_F_WIND * wind_ms)
        temp_factor = exp(W_F_TEMP * temp_C)

        climate_susc = veg_density * precip_factor * wind_factor * temp_factor
        return climate_susc * (1.0 + BETA_FIRE * fire_density)

    # Fallback: purely empirical
    return fire_density * veg_density


def compute_raw_poaching_risk(lat, lon):
    """
    Exponential poaching risk:
    r_p = exp(-a1 * d_road - a2 * d_waterhole + a3 * d_ranger)
    """
    a1 = 1.0 / L_ROAD
    a2 = 1.0 / L_WATERHOLE
    a3 = 1.0 / L_RANGER

    d_road = min_dist(lat, lon, ROAD_POINTS)
    d_waterhole = min_dist(lat, lon, WATERHOLES)
    d_ranger = min_dist(lat, lon, RANGER_STATIONS)

    return exp(-a1 * d_road - a2 * d_waterhole + a3 * d_ranger)


# =============================================================================
# 3. DEMAND VECTOR & PROTECTION
# =============================================================================

def compute_demand_vector(r_f_norm, r_p_norm):
    """
    Compute normalized demand vector λ(k) = E · r(k) / ‖E · r(k)‖.

    This captures what *mix* of resources the cell needs, independent
    of risk magnitude. A fire-dominated cell gets high drone demand;
    a poaching-dominated cell gets high ranger demand.
    """
    r_vec = np.array([r_f_norm, r_p_norm])
    raw = E @ r_vec   # shape (N_RESOURCES,)

    norm = np.linalg.norm(raw)
    if norm > 0:
        return raw / norm
    else:
        # No risk → equal demand (won't matter since R_k ≈ 0)
        return np.ones(N_RESOURCES) / sqrt(N_RESOURCES)


def compute_coverage(lam, c_vec):
    """
    Effective coverage: C_k = λ(k) · c(k).
    Dot product of demand (what cell needs) and allocation (what cell gets).
    """
    return np.dot(lam, c_vec)


def compute_protection(C_k, R_k, alpha=ALPHA):
    """
    Protection via exponential detection model (Koopman search theory):
    P_k = 1 - exp(-α · C_k / R_k)

    Derivation: threats arrive as Poisson process at rate R_k.
    Resources detect each threat independently with cumulative
    probability following exponential saturation.
    """
    if R_k <= 0 or C_k <= 0:
        return 0.0 if C_k <= 0 else 1.0

    ratio = alpha * C_k / R_k
    return 1.0 - exp(-ratio)


# =============================================================================
# 4. OPTIMIZATION
# =============================================================================

def optimize_allocation(cells, alpha=ALPHA):
    """
    Maximize Σ_k P_k = Σ_k [1 - exp(-α · C_k / R_k)]
    where C_k = λ(k) · c(k)
    subject to Σ_k c_n(k) ≤ B_n for each resource n, c_n(k) ≥ 0.

    λ(k) = E · r(k) / ‖E · r(k)‖ is the normalized demand vector.
    c_n(k) are free variables — not coupled through t_k.

    Solved via interleaved greedy water-filling: at each step, consider
    all (resource, cell) pairs and allocate one increment to the pair
    with the highest marginal protection gain:

        ∂P_k/∂c_n = (α · λ_n(k) / R_k) · exp(-α · C_k / R_k)

    For concave objectives, this converges to the true optimum as
    step size → 0. Resources interact through shared C_k: adding
    drones to a cell reduces marginal return for rangers in that
    same cell, preventing redundant over-coverage.
    """
    K = len(cells)
    lambdas = np.array([c['lambda'] for c in cells])   # (K, N_RESOURCES)
    R = np.array([c['risk'] for c in cells])            # (K,)
    active = R > 1e-8
    R_safe = np.where(active, R, 1.0)
    n_active = int(np.sum(active))

    print(f"\nOptimizing over {n_active} active cells "
          f"({K - n_active} zero-risk cells skipped)")
    print(f"Budgets: {', '.join(f'{RESOURCE_NAMES[n]}={BUDGETS[n]:.0f}' for n in range(N_RESOURCES))}")

    # Allocation state
    c_opt = np.zeros((N_RESOURCES, K))
    C = np.zeros(K)   # effective coverage per cell: C_k = λ(k) · c(k)
    budgets_left = BUDGETS.copy()

    # Step sizes: ~500 steps per resource
    steps = np.array([max(BUDGETS[n] / 500, 0.01) for n in range(N_RESOURCES)])

    total_steps = int(sum(BUDGETS / steps))
    print(f"Optimizing via interleaved greedy water-filling "
          f"(~{total_steps} steps)...")

    # Pre-compute the exp term (updated incrementally)
    exp_term = np.where(active, np.exp(-alpha * C / R_safe), 0.0)

    while np.any(budgets_left > steps * 0.5):
        # Compute marginal return for all (resource, cell) pairs
        # ∂P_k/∂c_n = (α · λ_n(k) / R_k) · exp(-α · C_k / R_k)
        best_val = -1.0
        best_n = -1
        best_k = -1

        for n in range(N_RESOURCES):
            if budgets_left[n] <= steps[n] * 0.5:
                continue
            marginals = (alpha * lambdas[:, n] / R_safe) * exp_term
            k_best = np.argmax(marginals)
            if marginals[k_best] > best_val:
                best_val = marginals[k_best]
                best_n = n
                best_k = k_best

        if best_n < 0:
            break

        # Allocate one increment
        alloc = min(steps[best_n], budgets_left[best_n])
        c_opt[best_n, best_k] += alloc
        C[best_k] += lambdas[best_k, best_n] * alloc
        budgets_left[best_n] -= alloc

        # Update exp term only for the changed cell
        exp_term[best_k] = (np.exp(-alpha * C[best_k] / R_safe[best_k])
                            if active[best_k] else 0.0)

    print(f"✓ Optimization complete")

    # Total protection
    ratios = np.where(active, alpha * C / R_safe, 0.0)
    total_P = float(np.sum(np.where(active, 1.0 - np.exp(-ratios), 0.0)))

    print(f"Total protection: {total_P:.2f} / {n_active} active cells "
          f"(mean P = {total_P / max(n_active, 1):.4f})")
    for n in range(N_RESOURCES):
        used = float(np.sum(c_opt[n]))
        n_cells_with = int(np.sum(c_opt[n] > 0.01))
        print(f"  {RESOURCE_NAMES[n]}: {used:.1f} / {BUDGETS[n]:.0f} deployed "
              f"across {n_cells_with} cells")

    return c_opt, total_P


# =============================================================================
# 5. MODEL RUNNER
# =============================================================================

def run_model(data):
    """
    Compute per-cell risks, normalize, optimize resource allocation,
    compute protection, save CSV, and print summary.
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
                    veg = 0.05 if is_pan(lat, lon) else 0.3
                    ndvi_missing += 1
            else:
                veg = 0.05 if is_pan(lat, lon) else 1.0
                ndvi_val = veg

            # Climate data
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

    if use_ndvi and ndvi_values:
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
        d['r_f'] = sat_norm(d['raw_f'], c_f)
        d['r_p'] = sat_norm(d['raw_p'], c_p)

        # Combined risk: geometric mean of normalized risks
        d['risk'] = sqrt(d['r_f'] * d['r_p'])

        # Demand vector: λ(k) = E · r(k) / ‖E · r(k)‖
        d['lambda'] = compute_demand_vector(d['r_f'], d['r_p'])

    # --- Optimize resource allocation ---
    print(f"\n{'='*60}")
    print("RESOURCE ALLOCATION OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Effectiveness matrix E:")
    for n in range(N_RESOURCES):
        print(f"  {RESOURCE_NAMES[n]:>8s}: fire={E[n,0]:.1f}, poaching={E[n,1]:.1f}")
    print(f"Resource efficiency α = {ALPHA}")

    c_opt, total_P = optimize_allocation(cells, alpha=ALPHA)

    # --- Compute per-cell protection with optimal allocation ---
    for k, d in enumerate(cells):
        c_vec = c_opt[:, k]
        for n in range(N_RESOURCES):
            d[f'c_{RESOURCE_NAMES[n].lower()}'] = c_vec[n]

        C_k = np.dot(d['lambda'], c_vec)
        d['coverage'] = C_k
        d['protection'] = compute_protection(C_k, d['risk'], ALPHA)

    # --- Also compute uniform allocation for comparison ---
    uniform_P = 0.0
    n_active = sum(1 for d in cells if d['risk'] > 1e-8)
    for k, d in enumerate(cells):
        if d['risk'] <= 1e-8:
            d['protection_uniform'] = 1.0
            continue
        c_uniform = np.array([BUDGETS[n] / len(cells) for n in range(N_RESOURCES)])
        C_k_u = np.dot(d['lambda'], c_uniform)
        P_k_u = compute_protection(C_k_u, d['risk'], ALPHA)
        d['protection_uniform'] = P_k_u
        uniform_P += P_k_u

    print(f"\nComparison:")
    print(f"  Uniform allocation:  total P = {uniform_P:.2f}, mean P = {uniform_P/max(n_active,1):.4f}")
    print(f"  Optimal allocation:  total P = {total_P:.2f}, mean P = {total_P/max(n_active,1):.4f}")
    improvement = (total_P - uniform_P) / max(uniform_P, 1e-12) * 100
    print(f"  Improvement: {improvement:.1f}%")

    # --- Save results ---
    out = "etosha_protectedness_grid.csv"
    # Build dynamic field list
    lambda_fields = [f'lambda_{RESOURCE_NAMES[n].lower()}' for n in range(N_RESOURCES)]
    alloc_fields = [f'c_{RESOURCE_NAMES[n].lower()}' for n in range(N_RESOURCES)]
    fields = ['row', 'col', 'lat', 'lon', 'fire_count', 'ndvi',
              'r_f', 'r_p', 'risk'] + lambda_fields + alloc_fields + [
              'coverage', 'protection', 'protection_uniform', 'in_pan']

    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for d in cells:
            # Flatten lambda for CSV
            for n in range(N_RESOURCES):
                d[f'lambda_{RESOURCE_NAMES[n].lower()}'] = round(d['lambda'][n], 4)
            # Round for readability
            for key in ['r_f', 'r_p', 'risk', 'coverage',
                        'protection', 'protection_uniform'] + alloc_fields:
                if key in d:
                    d[key] = round(d[key], 6)
            w.writerow(d)
    print(f"\n✓ Saved {len(cells)} cells to {out}")

    # --- Summary Statistics ---
    risks = [d['risk'] for d in cells]
    r_fs = [d['r_f'] for d in cells]
    r_ps = [d['r_p'] for d in cells]
    prots = [d['protection'] for d in cells]

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Fire risk (r̃_f):     [{min(r_fs):.4f}, {max(r_fs):.4f}], mean={np.mean(r_fs):.4f}")
    print(f"Poaching risk (r̃_p): [{min(r_ps):.4f}, {max(r_ps):.4f}], mean={np.mean(r_ps):.4f}")
    print(f"Combined risk (R):   [{min(risks):.4f}, {max(risks):.4f}], mean={np.mean(risks):.4f}")
    print(f"Protection (P):      [{min(prots):.4f}, {max(prots):.4f}], mean={np.mean(prots):.4f}")

    # Cells with fire detections
    fire_cells = sum(1 for d in cells if d['fire_count'] > 0)
    print(f"\nCells with fire detections: {fire_cells}/{len(cells)} "
          f"({100*fire_cells/len(cells):.1f}%)")

    # Top 10 highest risk
    top = sorted(cells, key=lambda d: d['risk'], reverse=True)
    print(f"\nTop 10 highest risk cells:")
    res_hdr = ''.join(f'{RESOURCE_NAMES[n]:>9s}' for n in range(N_RESOURCES))
    print(f"  {'Lat':>8} {'Lon':>8} {'R':>6} {'r_f':>6} {'r_p':>6} {res_hdr} {'P':>6}")
    for c in top[:10]:
        res_vals = ''.join(f"{c.get(f'c_{RESOURCE_NAMES[n].lower()}', 0):>9.2f}"
                           for n in range(N_RESOURCES))
        print(f"  {c['lat']:8.4f} {c['lon']:8.4f} {c['risk']:6.4f} "
              f"{c['r_f']:6.4f} {c['r_p']:6.4f} {res_vals} {c['protection']:6.4f}")

    # Top 10 most vulnerable (lowest protection among active cells)
    active_cells = [d for d in cells if d['risk'] > 1e-8]
    vuln = sorted(active_cells, key=lambda d: d['protection'])
    print(f"\nTop 10 most vulnerable (lowest protection):")
    res_hdr = ''.join(f'{RESOURCE_NAMES[n]:>9s}' for n in range(N_RESOURCES))
    print(f"  {'Lat':>8} {'Lon':>8} {'P':>6} {'R':>6} {res_hdr}")
    for c in vuln[:10]:
        res_vals = ''.join(f"{c.get(f'c_{RESOURCE_NAMES[n].lower()}', 0):>9.2f}"
                           for n in range(N_RESOURCES))
        print(f"  {c['lat']:8.4f} {c['lon']:8.4f} {c['protection']:6.4f} "
              f"{c['risk']:6.4f} {res_vals}")

    # Resource allocation summary by risk quartile
    sorted_by_risk = sorted(active_cells, key=lambda d: d['risk'])
    Q = len(sorted_by_risk) // 4
    if Q > 0:
        quartiles = [
            ("Low risk (Q1)", sorted_by_risk[:Q]),
            ("Med-low (Q2)", sorted_by_risk[Q:2*Q]),
            ("Med-high (Q3)", sorted_by_risk[2*Q:3*Q]),
            ("High risk (Q4)", sorted_by_risk[3*Q:]),
        ]
        res_hdr = ''.join(f'{"Σ "+RESOURCE_NAMES[n]:>11s}' for n in range(N_RESOURCES))
        print(f"\nResource allocation by risk quartile:")
        print(f"  {'Quartile':>16} {'Cells':>6} {'Mean R':>7} {'Mean P':>7} {res_hdr}")
        for label, qs in quartiles:
            mean_r = np.mean([d['risk'] for d in qs])
            mean_p = np.mean([d['protection'] for d in qs])
            res_sums = ''.join(
                f"{sum(d.get(f'c_{RESOURCE_NAMES[n].lower()}', 0) for d in qs):>11.1f}"
                for n in range(N_RESOURCES))
            print(f"  {label:>16} {len(qs):>6} {mean_r:>7.4f} {mean_p:>7.4f} {res_sums}")

    # Pan vs non-pan comparison
    pan_cells = [d for d in cells if d['in_pan']]
    nonpan_cells = [d for d in cells if not d['in_pan']]
    if pan_cells and nonpan_cells:
        print(f"\nPan vs Non-Pan:")
        print(f"  Pan:     {len(pan_cells)} cells, "
              f"mean R={np.mean([d['risk'] for d in pan_cells]):.4f}, "
              f"mean P={np.mean([d['protection'] for d in pan_cells]):.4f}")
        print(f"  Non-pan: {len(nonpan_cells)} cells, "
              f"mean R={np.mean([d['risk'] for d in nonpan_cells]):.4f}, "
              f"mean P={np.mean([d['protection'] for d in nonpan_cells]):.4f}")

    return cells


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    data = preprocess()
    cells = run_model(data)


if __name__ == "__main__":
    main()