"""
Etosha National Park — Sensitivity & Scenario Analysis
IMMC International Round — Requirement 4

Investigates how changes in assumptions or available resources affect
the protection strategy. Sweeps over:
  1. Individual resource budgets (one-at-a-time)
  2. Resource efficiency parameter α
  3. E matrix perturbations (resource effectiveness)
  4. Real-world scenarios (budget cuts, tech upgrades, no-tech baseline)

Uses a fixed baseline year (2024) with September NDVI and FIRMS data.

Usage:
    python sensitivity.py
"""

import numpy as np
import csv
import os
import copy
from math import sqrt

from preprocess import (
    lat_centers, lon_centers, n_rows, n_cols,
    CELL_AREA_KM2,
    load_firms, load_ndvi_multi,
    ClimateData,
)

from model import (
    E as E_DEFAULT, RESOURCE_NAMES, N_RESOURCES, BUDGETS as BUDGETS_DEFAULT,
    ALPHA as ALPHA_DEFAULT,
    optimize_allocation, compute_demand_vector, compute_protection,
)

from temporal import (
    SEPT_NDVI_FILES, filter_firms_by_year,
    load_climate_once, run_single_year,
)


# =============================================================================
# 1. HELPERS
# =============================================================================

def run_with_params(year_fires, get_ndvi, use_ndvi, climate,
                    budgets=None, alpha=None, E_override=None):
    """
    Run the model with optionally overridden parameters.
    Temporarily patches model globals, runs, then restores.
    """
    import model as mod

    # Save originals
    orig_budgets = mod.BUDGETS.copy()
    orig_alpha = mod.ALPHA
    orig_E = mod.E.copy()
    orig_N = mod.N_RESOURCES

    try:
        if budgets is not None:
            mod.BUDGETS = np.array(budgets, dtype=float)
        if alpha is not None:
            mod.ALPHA = alpha
        if E_override is not None:
            mod.E = np.array(E_override, dtype=float)
            mod.N_RESOURCES = mod.E.shape[0]

        stats = run_single_year(
            year_fires, get_ndvi, use_ndvi, climate,
            budgets=mod.BUDGETS, alpha=mod.ALPHA, quiet=True
        )
        return stats
    finally:
        mod.BUDGETS = orig_budgets
        mod.ALPHA = orig_alpha
        mod.E = orig_E
        mod.N_RESOURCES = orig_N


def load_baseline_data(target_year=2024):
    """Load NDVI, fires, and climate for the baseline year."""
    all_fires = load_firms("FIRMS_dataset.csv")
    climate = load_climate_once()

    ndvi_file = SEPT_NDVI_FILES.get(target_year)
    get_ndvi, use_ndvi = None, False
    if ndvi_file and os.path.isfile(ndvi_file):
        _, get_ndvi = load_ndvi_multi([ndvi_file], label=f"Sept {target_year}")
        use_ndvi = get_ndvi is not None

    year_fires = filter_firms_by_year(all_fires, target_year)
    print(f"Baseline: {target_year}, {len(year_fires)} fires")

    return year_fires, get_ndvi, use_ndvi, climate


# =============================================================================
# 2. BUDGET SENSITIVITY (one-at-a-time)
# =============================================================================

def budget_sensitivity(year_fires, get_ndvi, use_ndvi, climate):
    """
    Vary each resource budget independently while holding others at default.
    """
    print(f"\n{'='*60}")
    print("SENSITIVITY 1: Individual resource budgets")
    print(f"{'='*60}")
    print(f"Baseline budgets: {', '.join(f'{RESOURCE_NAMES[n]}={BUDGETS_DEFAULT[n]:.0f}' for n in range(N_RESOURCES))}")

    # Get baseline
    baseline = run_with_params(year_fires, get_ndvi, use_ndvi, climate)
    print(f"Baseline mean P = {baseline['mean_P']:.4f}")

    results = []

    for n in range(N_RESOURCES):
        name = RESOURCE_NAMES[n]
        base_val = BUDGETS_DEFAULT[n]

        # Sweep from 0% to 300% of default
        multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

        print(f"\n  --- {name} (baseline={base_val:.0f}) ---")
        print(f"  {'Mult':>6} {'Budget':>8} {'Mean P':>8} {'Min P':>8} {'ΔP%':>8}")

        for mult in multipliers:
            budgets = BUDGETS_DEFAULT.copy()
            budgets[n] = base_val * mult
            stats = run_with_params(year_fires, get_ndvi, use_ndvi, climate,
                                    budgets=budgets)
            delta = (stats['mean_P'] - baseline['mean_P']) / max(baseline['mean_P'], 1e-12) * 100
            print(f"  {mult:>6.2f} {budgets[n]:>8.0f} {stats['mean_P']:>8.4f} "
                  f"{stats['min_P']:>8.4f} {delta:>+8.1f}%")

            results.append({
                'resource': name,
                'multiplier': mult,
                'budget': budgets[n],
                'mean_P': stats['mean_P'],
                'min_P': stats['min_P'],
                'delta_pct': delta,
            })

    return results


# =============================================================================
# 3. ALPHA SENSITIVITY (resource efficiency)
# =============================================================================

def alpha_sensitivity(year_fires, get_ndvi, use_ndvi, climate):
    """
    Vary the resource efficiency parameter α.
    Higher α = resources more effective per unit.
    """
    print(f"\n{'='*60}")
    print("SENSITIVITY 2: Resource efficiency α")
    print(f"{'='*60}")
    print(f"Baseline α = {ALPHA_DEFAULT}")

    alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    baseline = run_with_params(year_fires, get_ndvi, use_ndvi, climate)
    results = []

    print(f"\n  {'α':>6} {'Mean P':>8} {'Min P':>8} {'ΔP%':>8} {'Q4 P':>8}")
    for a in alphas:
        stats = run_with_params(year_fires, get_ndvi, use_ndvi, climate, alpha=a)
        delta = (stats['mean_P'] - baseline['mean_P']) / max(baseline['mean_P'], 1e-12) * 100
        q4 = stats['quartile_means'].get('Q4_high', 0)
        marker = " ← baseline" if a == ALPHA_DEFAULT else ""
        print(f"  {a:>6.2f} {stats['mean_P']:>8.4f} {stats['min_P']:>8.4f} "
              f"{delta:>+8.1f}% {q4:>8.4f}{marker}")

        results.append({
            'alpha': a,
            'mean_P': stats['mean_P'],
            'min_P': stats['min_P'],
            'delta_pct': delta,
            'Q4_high': q4,
        })

    return results


# =============================================================================
# 4. E MATRIX SENSITIVITY (resource effectiveness)
# =============================================================================

def effectiveness_sensitivity(year_fires, get_ndvi, use_ndvi, climate):
    """
    Perturb each element of E by ±0.2 and measure impact on mean P.
    """
    print(f"\n{'='*60}")
    print("SENSITIVITY 3: E matrix (resource effectiveness)")
    print(f"{'='*60}")
    print(f"Baseline E:")
    risk_names = ['fire', 'poach']
    for n in range(N_RESOURCES):
        print(f"  {RESOURCE_NAMES[n]:>8s}: {', '.join(f'{risk_names[j]}={E_DEFAULT[n,j]:.2f}' for j in range(2))}")

    baseline = run_with_params(year_fires, get_ndvi, use_ndvi, climate)
    print(f"Baseline mean P = {baseline['mean_P']:.4f}")

    results = []

    print(f"\n  {'Resource':>10} {'Risk':>8} {'Base':>6} {'New':>6} {'Mean P':>8} {'ΔP%':>8}")

    for n in range(N_RESOURCES):
        for j in range(2):
            base_val = E_DEFAULT[n, j]
            for delta_e in [-0.2, +0.2]:
                new_val = np.clip(base_val + delta_e, 0.0, 1.0)
                if abs(new_val - base_val) < 0.01:
                    continue  # skip if clipped to same value

                E_mod = E_DEFAULT.copy()
                E_mod[n, j] = new_val

                stats = run_with_params(year_fires, get_ndvi, use_ndvi, climate,
                                        E_override=E_mod)
                delta = (stats['mean_P'] - baseline['mean_P']) / max(baseline['mean_P'], 1e-12) * 100

                print(f"  {RESOURCE_NAMES[n]:>10} {risk_names[j]:>8} {base_val:>6.2f} "
                      f"{new_val:>6.2f} {stats['mean_P']:>8.4f} {delta:>+8.1f}%")

                results.append({
                    'resource': RESOURCE_NAMES[n],
                    'risk_type': risk_names[j],
                    'base_val': base_val,
                    'new_val': new_val,
                    'mean_P': stats['mean_P'],
                    'delta_pct': delta,
                })

    return results


# =============================================================================
# 5. SCENARIO ANALYSIS (real-world what-ifs)
# =============================================================================

def scenario_analysis(year_fires, get_ndvi, use_ndvi, climate):
    """
    Test specific real-world scenarios against the baseline.
    """
    print(f"\n{'='*60}")
    print("SENSITIVITY 4: Scenario analysis")
    print(f"{'='*60}")

    scenarios = {
        'Baseline': {
            'budgets': BUDGETS_DEFAULT.copy(),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'No drones (ranger-only + sensors)': {
            'budgets': np.array([0.0, BUDGETS_DEFAULT[1], BUDGETS_DEFAULT[2]]),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'No sensors (drones + rangers only)': {
            'budgets': np.array([BUDGETS_DEFAULT[0], BUDGETS_DEFAULT[1], 0.0]),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'Rangers only (no technology)': {
            'budgets': np.array([0.0, BUDGETS_DEFAULT[1], 0.0]),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'Budget cut 50% (all resources halved)': {
            'budgets': BUDGETS_DEFAULT * 0.5,
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'Double all resources': {
            'budgets': BUDGETS_DEFAULT * 2.0,
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'Shift drones→sensors (0 drones, 70 sensors)': {
            'budgets': np.array([0.0, BUDGETS_DEFAULT[1], 70.0]),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'Shift sensors→drones (40 drones, 0 sensors)': {
            'budgets': np.array([40.0, BUDGETS_DEFAULT[1], 0.0]),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'IUCN minimum (1 ranger/100km², ~230 rangers)': {
            'budgets': np.array([BUDGETS_DEFAULT[0], 230.0, BUDGETS_DEFAULT[2]]),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'Current Etosha staff (295 rangers, as reported)': {
            'budgets': np.array([BUDGETS_DEFAULT[0], 295.0, BUDGETS_DEFAULT[2]]),
            'alpha': ALPHA_DEFAULT,
            'E': E_DEFAULT.copy(),
        },
        'Technology upgrade (α=1.5, better AI)': {
            'budgets': BUDGETS_DEFAULT.copy(),
            'alpha': 1.5,
            'E': E_DEFAULT.copy(),
        },
        'Low efficiency (α=0.5, harsh conditions)': {
            'budgets': BUDGETS_DEFAULT.copy(),
            'alpha': 0.5,
            'E': E_DEFAULT.copy(),
        },
    }

    results = []
    baseline_P = None

    print(f"\n  {'Scenario':<48} {'Mean P':>8} {'Min P':>8} {'ΔP%':>8} {'Q4 P':>8}")
    print(f"  {'-'*84}")

    for name, params in scenarios.items():
        stats = run_with_params(year_fires, get_ndvi, use_ndvi, climate,
                                budgets=params['budgets'],
                                alpha=params['alpha'],
                                E_override=params['E'])

        if baseline_P is None:
            baseline_P = stats['mean_P']

        delta = (stats['mean_P'] - baseline_P) / max(baseline_P, 1e-12) * 100
        q4 = stats['quartile_means'].get('Q4_high', 0)

        print(f"  {name:<48} {stats['mean_P']:>8.4f} {stats['min_P']:>8.4f} "
              f"{delta:>+8.1f}% {q4:>8.4f}")

        results.append({
            'scenario': name,
            'budgets': params['budgets'].tolist(),
            'alpha': params['alpha'],
            'mean_P': stats['mean_P'],
            'min_P': stats['min_P'],
            'delta_pct': delta,
            'Q4_high': q4,
        })

    return results


# =============================================================================
# 6. SAVE RESULTS
# =============================================================================

def save_budget_csv(results, filename="sensitivity_budgets.csv"):
    fields = ['resource', 'multiplier', 'budget', 'mean_P', 'min_P', 'delta_pct']
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v
                        for k, v in r.items()})
    print(f"✓ Saved to {filename}")


def save_alpha_csv(results, filename="sensitivity_alpha.csv"):
    fields = ['alpha', 'mean_P', 'min_P', 'delta_pct', 'Q4_high']
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v
                        for k, v in r.items()})
    print(f"✓ Saved to {filename}")


def save_effectiveness_csv(results, filename="sensitivity_E.csv"):
    fields = ['resource', 'risk_type', 'base_val', 'new_val', 'mean_P', 'delta_pct']
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v
                        for k, v in r.items()})
    print(f"✓ Saved to {filename}")


def save_scenario_csv(results, filename="sensitivity_scenarios.csv"):
    fields = ['scenario', 'alpha', 'mean_P', 'min_P', 'delta_pct', 'Q4_high']
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: round(v, 6) if isinstance(v, float) else v
                   for k, v in r.items() if k in fields}
            w.writerow(row)
    print(f"✓ Saved to {filename}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ETOSHA — SENSITIVITY & SCENARIO ANALYSIS (Requirement 4)")
    print("=" * 60)

    # Load baseline data
    year_fires, get_ndvi, use_ndvi, climate = load_baseline_data(2024)

    # --- Analysis 1: Budget sensitivity ---
    budget_results = budget_sensitivity(year_fires, get_ndvi, use_ndvi, climate)
    save_budget_csv(budget_results)

    # --- Analysis 2: Alpha sensitivity ---
    alpha_results = alpha_sensitivity(year_fires, get_ndvi, use_ndvi, climate)
    save_alpha_csv(alpha_results)

    # --- Analysis 3: E matrix sensitivity ---
    e_results = effectiveness_sensitivity(year_fires, get_ndvi, use_ndvi, climate)
    save_effectiveness_csv(e_results)

    # --- Analysis 4: Scenario analysis ---
    scenario_results = scenario_analysis(year_fires, get_ndvi, use_ndvi, climate)
    save_scenario_csv(scenario_results)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Output files:")
    print("  sensitivity_budgets.csv   — one-at-a-time budget sweeps")
    print("  sensitivity_alpha.csv     — resource efficiency parameter")
    print("  sensitivity_E.csv         — effectiveness matrix perturbations")
    print("  sensitivity_scenarios.csv — real-world scenario comparisons")

    # Key findings
    print(f"\nKey findings:")

    # Most sensitive resource
    if budget_results:
        # Find which resource at 0% causes the biggest drop
        zero_runs = [r for r in budget_results if r['multiplier'] == 0.0]
        if zero_runs:
            worst = min(zero_runs, key=lambda r: r['mean_P'])
            print(f"  Most critical resource: {worst['resource']} "
                  f"(removing it drops mean P by {worst['delta_pct']:.1f}%)")

    # Alpha sensitivity
    if alpha_results:
        low_a = next((r for r in alpha_results if r['alpha'] == 0.5), None)
        high_a = next((r for r in alpha_results if r['alpha'] == 2.0), None)
        if low_a and high_a:
            print(f"  α sensitivity: P ranges from {low_a['mean_P']:.4f} (α=0.5) "
                  f"to {high_a['mean_P']:.4f} (α=2.0)")

    # Most sensitive E element
    if e_results:
        biggest = max(e_results, key=lambda r: abs(r['delta_pct']))
        print(f"  Most sensitive E element: {biggest['resource']}-{biggest['risk_type']} "
              f"({biggest['base_val']:.2f}→{biggest['new_val']:.2f}, "
              f"ΔP={biggest['delta_pct']:+.1f}%)")


if __name__ == "__main__":
    main()