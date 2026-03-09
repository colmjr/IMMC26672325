"""
Etosha National Park — Figure Generation for IMMC Paper
Reads model output CSVs and produces publication-ready figures.

Usage:
    pip install matplotlib numpy
    python figures.py

Outputs PNG files in ./figures/ directory.
"""

import numpy as np
import csv
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── Configuration ─────────────────────────────────────────────────────────────

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Grid dimensions (must match model.py)
N_ROWS, N_COLS = 24, 57
DPI = 200

# Etosha Pan approximate bounds (for overlay)
PAN_LAT = (-19.00, -18.65)
PAN_LON = (15.40, 16.80)

# Color scheme
CMAP_RISK = 'YlOrRd'
CMAP_PROT = 'YlGnBu'
CMAP_ALLOC = 'viridis'


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_grid_csv(path="etosha_protectedness_grid.csv"):
    """Load the main model output CSV into a list of dicts."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = {}
            for k, v in row.items():
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
            rows.append(d)
    print(f"Loaded {len(rows)} cells from {path}")
    return rows


def to_grid(cells, field):
    """Convert a flat list of cell dicts to a 2D grid array."""
    grid = np.full((N_ROWS, N_COLS), np.nan)
    for d in cells:
        r, c = int(d['row']), int(d['col'])
        if 0 <= r < N_ROWS and 0 <= c < N_COLS:
            grid[r, c] = d.get(field, np.nan)
    return grid


def load_csv(path):
    """Load any CSV into a list of dicts with numeric conversion."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = {}
            for k, v in row.items():
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v.strip().strip('"')
            rows.append(d)
    return rows


# ── Pan overlay helper ────────────────────────────────────────────────────────

def add_pan_outline(ax, lat_edges, lon_edges):
    """Draw a dashed rectangle showing the Etosha Pan extent."""
    # Convert pan lat/lon to grid row/col indices
    lat_min_idx = np.searchsorted(lat_edges, PAN_LAT[0]) - 0.5
    lat_max_idx = np.searchsorted(lat_edges, PAN_LAT[1]) - 0.5
    lon_min_idx = np.searchsorted(lon_edges, PAN_LON[0]) - 0.5
    lon_max_idx = np.searchsorted(lon_edges, PAN_LON[1]) - 0.5

    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (lon_min_idx, lat_min_idx),
        lon_max_idx - lon_min_idx,
        lat_max_idx - lat_min_idx,
        linewidth=1.5, edgecolor='white', facecolor='none',
        linestyle='--', label='Etosha Pan'
    )
    ax.add_patch(rect)


# ── Figure 1: Spatial risk maps (r_f, r_p, R) ────────────────────────────────

def fig_risk_maps(cells):
    """Three-panel spatial risk heatmap: fire, poaching, combined."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    fields = [('r_f', 'Fire Risk  $\\tilde{r}_f$'),
              ('r_p', 'Poaching Risk  $\\tilde{r}_p$'),
              ('risk', 'Combined Risk  $R_k$')]

    for ax, (field, title) in zip(axes, fields):
        grid = to_grid(cells, field)
        # Flip so south is at bottom
        im = ax.imshow(grid, cmap=CMAP_RISK, aspect='auto',
                        origin='lower', vmin=0, vmax=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Column (→ East)')
        ax.set_ylabel('Row (→ North)')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Risk')

    path = os.path.join(FIGDIR, "fig1_risk_maps.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path}")


# ── Figure 2: Resource allocation maps ────────────────────────────────────────

def fig_allocation_maps(cells):
    """Three-panel resource allocation: drones, rangers, sensors."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    fields = [('c_drones', 'Drone Allocation'),
              ('c_rangers', 'Ranger Allocation'),
              ('c_sensors', 'Sensor Allocation')]

    for ax, (field, title) in zip(axes, fields):
        grid = to_grid(cells, field)
        vmax = np.nanpercentile(grid, 99) if np.nanmax(grid) > 0 else 1
        im = ax.imshow(grid, cmap=CMAP_ALLOC, aspect='auto',
                        origin='lower', vmin=0, vmax=max(vmax, 0.01))
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Column (→ East)')
        ax.set_ylabel('Row (→ North)')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Units allocated')

    path = os.path.join(FIGDIR, "fig2_allocation_maps.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path}")


# ── Figure 3: Protection heatmap ──────────────────────────────────────────────

def fig_protection_map(cells):
    """Single-panel protection heatmap with colorbar."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    grid = to_grid(cells, 'protection')
    im = ax.imshow(grid, cmap=CMAP_PROT, aspect='auto',
                    origin='lower', vmin=0, vmax=1)
    ax.set_title('Protection Index  $P_k$', fontsize=13, fontweight='bold')
    ax.set_xlabel('Column (→ East)')
    ax.set_ylabel('Row (→ North)')
    plt.colorbar(im, ax=ax, label='Protection $P_k$')

    path = os.path.join(FIGDIR, "fig3_protection_map.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path}")


# ── Figure 4: Staffing curve ─────────────────────────────────────────────────

def fig_staffing_curve(sweep_path="staffing_sweep.csv",
                        sweep_2015_path="staffing_sweep_2015.csv"):
    """Ranger count vs mean P, with IUCN and current staff markers."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    # 2024 sweep
    data = load_csv(sweep_path)
    if not data:
        print(f"  Skipping staffing curve: {sweep_path} not found")
        return

    rangers = [d['rangers'] for d in data]
    mean_p = [d['mean_P'] for d in data]
    ax.plot(rangers, mean_p, 'o-', color='#2563eb', linewidth=2,
            markersize=6, label='2024 data', zorder=3)

    # 2015 sweep (if available)
    if os.path.isfile(sweep_2015_path):
        data15 = load_csv(sweep_2015_path)
        r15 = [d['rangers'] for d in data15]
        p15 = [d['mean_P'] for d in data15]
        ax.plot(r15, p15, 's--', color='#9ca3af', linewidth=1.5,
                markersize=5, label='2015 data', zorder=2)

    # Reference lines
    ax.axvline(230, color='#f59e0b', linestyle=':', linewidth=1.5,
               label='IUCN benchmark (230)')
    ax.axvline(295, color='#10b981', linestyle=':', linewidth=1.5,
               label='Current staff (295)')
    ax.axhline(0.5, color='#ef4444', linestyle='--', linewidth=1,
               alpha=0.5, label='$P = 0.5$ target')

    ax.set_xlabel('Number of Rangers', fontsize=12)
    ax.set_ylabel('Mean Protection $\\bar{P}$', fontsize=12)
    ax.set_title('Staffing Requirements: Rangers vs. Protection', fontsize=13,
                  fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1050)
    ax.set_ylim(0.2, 0.65)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIGDIR, "fig4_staffing_curve.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path}")


# ── Figure 5: Budget sensitivity curves ───────────────────────────────────────

def fig_budget_sensitivity(path="sensitivity_budgets.csv"):
    """Three overlaid curves: budget multiplier vs mean P per resource."""
    data = load_csv(path)
    if not data:
        print(f"  Skipping budget sensitivity: {path} not found")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    colors = {'Drones': '#3b82f6', 'Rangers': '#ef4444', 'Sensors': '#10b981'}
    markers = {'Drones': 'D', 'Rangers': 's', 'Sensors': '^'}

    for resource in ['Drones', 'Rangers', 'Sensors']:
        subset = [d for d in data if d['resource'] == resource]
        mults = [d['multiplier'] for d in subset]
        pvals = [d['mean_P'] for d in subset]
        ax.plot(mults, pvals, f'{markers[resource]}-',
                color=colors[resource], linewidth=2, markersize=7,
                label=resource)

    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Budget Multiplier (1.0 = baseline)', fontsize=12)
    ax.set_ylabel('Mean Protection $\\bar{P}$', fontsize=12)
    ax.set_title('Budget Sensitivity by Resource Type', fontsize=13,
                  fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.1, 3.1)
    ax.grid(True, alpha=0.3)

    path_out = os.path.join(FIGDIR, "fig5_budget_sensitivity.png")
    fig.savefig(path_out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path_out}")


# ── Figure 6: Scenario comparison bar chart ───────────────────────────────────

def fig_scenario_bars(path="sensitivity_scenarios.csv"):
    """Horizontal bar chart of scenarios sorted by mean P."""
    data = load_csv(path)
    if not data:
        print(f"  Skipping scenarios: {path} not found")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    # Sort by mean_P
    data_sorted = sorted(data, key=lambda d: d['mean_P'])

    names = [d['scenario'] for d in data_sorted]
    vals = [d['mean_P'] for d in data_sorted]

    # Color by category
    colors = []
    for d in data_sorted:
        name = d['scenario']
        if 'Baseline' in name:
            colors.append('#6b7280')
        elif 'only' in name.lower() or 'No ' in name or 'no ' in name:
            colors.append('#ef4444')
        elif 'cut' in name.lower() or 'Harsh' in name:
            colors.append('#f59e0b')
        elif 'Double' in name or 'upgrade' in name.lower():
            colors.append('#10b981')
        else:
            colors.append('#3b82f6')

    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white',
                    linewidth=0.5, height=0.7)

    # Baseline reference line
    baseline_p = next((d['mean_P'] for d in data if 'Baseline' in str(d.get('scenario', ''))), 0.455)
    ax.axvline(baseline_p, color='#6b7280', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Baseline ($P={baseline_p:.3f}$)')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Mean Protection $\\bar{P}$', fontsize=12)
    ax.set_title('Scenario Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, max(vals) * 1.1)
    ax.grid(True, axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

    path_out = os.path.join(FIGDIR, "fig6_scenarios.png")
    fig.savefig(path_out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path_out}")


# ── Figure 7: Alpha sensitivity ───────────────────────────────────────────────

def fig_alpha_sensitivity(path="sensitivity_alpha.csv"):
    """Alpha vs mean P and Q4 P on dual axis."""
    data = load_csv(path)
    if not data:
        print(f"  Skipping alpha sensitivity: {path} not found")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    alphas = [d['alpha'] for d in data]
    mean_p = [d['mean_P'] for d in data]
    q4_p = [d['Q4_high'] for d in data]

    ax1.plot(alphas, mean_p, 'o-', color='#2563eb', linewidth=2,
             markersize=7, label='Mean $P$')
    ax1.set_xlabel('Resource Efficiency $\\alpha$', fontsize=12)
    ax1.set_ylabel('Mean Protection $\\bar{P}$', fontsize=12, color='#2563eb')
    ax1.tick_params(axis='y', labelcolor='#2563eb')

    ax2 = ax1.twinx()
    ax2.plot(alphas, q4_p, 's--', color='#ef4444', linewidth=2,
             markersize=6, label='Q4 (high risk) $P$')
    ax2.set_ylabel('Q4 Protection', fontsize=12, color='#ef4444')
    ax2.tick_params(axis='y', labelcolor='#ef4444')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

    ax1.set_title('Effect of Resource Efficiency on Protection',
                   fontsize=13, fontweight='bold')
    ax1.axvline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    path_out = os.path.join(FIGDIR, "fig7_alpha_sensitivity.png")
    fig.savefig(path_out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {path_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("GENERATING FIGURES FOR IMMC PAPER")
    print("=" * 50)

    # Load main grid data
    grid_path = "etosha_protectedness_grid.csv"
    if os.path.isfile(grid_path):
        cells = load_grid_csv(grid_path)
        fig_risk_maps(cells)
        fig_allocation_maps(cells)
        fig_protection_map(cells)
    else:
        print(f"⚠ {grid_path} not found — run model.py first")
        print("  Skipping spatial maps")

    # Staffing curve
    fig_staffing_curve()

    # Sensitivity figures
    fig_budget_sensitivity()
    fig_scenario_bars()
    fig_alpha_sensitivity()

    print(f"\n✓ All figures saved to ./{FIGDIR}/")
    print("  Include in Typst with: #image(\"figures/fig1_risk_maps.png\")")


if __name__ == "__main__":
    main()