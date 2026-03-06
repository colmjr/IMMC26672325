"""
Etosha National Park — Data Loading & Preprocessing
IMMC International Round

Loads raw data (FIRMS fires, NDVI, NASA POWER climate) and prepares
it for the risk model. Grid setup, helper functions, and known locations
all live here.
"""

import numpy as np
import csv
import os
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from datetime import datetime

# =============================================================================
# 1. GRID SETUP
# =============================================================================

LAT_MIN, LAT_MAX = -19.50, -18.45
LON_MIN, LON_MAX = 14.45, 17.15

CELL_SIZE_KM = 5

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * cos(radians(19.0))  # ~104.9 km at 19°S

CELL_SIZE_LAT = CELL_SIZE_KM / KM_PER_DEG_LAT
CELL_SIZE_LON = CELL_SIZE_KM / KM_PER_DEG_LON

lat_edges = np.arange(LAT_MIN, LAT_MAX, CELL_SIZE_LAT)
lon_edges = np.arange(LON_MIN, LON_MAX, CELL_SIZE_LON)
lat_centers = lat_edges + CELL_SIZE_LAT / 2
lon_centers = lon_edges + CELL_SIZE_LON / 2

n_rows = len(lat_centers)
n_cols = len(lon_centers)
CELL_AREA_KM2 = CELL_SIZE_KM ** 2


# =============================================================================
# 2. HELPERS
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def min_dist(lat, lon, points):
    """Minimum distance from (lat,lon) to a list of (lat,lon) points."""
    if not points:
        return float('inf')
    return min(haversine(lat, lon, p[0], p[1]) for p in points)

def to_cell(lat, lon):
    """Convert lat/lon to grid cell indices (row, col)."""
    row = int((lat - LAT_MIN) / CELL_SIZE_LAT)
    col = int((lon - LON_MIN) / CELL_SIZE_LON)
    if 0 <= row < n_rows and 0 <= col < n_cols:
        return (row, col)
    return None

def sat_norm(x, c):
    """Saturating normalization: x/(x+c). Maps [0,∞) → [0,1)."""
    return x / (x + c) if c > 0 else 0.0


# =============================================================================
# 3. KNOWN LOCATIONS
# =============================================================================

RANGER_STATIONS = [
    (-19.18, 15.92),   # Okaukuejo (HQ)
    (-18.81, 16.33),   # Halali
    (-18.81, 16.94),   # Namutoni
    (-19.04, 14.98),   # Dolomite Camp
]

ENTRY_GATES = [
    (-19.18, 15.90),   # Anderson Gate
    (-18.73, 16.93),   # Von Lindequist Gate
    (-18.47, 16.72),   # King Nehale Gate
    (-19.30, 15.70),   # Galton Gate
]

# Documented waterholes — rhino concentration proxy for poaching target value
WATERHOLES = [
    (-19.18, 15.92),   # Okaukuejo
    (-18.81, 16.33),   # Halali
    (-18.81, 16.94),   # Klein Namutoni / Namutoni
    (-18.83, 16.85),   # Chudop
    (-18.76, 16.78),   # Klein Okevi
    (-18.78, 16.55),   # Goas
    (-18.75, 16.48),   # Rietfontein
    (-18.90, 16.18),   # Charitsaub
    (-18.94, 16.00),   # Salvadora
    (-18.92, 15.97),   # Nebrownii
    (-19.17, 15.88),   # Olifantsbad
    (-19.08, 15.80),   # Gemsbokvlakte
]

# Sampled boundary points around the park perimeter
BOUNDARY_POINTS = [
    (-18.45, 15.50), (-18.45, 15.80), (-18.45, 16.10), (-18.45, 16.40),
    (-18.45, 16.70), (-18.50, 17.00), (-18.55, 17.10),
    (-18.70, 17.15), (-18.90, 17.10), (-19.10, 17.10), (-19.30, 17.05),
    (-19.35, 16.80), (-19.40, 16.50), (-19.45, 16.20), (-19.45, 15.90),
    (-19.40, 15.60), (-19.35, 15.30), (-19.30, 15.00),
    (-19.20, 14.70), (-19.10, 14.50), (-18.90, 14.50), (-18.70, 14.55),
    (-18.55, 14.70), (-18.45, 14.90), (-18.45, 15.20),
]

# Sampled road points along the main tourist loop
ROAD_POINTS = [
    # Okaukuejo → Halali (south of pan)
    (-19.18, 15.92), (-19.15, 16.00), (-19.10, 16.08), (-19.05, 16.15),
    (-19.00, 16.20), (-18.95, 16.25), (-18.90, 16.30), (-18.81, 16.33),
    # Halali → Namutoni (east)
    (-18.81, 16.40), (-18.80, 16.50), (-18.80, 16.60), (-18.80, 16.70),
    (-18.80, 16.80), (-18.81, 16.94),
    # Northern road segments
    (-18.65, 15.40), (-18.60, 15.60), (-18.55, 15.80),
    (-18.55, 16.00), (-18.55, 16.20), (-18.55, 16.40),
    # Western approach (Dolomite)
    (-19.10, 14.95), (-19.15, 15.10), (-19.18, 15.30),
    (-19.18, 15.50), (-19.18, 15.70), (-19.18, 15.90),
]

# Etosha Pan approximate bounds
PAN_LAT = (-19.00, -18.65)
PAN_LON = (15.40, 16.80)

def is_pan(lat, lon):
    return PAN_LAT[0] <= lat <= PAN_LAT[1] and PAN_LON[0] <= lon <= PAN_LON[1]


# =============================================================================
# 4. NDVI LOADING (GeoTiff from AppEEARS)
# =============================================================================

# Dry-season months for NDVI averaging (peak fire risk period)
DRY_SEASON_KEYWORDS = ['may', 'jun', '06-', 'jul', 'aug', 'sep', 'sept', 'oct']
ALL_MONTH_KEYWORDS = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
    'jun': 6, '06-10': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9,
    'oct': 10, 'nov': 11, 'dec': 12
}


def _read_single_ndvi(tiff_path):
    """
    Read a single MODIS NDVI GeoTiff → (ndvi_2d_array, transform).
    Returns scaled NDVI [0,1] with NaN for invalid pixels.
    """
    import rasterio
    with rasterio.open(tiff_path) as src:
        raw = src.read(1)
        transform = src.transform
        nodata = src.nodata
        ndvi = raw.astype(float) / 10000.0
        if nodata is not None:
            ndvi[raw == nodata] = np.nan
        ndvi[ndvi < -0.2] = np.nan
        ndvi[ndvi > 1.0] = np.nan
        return ndvi, transform, src.shape, src.bounds


def load_ndvi_multi(tiff_paths, label="NDVI"):
    """
    Load one or more MODIS NDVI GeoTiffs and average them pixel-wise.

    Returns:
        ndvi_avg: 2D numpy array of averaged NDVI
        get_ndvi(lat, lon): function returning NDVI for a coordinate
    """
    try:
        import rasterio
    except ImportError:
        print("Warning: rasterio not installed. Run: pip install rasterio")
        return None, None

    if not tiff_paths:
        return None, None

    try:
        stack = []
        transform = None
        shape = None

        for p in tiff_paths:
            ndvi, tf, sh, bounds = _read_single_ndvi(p)
            stack.append(ndvi)
            if transform is None:
                transform = tf
                shape = sh
                print(f"  {label} raster: {sh}, bounds: {bounds}")
            fname = os.path.basename(p)
            valid = ndvi[~np.isnan(ndvi)]
            print(f"  {fname}: mean={valid.mean():.4f}, range=[{valid.min():.4f}, {valid.max():.4f}]")

        # Pixel-wise nanmean across all months
        ndvi_avg = np.nanmean(np.stack(stack, axis=0), axis=0)
        valid = ndvi_avg[~np.isnan(ndvi_avg)]
        print(f"  → Averaged {len(stack)} layers: mean={valid.mean():.4f}, "
              f"range=[{valid.min():.4f}, {valid.max():.4f}]")

        def get_ndvi(lat, lon):
            """Sample averaged NDVI at a lat/lon coordinate."""
            try:
                col_idx, row_idx = ~transform * (lon, lat)
                row_idx, col_idx = int(row_idx), int(col_idx)
                if 0 <= row_idx < ndvi_avg.shape[0] and 0 <= col_idx < ndvi_avg.shape[1]:
                    val = ndvi_avg[row_idx, col_idx]
                    if not np.isnan(val):
                        return max(val, 0.0)
            except:
                pass
            return None

        return ndvi_avg, get_ndvi

    except Exception as e:
        print(f"Warning: Could not load NDVI: {e}")
        return None, None


def compute_cell_ndvi(get_ndvi, lat, lon, n_samples=9):
    """
    Average NDVI over a cell by sampling multiple points within it.
    More robust than single-point sampling at cell center.
    """
    if get_ndvi is None:
        return None

    offsets = np.linspace(-CELL_SIZE_LAT/3, CELL_SIZE_LAT/3, 3)
    values = []
    for dlat in offsets:
        for dlon in np.linspace(-CELL_SIZE_LON/3, CELL_SIZE_LON/3, 3):
            v = get_ndvi(lat + dlat, lon + dlon)
            if v is not None:
                values.append(v)

    return np.mean(values) if values else None


# =============================================================================
# 5. LOAD NASA POWER CLIMATE DATA
# =============================================================================

def load_power_csv(path):
    """
    Load a NASA POWER regional monthly CSV.
    Returns dict: { (lat, lon): { 'JAN': val, ..., 'ANN': val } }
    averaged across all years in the file.
    """
    data = defaultdict(lambda: defaultdict(list))
    months = ['JAN','FEB','MAR','APR','MAY','JUN',
              'JUL','AUG','SEP','OCT','NOV','DEC','ANN']

    try:
        with open(path, 'r') as f:
            # Skip header lines until we find the column header
            for line in f:
                if line.startswith('PARAMETER,'):
                    header = line.strip().split(',')
                    break
            else:
                print(f"Warning: No header found in {path}")
                return {}

            reader = csv.DictReader(f, fieldnames=header)
            for row in reader:
                try:
                    lat = float(row['LAT'])
                    lon = float(row['LON'])
                    for m in months:
                        val = float(row.get(m, -999))
                        if val != -999:
                            data[(lat, lon)][m].append(val)
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Warning: '{path}' not found.")
        return {}

    # Average across years
    result = {}
    for key, month_data in data.items():
        result[key] = {}
        for m, vals in month_data.items():
            result[key][m] = np.mean(vals) if vals else -999

    return result


def interpolate_climate(climate_data, lat, lon, months=None):
    """
    Bilinear interpolation of climate data to a specific lat/lon.
    If months is provided, average over those months; else use 'ANN'.
    """
    if not climate_data:
        return None

    # Get the unique lat/lon grid from the data
    lats_grid = sorted(set(k[0] for k in climate_data.keys()))
    lons_grid = sorted(set(k[1] for k in climate_data.keys()))

    if not lats_grid or not lons_grid:
        return None

    # Find bracketing indices
    def bracket(val, grid):
        for i in range(len(grid) - 1):
            if grid[i] <= val <= grid[i+1]:
                return i, i+1
        # Clamp to edges
        if val < grid[0]:
            return 0, 0
        return len(grid)-1, len(grid)-1

    i0, i1 = bracket(lat, lats_grid)
    j0, j1 = bracket(lon, lons_grid)

    lat0, lat1 = lats_grid[i0], lats_grid[i1]
    lon0, lon1 = lons_grid[j0], lons_grid[j1]

    def get_val(lt, ln):
        if (lt, ln) not in climate_data:
            return None
        d = climate_data[(lt, ln)]
        if months:
            vals = [d.get(m, -999) for m in months]
            vals = [v for v in vals if v != -999]
            return np.mean(vals) if vals else None
        else:
            return d.get('ANN', None)

    # Get corner values
    q11 = get_val(lat0, lon0)
    q21 = get_val(lat1, lon0)
    q12 = get_val(lat0, lon1)
    q22 = get_val(lat1, lon1)

    corners = [q11, q21, q12, q22]
    valid = [c for c in corners if c is not None]
    if not valid:
        return None

    # If all same point (edge case), return mean
    if lat0 == lat1 and lon0 == lon1:
        return q11

    # Replace None with mean of valid
    mean_v = np.mean(valid)
    q11 = q11 if q11 is not None else mean_v
    q21 = q21 if q21 is not None else mean_v
    q12 = q12 if q12 is not None else mean_v
    q22 = q22 if q22 is not None else mean_v

    # Bilinear interpolation
    if lat1 != lat0:
        t = (lat - lat0) / (lat1 - lat0)
    else:
        t = 0.0
    if lon1 != lon0:
        u = (lon - lon0) / (lon1 - lon0)
    else:
        u = 0.0

    val = (q11 * (1-t)*(1-u) + q21 * t*(1-u) +
           q12 * (1-t)*u + q22 * t*u)
    return val


class ClimateData:
    """Container for interpolated climate data at each grid cell."""

    def __init__(self, temp_path=None, precip_path=None, wind_path=None):
        self.temp = {}      # T2M: temperature at 2m (°C)
        self.precip = {}    # PRECTOTCORR: precipitation (mm/day)
        self.wind = {}      # WS2M_RANGE: wind speed range (m/s)
        self.available = False

        # Dry season months (May-Oct) for fire risk
        self.dry_months = ['MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT']

        t_data = load_power_csv(temp_path) if temp_path else {}
        p_data = load_power_csv(precip_path) if precip_path else {}
        w_data = load_power_csv(wind_path) if wind_path else {}

        if t_data or p_data or w_data:
            self.available = True
            print(f"  Temperature grid points: {len(t_data)}")
            print(f"  Precipitation grid points: {len(p_data)}")
            print(f"  Wind speed grid points: {len(w_data)}")

        self._t_data = t_data
        self._p_data = p_data
        self._w_data = w_data

    def get(self, lat, lon):
        """Return (temp_C, precip_mm_day, wind_ms) for a cell, using dry-season avg."""
        temp = interpolate_climate(self._t_data, lat, lon, self.dry_months)
        precip = interpolate_climate(self._p_data, lat, lon, self.dry_months)
        wind = interpolate_climate(self._w_data, lat, lon, self.dry_months)
        return temp, precip, wind


# =============================================================================
# 6. LOAD FIRMS FIRE DATA
# =============================================================================

def load_firms(path):
    """Load NASA FIRMS MODIS fire hotspot CSV."""
    fires = []
    try:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    fires.append({
                        'lat': lat, 'lon': lon,
                        'confidence': float(row.get('confidence', 50)),
                        'frp': float(row.get('frp', 0)),
                        'date': row.get('acq_date', ''),
                    })
    except FileNotFoundError:
        print(f"Warning: '{path}' not found.")
    print(f"Loaded {len(fires)} fire detections within Etosha bounds")
    return fires


def bin_fires(fires):
    """Bin fire detections into grid cells, weighted by confidence."""
    counts = defaultdict(float)
    frp_sums = defaultdict(float)
    for f in fires:
        cell = to_cell(f['lat'], f['lon'])
        if cell:
            w = f['confidence'] / 100.0
            counts[cell] += w
            frp_sums[cell] += f['frp'] * w
    return counts, frp_sums


# =============================================================================
# 7. NDVI FILE DISCOVERY
# =============================================================================

def find_ndvi_files():
    """
    Search for NDVI GeoTiffs and return (dry_season_files, all_files).
    Dry season = May–Oct; used for fire fuel load estimation.
    """
    search_dirs = ['.', './data', '../data']
    all_ndvi = []

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if 'NDVI' in f.upper() and f.endswith('.tif') and 'QUALITY' not in f.upper():
                all_ndvi.append(os.path.join(d, f))

    if not all_ndvi:
        return [], []

    # Categorize into dry-season (May–Oct) vs all
    dry = []
    for path in all_ndvi:
        fname_lower = os.path.basename(path).lower()
        for kw in DRY_SEASON_KEYWORDS:
            if kw in fname_lower:
                dry.append(path)
                break

    return sorted(dry), sorted(all_ndvi)


# =============================================================================
# 8. PREPROCESS ORCHESTRATOR
# =============================================================================

def preprocess():
    """
    Load and prepare all data for the model.

    Returns a dict with:
        fires, counts, frps, T, get_ndvi, use_ndvi, climate,
        plus grid info (lat_centers, lon_centers, n_rows, n_cols)
    """
    print("=" * 60)
    print("ETOSHA NATIONAL PARK — PROTECTEDNESS MODEL")
    print("=" * 60)
    print(f"\nGrid: {n_rows} rows × {n_cols} cols = {n_rows * n_cols} cells")
    print(f"Cell size: {CELL_SIZE_KM}km × {CELL_SIZE_KM}km = {CELL_AREA_KM2} km²")

    # --- Load fire data ---
    fires = load_firms("FIRMS_dataset.csv")

    if not fires:
        print("\n⚠  No fire data loaded — using synthetic data for testing.")
        np.random.seed(42)
        fires = [{'lat': np.random.uniform(LAT_MIN, LAT_MAX),
                  'lon': np.random.uniform(LON_MIN, LON_MAX),
                  'confidence': np.random.uniform(30, 100),
                  'frp': np.random.uniform(5, 200),
                  'date': '2020-06-15'} for _ in range(500)]

    counts, frps = bin_fires(fires)

    # Time span
    dates = sorted(set(f['date'] for f in fires if f['date']))
    if len(dates) >= 2:
        d0 = datetime.strptime(dates[0], '%Y-%m-%d')
        d1 = datetime.strptime(dates[-1], '%Y-%m-%d')
        T = max((d1 - d0).days / 365.25, 1.0)
    else:
        T = 10.0
    print(f"Fire data spans ~{T:.1f} years ({len(dates)} unique dates)")

    # --- Load NDVI ---
    dry_files, all_files = find_ndvi_files()
    get_ndvi = None
    use_ndvi = False

    if dry_files:
        print(f"\nLoading dry-season NDVI ({len(dry_files)} files: May–Oct)...")
        _, get_ndvi = load_ndvi_multi(dry_files, label="Dry-season NDVI")
        if get_ndvi:
            use_ndvi = True
            print("✓ Dry-season NDVI composite loaded — best for fire fuel estimation")
    elif all_files:
        print(f"\nNo dry-season NDVI found. Loading all available ({len(all_files)} files)...")
        _, get_ndvi = load_ndvi_multi(all_files, label="All-month NDVI")
        if get_ndvi:
            use_ndvi = True
            print("✓ Annual NDVI composite loaded")

    if not use_ndvi:
        print("\n⚠  No NDVI GeoTiff found. Using default vegetation density (0.5).")
        print("   Place NDVI .tif files in this directory and re-run.")

    # --- Load Climate Data ---
    print("\nLoading NASA POWER climate data...")

    def find_power_csv(keyword):
        """Find POWER CSV containing a specific parameter."""
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
    if climate.available:
        print("✓ Climate data loaded (dry-season averages for fire risk)")
    else:
        print("⚠  No climate CSVs found. Fire risk will use fallback formula.")
        print("   Place POWER Regional Monthly CSVs in this directory.")

    return {
        'fires': fires,
        'counts': counts,
        'frps': frps,
        'T': T,
        'get_ndvi': get_ndvi,
        'use_ndvi': use_ndvi,
        'climate': climate,
    }
