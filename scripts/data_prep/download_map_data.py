"""
download_map_data.py -- download the public layers used by the map of Fig. 1.

Writes into MAPS_ROOT (default ./data/maps):

  bathymetry/med.npz, ross.npz, drake.npz
      NOAA ETOPO 2022 Ice Surface, 60 arc-second grid (doi:10.25921/fd45-gt74),
      subsets served by the NOAA ERDDAP of the PIFSC OceanWatch node:
        med    35..47 N,    7..21 E
        ross   82..68 S,  154..202 E   (longitudes kept in 0..360)
        drake  60..53 S,  287..297 E   (converted to -73..-63)
      Each .npz holds 'longitude', 'latitude' (ascending) and 'elevation'
      (m, positive up, EGM2008), without interpolation.

  basemap/ne_10m_land, ne_50m_land, ne_50m_antarctic_ice_shelves_polys
      Natural Earth vectors (public domain).

The navigation files of the seismic lines are not public; see
data/README.md.

Usage:
  python scripts/data_prep/download_map_data.py
"""
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import MAPS_ROOT

ERDDAP = "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/ETOPO_2022_v1_60s.nc"
# name: (lat_min, lat_max, lon_min, lon_max) in the 0..360 convention of the server
SUBSETS = {
    "med":   (35, 47, 7, 21),
    "ross":  (-82, -68, 154, 202),
    "drake": (-60, -53, 287, 297),
}
NATURAL_EARTH = {
    "ne_10m_land": "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
    "ne_50m_land": "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip",
    "ne_50m_antarctic_ice_shelves_polys":
        "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_antarctic_ice_shelves_polys.zip",
}


def fetch(url: str) -> bytes:
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def etopo_url(lat0, lat1, lon0, lon1) -> str:
    return f"{ERDDAP}?z%5B%28{lat0}%29%3A1%3A%28{lat1}%29%5D%5B%28{lon0}%29%3A1%3A%28{lon1}%29%5D"


def netcdf_to_npz(raw: bytes, out: Path, wrap_west: bool) -> None:
    """Convert an ERDDAP NetCDF-3 subset to the .npz layout used by make_fig1_map.py."""
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(raw)
        tmp.flush()
        with netcdf_file(tmp.name, "r", mmap=False) as nc:
            lat = np.array(nc.variables["latitude"][:], dtype=np.float64)
            lon = np.array(nc.variables["longitude"][:], dtype=np.float64)
            z = np.array(nc.variables["z"][:], dtype=np.float32)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1, :]
    if wrap_west:
        lon = lon - 360.0
    np.savez(out, longitude=lon, latitude=lat, elevation=z)
    print(f"  wrote {out}  ({len(lat)} x {len(lon)})")


def main() -> None:
    bathy_dir = MAPS_ROOT / "bathymetry"
    base_dir = MAPS_ROOT / "basemap"
    bathy_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("ETOPO 2022 subsets:")
    for name, (lat0, lat1, lon0, lon1) in SUBSETS.items():
        out = bathy_dir / f"{name}.npz"
        if out.exists():
            print(f"  {out} exists, skipped")
            continue
        netcdf_to_npz(fetch(etopo_url(lat0, lat1, lon0, lon1)), out,
                      wrap_west=(lon0 > 180 and name == "drake"))

    print("Natural Earth:")
    for name, url in NATURAL_EARTH.items():
        target = base_dir / name
        if (target / f"{name}.shp").exists():
            print(f"  {target} exists, skipped")
            continue
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(url))) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                # flatten: keep only the file name (no paths from the archive)
                (target / Path(member).name).write_bytes(zf.read(member))
        print(f"  extracted {target}")

    print(f"Done. Map layers are in {MAPS_ROOT}")


if __name__ == "__main__":
    main()
