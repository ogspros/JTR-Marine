# Data

No seismic data are distributed with this repository. This folder only
defines where the code expects to find them:

```
data/
├── raw/        # SEG-Y files of the field lines and of Marmousi (you provide them)
├── marmousi/   # original Marmousi files from Madagascar (marmrefl.hh), before conversion
└── maps/       # navigation, ETOPO 2022 bathymetry and Natural Earth layers for Fig. 1
```

Everything placed in these folders is ignored by git.

The SEG-Y folder can be moved anywhere by setting the `DATA_ROOT`
environment variable (default: `./data/raw`):

```bash
export DATA_ROOT=/path/to/your/segy/files
```

If `DATA_ROOT` is not set, `SEISMIC_DATA_ROOT` is also accepted (files are
then read from `$SEISMIC_DATA_ROOT/raw/`).

## Expected files in `data/raw/`

The file names are hard-coded in `config.py` (training pool) and in the
cross-survey / Marmousi evaluation scripts. Use exactly these names, or
edit the paths in the code.

| Expected file                | Line in the paper | Streamer | Role                                      |
|------------------------------|-------------------|----------|-------------------------------------------|
| `MEPA_SP_200_500.sgy`        | SISOC-MEPA        | 1500 m   | training pool                             |
| `GC01_SP1100_1400.sgy`       | CG-1              | 1500 m   | training pool                             |
| `IT065_SP1000_1300.sgy`      | IT90-65           | 3000 m   | training pool                             |
| `IT89_33_SP600_900.sgy`      | IT89-33           | 3000 m   | training pool                             |
| `M31_SP250_550.sgy`          | CROP-M31          | 4500 m   | training pool                             |
| `M32_SP500_800.sgy`          | CROP-M32          | 4500 m   | training pool                             |
| `CG03_SP150_450.sgy`         | CG-3              | 1500 m   | cross-survey (held out)                   |
| `IT245_SP_800_1100.sgy`      | IT97-245          | 3000 m   | cross-survey (held out; shots on the shelf) |
| `CROP_M18_SP_500_800.sgy`    | CROP-M18          | 4500 m   | cross-survey (held out)                   |
| `MARMOUSI_1500.sgy`          | Marmousi          | 1500 m   | synthetic test                            |
| `MARMOUSI_full.sgy`          | Marmousi          | 2575 m   | synthetic test                            |

Each field file is a segment of 301 consecutive shot points of the
original line; the shot-point range is encoded in the file name
(e.g. `M31_SP250_550` = CROP-M31, shot points 250 to 550).

## How to obtain the field data

The sources are those listed in the *Data and code availability* section
of the paper.

| Lines                         | Source |
|-------------------------------|--------|
| CG-1, CG-3                    | Mendeley Data, <https://data.mendeley.com/datasets/dm76kg3xmf> |
| SISOC-MEPA                    | Mendeley Data, <https://data.mendeley.com/datasets/xgv9nd49xg> |
| IT89-33, IT90-65, IT97-245    | Antarctic Seismic Data Library System (SDLS), <https://www.sdls.ogs.it> (registration required) |
| CROP-M18, CROP-M31, CROP-M32  | CROP project, <https://www.crop.cnr.it> (data request required) |

The SDLS and CROP data cannot be redistributed here.

### Extracting the shot-point segments

The full lines must be cut to the shot-point ranges in the table above,
keeping the shot gathers in their original order. Any SEG-Y tool can be
used; for example with Seismic Unix:

```bash
# keep shot points 250..550 of CROP-M31 (adjust the header key to the one
# that stores the shot-point number in your copy of the line)
suwind < M31_full.su key=fldr min=250 max=550 | segyhdrs | segywrite tape=M31_SP250_550.sgy
```

## Expected SEG-Y format

The loader (`jtr_marine/data/dataset.py`) reads only:

* the trace samples and the sample interval from the binary header;
* trace header `FieldRecord` (bytes 9-12) to group traces into shot gathers;
* trace header `offset` (bytes 37-40); its absolute value is used, so the
  sign convention does not matter.

Source/receiver coordinates are not used. Sample interval and record length
may differ between lines: every gather is resampled to the common grid
(2 ms, 4.5 s, 25 m offset spacing, 0-4500 m). Lines recorded at 12.5 m are
mapped to the 25 m grid by taking one trace every two, without spatial
anti-alias filtering (see *Known behaviours* in the main README).

## Marmousi

The synthetic test uses the original Marmousi shot records (Brougois et al.,
1990, doi:[10.3997/2214-4609.201411190](https://doi.org/10.3997/2214-4609.201411190)),
distributed with Madagascar as `marmrefl.hh`. Put `marmrefl.hh` in
`data/marmousi/` and convert it to SEG-Y with
`scripts/data_prep/marmousi_to_segy.py`, which writes the files listed above
into `data/raw/`.

## Map of Fig. 1

`scripts/figures/make_fig1_map.py` reads everything from `data/maps/`
(or `$MAPS_ROOT`) and needs no network access:

```
data/maps/
├── survey_tracks.geojson   track geometry of the nine lines (in the repository)
├── bathymetry/   med.npz  ross.npz  drake.npz                      (download)
├── basemap/      ne_10m_land/  ne_50m_land/  ne_50m_antarctic_ice_shelves_polys/  (download)
└── navigation/   GC_01_Nav_UTM32.txt  GC_03_Nav_UTM32.txt  CROPM31.NAV
                  IT89-33.txt  IT90AR65.uko  crop.kml               (optional, not public)
```

* **bathymetry/** and **basemap/** are public and are downloaded by

  ```bash
  python scripts/data_prep/download_map_data.py
  ```

  Bathymetry: NOAA ETOPO 2022 Ice Surface, 1 arc-minute
  (doi:[10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74)), regional
  subsets (Mediterranean 35-47 N / 7-21 E; Ross Sea 82-68 S / 154-202 E;
  Drake Passage 60-53 S / 73-63 W) from the NOAA ERDDAP server, depths
  relative to EGM2008. Land and ice shelves: Natural Earth 1:10M and 1:50M
  (<https://www.naturalearthdata.com>, public domain).

* **survey_tracks.geojson** is distributed with the repository: it holds
  the track geometry (longitude/latitude) of the nine lines as drawn in
  Fig. 1, without shot-point numbers. With it, the figure is reproduced
  exactly without the original navigation.
* **navigation/** (optional) holds the original navigation files supplied
  with the field data (CG-1, CG-3, CROP-M31, IT89-33, IT90-65) and the CROP
  line geometry (`crop.kml`, used for CROP-M18 and CROP-M32). They come from
  the same sources as the seismic data and are not redistributed here. When
  the folder exists, the script reads the tracks from it and rewrites
  `survey_tracks.geojson` and the navigation tables under `results/map/`.
  SISOC-MEPA and IT97-245 are drawn as geodesic connectors between their
  end points, which are written in the script.

The assumptions made when reading the navigation (coordinate scaling and
datum of the CG files, antenna positions of IT89-33, KML geometry for
CROP-M18/M32) are listed in the script docstring and in the
`navigation_summary.csv` it writes next to the figure.
