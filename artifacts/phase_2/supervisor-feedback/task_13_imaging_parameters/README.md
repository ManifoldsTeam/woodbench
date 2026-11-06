# Task 13: imaging parameters for Methods (06-11-2026)

Supervisor request 5 (06-11-2026): imaging parameters for the Methods section.

Script: `notebooks/phase 2/supervisor-feedback/task_13_imaging_parameters/imaging_parameters.py`
writes `imaging_by_species.csv` (needs `openpyxl`, installed in env `wood` on 06-11-2026). It combines what the image files contain with
`docs/references/S2_Metadata.xlsx` (merged cells expanded), for the 17
benchmark species in `data/task-B-group-split`.

## 1. What can be stated, and the source

| Item | Value | Source | Status |
|---|---|---|---|
| Section plane | Transverse (cross) section, macroscopic | All project docs; visible in the images | Consistent |
| Magnification | 50x | `WoodID_plan.md`, Assignment S2, `docs/reports/phase_0_report.md`, paper macro `\Magnification` | Stated in docs only; not in image files or metadata |
| Camera / optics | Olympus UC30 digital camera on a stereomicroscope | `WoodID/sections/03_methods.tex` only | **Unverified**; the metadata `Devices` column is empty; see section 2 |
| File format | PNG, 8-bit RGB, no EXIF | Image files | Verified |
| Pixel dimensions | 1280 x 1024: *Callitris*, *Cunninghamia*, *Fokienia* (685 images). 2560 x 2048 / 1920 / ~1909 (1856-1944, cropped): other 14 species | Image files | Verified (`sizes` column) |
| PNG dpi tag | 240 (1280 x 1024 group), 300 or none (2560 group) | Image files | Written by the saving software; **not** a physical scale |
| Capture batches | 2023 (Aug-Sep): the three 1280 x 1024 species, captured at the timber stores (Dong Ky, Hai Phong). 2024 (Oct-Dec): the 14 other species, captured at RIFI | Metadata `Collected period`, `Captured place` | From metadata |
| Specimens | 10 sample IDs per species (9 for *Picea glauca*, *Pinus radiata*) | Metadata `Sample ID` | From metadata |
| Moisture state | 20-50% for all 17 species | Metadata `Humidity state` | From metadata; method of measurement unknown |
| Illumination | "natural light": 10 species; "artificial light": 7 species (*Calocedrus*, *Chamaecyparis*, *P. krempfii*, *P. radiata*, *Taxus*, *Thujopsis*, *Xanthocyparis*) | Metadata `Light conditions` | From metadata; set per species, not per image |
| Specimen origin | Timber stores (Dong Ky, Hai Phong) for the 2023 batch; Can Tho, Ho Chi Minh City, Ha Noi, Binh Duong for the 2024 batch | Metadata `Originin` | From metadata |

## 2. Inconsistencies to resolve before writing

1. **Camera vs pixel size.** The Olympus UC30 is a 2080 x 1544 px CCD camera
   (4:3, 3.45 um pixels, discontinued). No image has that size. 2560 x 2048 and 1280 x
   1024 are 5:4, which a 4:3 sensor can only give after cropping or resampling.
   2560 x 1920 is 4:3 but larger than the sensor. Either the export software
   resized the images, or another camera was used for some batches. A power-spectrum
   check was not conclusive. Only the cropped 2560 x ~1909 group is clearly softer
   (consistent with an extra resampling step; see also task 11 sharpness).
2. **No physical scale.** No file or document gives um/pixel or the field of
   view, so a scale bar (Fig. 1) or a statement like "each image covers X x Y mm"
   cannot be written yet. With two pixel formats, the scale probably differs
   between the 2023 and 2024 batches.
3. **Origin wording.** The current paper draft says the specimens come from "reference xylarium
   collections preserved at RIFI". The metadata says the 2023 batch was collected
   and photographed at commercial timber stores.
4. **Illumination is confounded with species.** Light condition is recorded per
   species. A model can partly learn the lighting instead of the wood. This fits
   task 11 (*P. dalatensis*, natural light, darker; *P. krempfii*, artificial light,
   brighter) and is worth one sentence in Limitations.
5. **Image counts.** The metadata lists more images than the benchmark for several
   species. The gap is largest for the 2023 batch: *Callitris* 388 vs 214,
   *Cunninghamia* 410 vs 186, *Fokienia* 400 vs 285. Also *Pseudotsuga* 289 vs 235 and
   *Taxus* 295 vs 223. This task did not trace where the images were dropped.
6. **File numbers vs metadata ranges.** The metadata assigns image ranges to each
   specimen, e.g. *Cunninghamia* 3274.1 = images 1-42. The dataset files of the three 2023
   species are renumbered per split (*Cunninghamia*: 186 files, 86 distinct numbers,
   10-185), so these ranges cannot be applied. A mapping from the original file
   names would give real specimen IDs for these species (compare task 10).

## 3. Questions for the data-collection team

1. Camera model and microscope model for each batch (2023 on site, 2024 at RIFI).
   Was it really an Olympus UC30?
2. Objective / zoom setting, and the resulting um per pixel. A photo of a stage
   micrometer or ruler at the same setting is enough.
3. How were the images exported (software, resizing, cropping)? Why are some 2024
   species 2560 x ~1909 instead of 2560 x 2048?
4. Surface preparation of the cross sections (knife cut, sanding grits, polishing).
5. Light source and set-up for "natural" and "artificial" light (ring light,
   oblique LED, window light?).
6. How was the moisture state (20-50%) measured?
7. The original file names of *Callitris*, *Cunninghamia*, *Fokienia* images, so that
   the metadata image ranges can be mapped to specimens.

## 4. Draft Methods text (only facts with a source; [brackets] = still missing)

> Macroscopic images of transverse surfaces were captured at 50x magnification
> with a [camera model] mounted on a [stereomicroscope model]. The transverse surfaces were
> prepared by [surface preparation]. Specimens of three species (*Callitris
> columellaris*, *Cunninghamia lanceolata*, *Fokienia hodginsii*) were collected from timber
> stores in Vietnam and photographed on site in August-September 2023
> (1280 x 1024 pixels). The remaining 14 species were photographed at the Research
> Institute of Forest Industry (RIFI) between October and December 2024
> (2560 pixels wide, 1909-2048 pixels high). Each species is represented by 9-10
> specimens, at a moisture content of 20-50%. Illumination was [light source];
> [natural / artificial light] was used depending on the batch. Images were stored as 8-bit
> RGB PNG files, giving a pixel size of [um/pixel] ([field of view] mm).

## Files

| File | Content |
|---|---|
| `imaging_by_species.csv` | Per species: image counts (dataset / metadata), pixel sizes, PNG mode, dpi tag, collection date, origin, humidity, capture place, light, capturer, device |
