# Task 12: Fig. 1, one representative image per species (06-11-2026)

Supervisor request 4 (06-11-2026): Fig. 1, 17 tiles, one representative image
per species, species names in italics.

Script: `notebooks/phase 2/supervisor-feedback/task_12_figure1_species_grid/figure1_species_grid.py`
(run from `woodbench/` with `HF_HUB_OFFLINE=1`, ~1 min, GPU optional).

## Figure

- `fig1_species_grid.pdf`: 178 mm (7 in) wide, full page width. Text is
  vector (Liberation Sans Italic 7 pt, embedded), photos are at 600 dpi.
- `fig1_species_grid.png`: same figure at 300 dpi (2100 x 1095 px).
- Layout is 6 + 6 + 5 tiles, with the last row centred, in alphabetical order (as in
  `class_to_idx`). Each tile is a centre crop at 5:4 of the full photo. Genus and
  epithet sit on two lines so the longest name (*Xanthocyparis vietnamensis*) fits.
- No scale bar. The pixel size is not known yet (task 5, imaging
  parameters). Note that *Callitris*, *Cunninghamia* and *Fokienia* were taken at
  1280 x 1024 and the other 14 species at 2560 x 1909-2048, so a scale bar may
  differ between tiles.

## How the image was chosen

1. Candidates: all 5 CV v2 models classify the photo correctly (held-out
   predictions), no glare (< 0.1% near-white pixels), sharpness (variance of
   Laplacian) >= species median.
2. Ranking: cosine similarity to the species centroid of ImageNet-pretrained
   ConvNeXt-Tiny features (timm `convnext_tiny`, 224 cache). A generic
   backbone is used so the choice does not depend on our trained models. Rank 1 is used.

`candidates.jpg` shows the 8 best candidates per species (rank 1 in red).
`selection_table.csv` lists every image with its rank. To swap a tile:

```zsh
python ".../figure1_species_grid.py" --override "Pinus radiata=train/Pinus radiata/3379. Pinus radiata.3 (100).png"
```

The *P. dalatensis* and *P. krempfii* tiles do not come from specimens 3393.1 /
3384.4 (task 11).

| Species | Image (relative to `data/task-B-group-split`) | Size |
|---|---|---|
| *Araucaria araucana* | `train/Araucaria araucana/3351. Araucaria araucana.5.(146).png` | 2560x2048 |
| *Callitris columellaris* | `val/Callitris columellaris/135__from-test__76165962.png` | 1280x1024 |
| *Calocedrus rupestris* | `test/Calocedrus rupestris/3342. Calocedrus rupestris.7.(185).png` | 2560x1908 |
| *Chamaecyparis obtusa* | `train/Chamaecyparis obtusa/3364. Chamaecyparis obtusa.9 (266).png` | 2560x2048 |
| *Cryptomeria japonica* | `train/Cryptomeria japonica/3350. Cryptomeria japonica.3.(93).png` | 2560x1920 |
| *Cunninghamia lanceolata* | `val/Cunninghamia lanceolata/135__from-val__8efd84e0.png` | 1280x1024 |
| *Fokienia hodginsii* | `train/Fokienia hodginsii/pơ mu41.png` | 1280x1024 |
| *Picea glauca* | `train/Picea glauca/3378. Picea glauca.9 (268).png` | 2560x1908 |
| *Pinus cernua* | `train/Pinus cernua/3385. Pinus cernua.6 (142).png` | 2560x2048 |
| *Pinus dalatensis* | `train/Pinus dalatensis/3393. Pinus dalatensis.4 (126).png` | 2560x2048 |
| *Pinus krempfii* | `train/Pinus krempfii/3384. Pinus krempfii.6 (182).png` | 2560x1909 |
| *Pinus radiata* | `train/Pinus radiata/3379. Pinus radiata.6 (170).png` | 2560x1920 |
| *Podocarpus pilgeri* | `train/Podocarpus pilgeri/3386. Podocarpus pilgeri.5 (161).png` | 2560x2048 |
| *Pseudotsuga brevifolia* | `train/Pseudotsuga brevifolia/3383. Pseudotsuga brevifolia.5 (125).png` | 2560x2048 |
| *Taxus wallichiana* | `train/Taxus wallichiana/3391. Taxus wallichiana.8 (223).png` | 2560x1908 |
| *Thujopsis dolabrata* | `test/Thujopsis dolabrata/3387. Thujopsis dolabrata.9 (280).png` | 2560x2048 |
| *Xanthocyparis vietnamensis* | `train/Xanthocyparis vietnamensis/3341. Xanthocyparis vietnamensis.1.(23).png` | 2560x1909 |

## Note on families

The species-to-family map in `notebooks/phase -1/dataset_exploration.ipynb`
(and `WoodID_plan.md` section 3.2) puts *Araucaria araucana* in Pinaceae. It belongs to
Araucariaceae. The figure does not show families, but any family table or
Task A label taken from that map should be checked.
