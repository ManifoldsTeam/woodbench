# Task 11: inspection of specimens 3384.4 and 3393.1 (06-11-2026)

Supervisor request 3 (06-11-2026): look at specimens 3384.4 (*P. krempfii*)
and 3393.1 (*P. dalatensis*), the ones the models misclassify most, for
mislabels, blur or anomalies.

Script: `notebooks/phase 2/supervisor-feedback/task_11_specimen_inspection/inspect_specimens.py`
(run from `woodbench/`, ~1 min). Data: 17-species `data/task-B-group-split`.
Errors come from the 25 CV v2 held-out prediction files
(`../task_01_fivefold_cv_training/results-v2/*/fold_*/predictions.csv`). Each
origin is held out exactly once, so every image has 5 predictions (one per model).
No model was run for this task.

## Errors

| Specimen | Images | CV v2 error | Images wrong for >=3/5 models | Predicted as |
|---|---:|---:|---:|---|
| *P. dalatensis* 3393.1 (fold 1) | 28 | 32.1% (45/140) | 12 | *P. krempfii* 39, *Pseudotsuga* 4, *P. cernua* 2 |
| *P. krempfii* 3384.4 (fold 5) | 36 | 13.9% (25/180) | 5 | *P. dalatensis* 24, *P. radiata* 1 |

The two specimens are confused mainly with each other. Other origins of the same two
species: 0-14.4% error (`origin_table.csv`).

Errors per photo number (`number:models wrong`):

- 3384.4: 102-127 all 0; **128:5 129:4 130:5 131:4 132:1 133:4**; 134-138: 0 or 1.
  The errors are a block of consecutive photos, i.e. one area of the specimen.
- 3393.1: 1-7 all 0 except 3:1; then 8:3 9:5 10:3 12:4 13:5 18:3 20:3 22:3
  23:5 24:3 27:3 28:4. The errors are spread over the second half of the series.

## Checks

| Check | 3384.4 | 3393.1 |
|---|---|---|
| Blur (variance of Laplacian) | No. Median 500, the highest of all *P. krempfii* origins (325-460). Wrong vs correct images: 488 vs 500 | No. Median 903, in the upper half of *P. dalatensis* (296-1121). Wrong vs correct: 891 vs 920 |
| Brightness / contrast | Within the species range; contrast slightly the highest | Within the species range |
| Colour | Wrong photos are less saturated (median 138 vs 147), visibly more pink-grey | Wrong photos are more saturated (159 vs 148) |
| Glare | None | 2 photos have a bright spot in a corner (1.2% of the frame). Both were classified correctly |
| Image size | 2560 x 1899-1944, like every *P. krempfii* origin | 2560 x 2048, like every *P. dalatensis* origin |
| Photo shared with the other species (ORB + RANSAC, as in task 10) | No. Max 5 inliers against any *P. dalatensis* photo | No. Max 4 inliers against any *P. krempfii* photo |
| Photo shared with own species | **Yes**: 14 photos overlap photos of 3384 (up to 86 inliers) and 3384.5 (78). Both origins are in training when 3384.4 is held out (folds 3 and 1 vs 5) | No |

## Reading

- **3384.4: no sign of a mislabel, blur or a broken photo.** 30 of 36 photos
  are classified correctly. The errors sit in one area (photos 128-133) whose
  tissue looks more pink-grey than the rest. The photos that overlap 3384 and
  3384.5 are all classified correctly, while the 6 wrong photos have no overlap.
  The training set may therefore help 3384.4 through shared surfaces, which
  would make its 13.9% error an underestimate rather than an overestimate. This
  is a correlation, not a proof.
- **3393.1: the only candidate for a label check.** Errors cover much of the
  specimen, almost always towards *P. krempfii*. The photos are sharp, normally
  exposed, and use the *P. dalatensis* camera format. No photo is a copy of a
  *P. krempfii* photo, so this is not a file mix-up. Whether the wood itself is
  *P. krempfii* cannot be decided from the images. **The physical specimen 3393.1
  needs to be checked by the wood anatomist or the collection team.**
- Side note: the two species were photographed differently. *P. krempfii* images
  are cropped to 2560 x ~1909, *P. dalatensis* images are 2560 x 2048 and on
  average darker (brightness about 142 vs 148). After the square 224 resize, the
  aspect difference is about 7%. This does not explain the errors (3393.1 has
  the *P. dalatensis* format and is still called *P. krempfii*), but it is a
  possible shortcut between these two species.

## Files

| File | Content |
|---|---|
| `image_table.csv` | All 638 images of both species: origin, split, models_wrong, wrong predictions with confidence, size, sharpness, brightness, contrast, hue, saturation, glare_pct |
| `origin_table.csv` | Same per origin (medians, CV error rate) |
| `overlap_pairs.csv` | Pairs of a specimen photo and another photo with >= 15 inliers |
| `origin_statistics.png` | Box plots per origin (red = inspected specimen) |
| `sheet_3384.4.jpg`, `sheet_3393.1.jpg` | All photos of the specimen sorted by errors, 16 photos of other origins of the same species, 16 photos of the predicted species |
| `summary.json` | Specimen rows and max overlap per origin |
