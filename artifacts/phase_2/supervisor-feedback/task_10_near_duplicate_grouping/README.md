# Task 10: near-duplicate groups for Callitris, Cunninghamia, Fokienia (06-11-2026)

Supervisor request 2 (06-11-2026): pHash for the three species without specimen
IDs, grouping near-duplicate photos into an image -> group table for the
data-collection team. This is a helper for the collection team, not a paper
result.

Script: `notebooks/phase 2/supervisor-feedback/task_10_near_duplicate_grouping/near_duplicate_groups.py`
(run from `woodbench/`, ~2 min on 16 CPU cores). Data: `data/task-B-group-split`, 17 species
folder, only these three species are read (685 images, all 1280x1024).

## Method

pHash alone found nothing: the closest within-species pairs are 16 / 14 / 10
bits, while different species can already be 10 bits apart. The photos are
mostly overlapping frames of the same wood surface (shifted, sometimes rotated
90 degrees), which pHash does not detect. So every within-species pair (80,466)
is also matched with ORB keypoints + RANSAC similarity transform.

- Link: pHash distance <= 9, or >= 30 RANSAC inliers with scale 0.9-1.1.
- Group: connected components of the links (frames can chain along a specimen).
- Check of the threshold: 30,000 random cross-species pairs never exceeded 12
  inliers; pairs with 30-40 inliers that were checked by eye were real overlaps
  (e.g. Callitris 112 / 113, rotated 90 degrees). In the end no link came from pHash.

## Result

| Species | Images | Groups | Groups with >1 image | Images in those groups | Largest group |
|---|---:|---:|---:|---:|---:|
| *Callitris columellaris* | 214 | 35 | 15 | 194 | 61 |
| *Cunninghamia lanceolata* | 186 | 100 | 40 | 126 | 10 |
| *Fokienia hodginsii* | 285 | 78 | 42 | 249 | 22 |

Consecutive file numbers usually fall in the same group (e.g. pơ mu102-104,
pơ mu73-80), which fits "several photos per specimen". pơmu46 / pơmu47, which
pHash ranked closest, are the same surface shifted (928 inliers), group FOKI-010.

Caveats for the team:

- A group is "photos that overlap, directly or through a chain", not a verified
  specimen. Two specimens are merged if any single link is wrong; large groups
  (Callitris 61, 29, 22) are the ones to check first. `weakest_link_inliers`
  near 30 means the group hangs on a borderline link.
- Fokienia has two file-name schemes, `pơ mu<N>` (greyish photos) and
  `pơmu<N>` (orange photos), which look like two photo sessions. 13 groups mix
  both schemes; they are marked `check_by_eye = True`.
- Groups cross train/val/test in 67 of 97 multi-image groups.

## Files

| File | Content |
|---|---|
| `image_groups.csv` | One row per image: species, split, file_name, path, source_filename, origin_key, md5, phash, **group_id**, group_size, weakest_link_inliers, closest_image (+ inliers, pHash distance), name_scheme, check_by_eye |
| `pair_scores.csv` | Every within-species pair with >= 10 inliers or linked: inliers, pHash distance, scale, rotation, linked |
| `groups_<genus>.jpg` | One row per multi-image group (first 8 images shown) |
| `summary.json` | Rule, ORB parameters, counts |

File names are NFC-normalized (the Fokienia folder stores them as NFD).
