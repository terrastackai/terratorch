"""
Prepare the xView2 dataset for training TerraMind with TerraTorch.

This script converts the raw xView2 tier-1 GeoTIFF + JSON dataset into a
TerraTorch-compatible layout:

  <output_dir>/
    images/   – 6-band int16 GeoTIFFs: bands 0-2 from pre-event, 3-5 from post-event
    masks/    – int16 GeoTIFFs: 0=background, 1=no-damage, 2=minor-damage,
                 3=major-damage, 4=destroyed, -1=un-classified (ignored in loss)
    splits/
      train.txt / val.txt / test.txt  – bare stems, one per line

Usage
-----
  python prepare_xview2.py \\
      --src  /dccstor/geofm-datasets/datasets/xview2/geotiffs/tier1 \\
      --dst  /dccstor/geofm-datasets/datasets/xview2_terratorch \\
      --seed 42 \\
      --train-ratio 0.70 --val-ratio 0.15 \\
      --compute-stats

The --compute-stats flag prints per-band means and stds (over the training
split) that you can paste directly into the YAML training config.
"""

import argparse
import json
import math
import pathlib
import random

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.transform import Affine
from shapely import wkt as shapely_wkt

# ─── Damage class map ────────────────────────────────────────────────────────
# -1 means "ignore" in TerraTorch loss (no_label_replace=-1)
SUBTYPE_TO_CLASS: dict[str, int] = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": -1,  # handled via two-pass rasterization
}
# Temporary placeholder used during rasterization for un-classified
# (int16 supports -1, but rasterio.features.rasterize works better with a
# positive sentinel that we replace afterwards)
_UNCLASSIFIED_SENTINEL = 255


def rasterize_label(label_json_path: pathlib.Path, height: int, width: int) -> np.ndarray:
    """Return an int16 (H, W) mask rasterized from an xView2 JSON label file.

    Coordinate system: the JSON ``features.xy`` field stores WKT polygons
    whose (x, y) coordinates are already in pixel space – (column, row) with
    sub-pixel precision.  We use an identity affine transform so that rasterio
    maps x→column and y→row directly.
    """
    mask = np.zeros((height, width), dtype=np.int16)

    with open(label_json_path) as fh:
        data = json.load(fh)

    features_xy = data.get("features", {}).get("xy", [])
    if not features_xy:
        return mask  # no buildings – all background

    classified_shapes = []   # (geometry, class_value) for classes 1-4
    unclassified_shapes = [] # (geometry, sentinel) for un-classified

    for feat in features_xy:
        subtype = feat.get("properties", {}).get("subtype", "un-classified")
        class_val = SUBTYPE_TO_CLASS.get(subtype, -1)
        wkt_str = feat.get("wkt", "")
        if not wkt_str:
            continue
        try:
            geom = shapely_wkt.loads(wkt_str)
        except Exception:
            continue
        if class_val == -1:
            unclassified_shapes.append((geom.__geo_interface__, _UNCLASSIFIED_SENTINEL))
        else:
            classified_shapes.append((geom.__geo_interface__, class_val))

    # Identity transform: coordinate (x, y) → pixel (col=x, row=y)
    identity = Affine.identity()

    # Pass 1 – burn un-classified buildings as sentinel (so they appear on mask
    # as "un-classified" unless a classified feature overwrites them)
    if unclassified_shapes:
        rasterio.features.rasterize(
            unclassified_shapes,
            out=mask,
            transform=identity,
            merge_alg=rasterio.features.MergeAlg.replace,
        )

    # Pass 2 – burn classified buildings on top (overwrites sentinel)
    if classified_shapes:
        rasterio.features.rasterize(
            classified_shapes,
            out=mask,
            transform=identity,
            merge_alg=rasterio.features.MergeAlg.replace,
        )

    # Replace sentinel → -1
    mask[mask == _UNCLASSIFIED_SENTINEL] = -1

    return mask


def write_stacked_image(
    pre_path: pathlib.Path,
    post_path: pathlib.Path,
    out_path: pathlib.Path,
    nodata_val: int = -9999,
) -> tuple[int, int]:
    """Stack pre (bands 0-2) and post (bands 3-5) into a 6-band int16 GeoTIFF.

    Returns (height, width) of the written image.
    """
    with rasterio.open(pre_path) as pre_src, rasterio.open(post_path) as post_src:
        pre_data = pre_src.read()   # (3, H, W)
        post_data = post_src.read() # (3, H, W)
        profile = post_src.profile.copy()

    # Guard against mismatched sizes (rare but possible in xView2 edge tiles)
    if pre_data.shape[1:] != post_data.shape[1:]:
        raise ValueError(
            f"Shape mismatch: {pre_path.name} {pre_data.shape[1:]} vs "
            f"{post_path.name} {post_data.shape[1:]}"
        )

    stacked = np.concatenate([pre_data, post_data], axis=0)  # (6, H, W)
    _, height, width = stacked.shape

    profile.update(
        count=6,
        dtype="int16",
        nodata=nodata_val,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)

    return height, width


def write_mask(mask: np.ndarray, out_path: pathlib.Path, reference_path: pathlib.Path) -> None:
    """Write an int16 mask GeoTIFF, borrowing georeference from reference_path."""
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()

    profile.update(
        count=1,
        dtype="int16",
        nodata=-9999,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])  # (1, H, W)


def compute_band_stats(
    image_paths: list[pathlib.Path],
    num_bands: int = 6,
    spatial_stride: int = 16,
) -> tuple[list[float], list[float]]:
    """Compute per-band mean and std over a list of images via spatial subsampling."""
    print(f"\nComputing band statistics over {len(image_paths)} training images "
          f"(spatial stride={spatial_stride}) …")
    n = np.zeros(num_bands, dtype=np.float64)
    s1 = np.zeros(num_bands, dtype=np.float64)  # sum
    s2 = np.zeros(num_bands, dtype=np.float64)  # sum of squares

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(image_paths)}")
        with rasterio.open(img_path) as src:
            data = src.read(
                out_dtype=np.float32,
            )  # (6, H, W)
            nodata = src.nodata

        # Subsample spatially
        data = data[:, ::spatial_stride, ::spatial_stride]  # (6, h', w')

        # Mask nodata
        if nodata is not None:
            valid = data != nodata
        else:
            valid = np.ones(data.shape, dtype=bool)

        for b in range(num_bands):
            vals = data[b][valid[b]].astype(np.float64)
            n[b] += vals.size
            s1[b] += vals.sum()
            s2[b] += (vals ** 2).sum()

    means = (s1 / n).tolist()
    stds = (np.sqrt(s2 / n - (s1 / n) ** 2)).tolist()
    return means, stds


def process_dataset(
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    compute_stats: bool,
) -> None:
    images_in = src_dir / "images"
    labels_in = src_dir / "labels"
    images_out = dst_dir / "images"
    masks_out = dst_dir / "masks"
    splits_out = dst_dir / "splits"

    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    # ── Discover valid pairs ──────────────────────────────────────────────────
    pre_files = sorted(images_in.glob("*_pre_disaster.tif"))
    print(f"Found {len(pre_files)} pre-disaster images in {images_in}")

    valid_stems: list[str] = []
    skipped = 0

    for pre_path in pre_files:
        stem = pre_path.stem.replace("_pre_disaster", "")
        post_path = images_in / f"{stem}_post_disaster.tif"
        label_path = labels_in / f"{stem}_post_disaster.json"

        if not post_path.exists():
            print(f"  SKIP (no post image): {stem}")
            skipped += 1
            continue
        if not label_path.exists():
            print(f"  SKIP (no label JSON): {stem}")
            skipped += 1
            continue

        valid_stems.append(stem)

    print(f"Valid pairs: {len(valid_stems)}  |  Skipped: {skipped}")

    # ── Generate splits ───────────────────────────────────────────────────────
    rng = random.Random(seed)
    shuffled = valid_stems[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = math.floor(n_total * train_ratio)
    n_val = math.floor(n_total * val_ratio)
    # remaining all go to test
    train_stems = shuffled[:n_train]
    val_stems = shuffled[n_train: n_train + n_val]
    test_stems = shuffled[n_train + n_val:]

    print(f"Splits → train: {len(train_stems)}, val: {len(val_stems)}, test: {len(test_stems)}")

    for split_name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        split_file = splits_out / f"{split_name}.txt"
        with open(split_file, "w") as fh:
            fh.write("\n".join(stems) + "\n")
        print(f"  Wrote {split_file}")

    # ── Process each pair ─────────────────────────────────────────────────────
    processed = 0
    errors = 0

    for i, stem in enumerate(valid_stems):
        img_out = images_out / f"{stem}.tif"
        mask_out = masks_out / f"{stem}.tif"

        # Idempotent: skip if both outputs already exist
        if img_out.exists() and mask_out.exists():
            continue

        pre_path = images_in / f"{stem}_pre_disaster.tif"
        post_path = images_in / f"{stem}_post_disaster.tif"
        label_path = labels_in / f"{stem}_post_disaster.json"

        try:
            # Stack pre + post → 6-band image
            height, width = write_stacked_image(pre_path, post_path, img_out)

            # Rasterize polygon annotations → int16 mask
            mask = rasterize_label(label_path, height, width)
            write_mask(mask, mask_out, post_path)

            processed += 1
        except Exception as exc:
            print(f"  ERROR [{stem}]: {exc}")
            errors += 1
            # Remove partial output to stay consistent
            img_out.unlink(missing_ok=True)
            mask_out.unlink(missing_ok=True)
            continue

        if (processed + errors) % 200 == 0:
            print(f"  Processed {processed + errors}/{len(valid_stems)} pairs …")

    print(f"\nDone.  Processed: {processed}  |  Errors: {errors}")

    # ── Compute statistics ────────────────────────────────────────────────────
    if compute_stats:
        train_img_paths = [images_out / f"{s}.tif" for s in train_stems
                           if (images_out / f"{s}.tif").exists()]
        means, stds = compute_band_stats(train_img_paths)

        band_labels = [
            "pre_B0 (pre-red)", "pre_B1 (pre-green)", "pre_B2 (pre-blue)",
            "post_B3 (post-red)", "post_B4 (post-green)", "post_B5 (post-blue)",
        ]
        print("\n─── Band statistics (copy into terramind_xview2_segmentation.yaml) ───")
        print("means:")
        for label, m in zip(band_labels, means):
            print(f"  - {m:.4f}  # {label}")
        print("stds:")
        for label, s in zip(band_labels, stds):
            print(f"  - {s:.4f}  # {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src",
        type=pathlib.Path,
        default=pathlib.Path("/dccstor/geofm-datasets/datasets/xview2/geotiffs/tier1"),
        help="Root of the raw xView2 tier-1 dataset (contains images/ and labels/).",
    )
    parser.add_argument(
        "--dst",
        type=pathlib.Path,
        default=pathlib.Path("/dccstor/geofm-datasets/datasets/xview2_terratorch"),
        help="Output root for the TerraTorch-compatible dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="After processing, compute per-band means/stds from the training split.",
    )
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    print(f"Source : {args.src}")
    print(f"Output : {args.dst}")
    print(f"Split  : {args.train_ratio:.0%} / {args.val_ratio:.0%} / {test_ratio:.0%}  (seed={args.seed})")

    process_dataset(
        src_dir=args.src,
        dst_dir=args.dst,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        compute_stats=args.compute_stats,
    )


if __name__ == "__main__":
    main()
