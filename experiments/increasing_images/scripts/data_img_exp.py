#!/usr/bin/env python3

from __future__ import annotations
import os, sys, shutil, random, yaml, logging
from pathlib import Path
from collections import defaultdict

# ───────────── import your helpers ──────────────
from image_processing import load_config, run_pipeline, setup_logging

# ------- configuration to tweak quickly -------
CONFIG_FILE   = Path("/home/java/repos/cgras_coral_detection/experiments/increasing_images/config/amag_experiment.yaml")   # the YAML you just sent
SEEDS         = [42, 73, 105]                  # three experiment runs
BUCKET_SIZE   = 10                             # each bucket = 10 images
VAL_COUNT     = 15
TEST_COUNT    = 15

LABEL_NAMES = {0: "alive", 1: "dead",
               2: "mask_live", 3: "mask_dead"}

# ═════════════════  STEP 0  run standard pipeline  ═══════════════════
def preprocess_dataset(cfg: dict, logger: logging.Logger) -> Path:
    """
    Run filter ➜ patch ➜ balance once with the parameters in cfg.
    Returns the directory that contains the balanced images & labels.
    """
    cfg_no_split = cfg.copy()
    cfg_no_split['pipeline'] = [s for s in cfg['pipeline']
                                if s != 'split']   # drop the old 0.7/0.15/0.15 split

    run_pipeline(cfg_no_split, logger)

    # work out where the balanced data was written
    proj = cfg_no_split['project_name']
    out  = Path(cfg_no_split['output_base_path'])
    final_dir = out / f"{proj}_filtered_tiled_balanced"
    if not final_dir.exists():
        logger.error(f"Balanced directory not found at {final_dir}")
        sys.exit(1)

    # there is exactly one YAML in that folder → use it later
    yaml_path = next(final_dir.rglob("*.yaml"))
    logger.info(f"Balanced dataset ready at {final_dir} (yaml: {yaml_path})")
    return yaml_path

# ═══════════════  STEP 1  index tiles by original image  ═════════════
def index_tiles(dataset_yaml: Path) -> dict[str, list[Path]]:
    """
    Map original-image-stem → list[patched-image Paths].
    The stem is everything *except* the last two '_0000_0000' coords.
    """
    ds_root = dataset_yaml.parent            # e.g. …/amag_exp_filtered_tiled_balanced
    images_dirs = list(ds_root.rglob("images"))

    mapping = defaultdict(list)
    for img_dir in images_dirs:
        for p in Path(img_dir).rglob("*.jpg"):
            stem = "_".join(p.stem.split("_")[:-2])   # drop patch_x & patch_y
            mapping[stem].append(p)
    return mapping

# ═══════════════  STEP 2  copy/link helper  ══════════════════════════
def copy_tiles(stem: str, tiles: list[Path], dst_img: Path, dst_lbl: Path):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for t in tiles:
        # image
        _copy_one(t, dst_img)

        # label (if it exists)
        lbl = t.parent.parent / "labels" / f"{t.stem}.txt"
        if lbl.exists():
            _copy_one(lbl, dst_lbl)

def _copy_one(src: Path, dst_root: Path):
    dst = dst_root / src.name
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ═══════════════  STEP 3  write one dataset yaml  ════════════════════
def write_yaml(run_dir: Path, upto_bucket: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    buckets = [f"train_{i}" for i in range(10, upto_bucket + 1, 10)]
    data = {
        "path": run_dir.resolve().as_posix(),
        "train": [f"{b}/images" for b in buckets],
        "val":   ["val_15/images"],
        "test":  ["test_15/images"],
        "names": LABEL_NAMES
    }
    with open(run_dir / f"cgras_data_{upto_bucket}.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

# ═══════════════  STEP 4  build one run  ═════════════════════════════
def build_run(seed: int, mapping: dict[str, list[Path]], logger: logging.Logger, OUT_ROOT: Path):
    random.seed(seed)
    stems = list(mapping)
    random.shuffle(stems)

    val_stems  = stems[:VAL_COUNT]
    test_stems = stems[VAL_COUNT:VAL_COUNT+TEST_COUNT]
    train_pool = stems[VAL_COUNT+TEST_COUNT:VAL_COUNT+TEST_COUNT+100]

    run_dir = OUT_ROOT / f"seed_{seed}"
    for group, stem_list in [("val_15", val_stems), ("test_15", test_stems)]:
        for s in stem_list:
            copy_tiles(s, mapping[s],
                       run_dir / group / "images",
                       run_dir / group / "labels")

    # 10-image buckets
    for i in range(0, 100, BUCKET_SIZE):
        bucket_name = f"train_{i+BUCKET_SIZE}"
        for s in train_pool[i:i+BUCKET_SIZE]:
            copy_tiles(s, mapping[s],
                       run_dir / bucket_name / "images",
                       run_dir / bucket_name / "labels")

    # one YAML per cumulative bucket
    for upto in range(10, 101, 10):
        write_yaml(run_dir, upto)

    # summary printout
    logger.info(f"[seed {seed}] patch totals:")
    for group in ["val_15", "test_15"]+[f"train_{i}" for i in range(10, 101, 10)]:
        n = sum(1 for _ in (run_dir/group/"images").glob("*.jpg"))
        logger.info(f"  {group:8s}: {n:6d} patches")

# ═══════════════  MAIN  ══════════════════════════════════════════════
def main():
    log = setup_logging(logging.INFO)
    cfg = load_config(CONFIG_FILE, log)

    balanced_yaml = preprocess_dataset(cfg, log)
    mapping       = index_tiles(balanced_yaml)
    
    OUT_ROOT = Path(cfg['output_base_path']).expanduser().resolve() / "data_img_exp"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sd in SEEDS:
        build_run(sd, mapping, log, OUT_ROOT)

    log.info("✔  All experiment datasets ready in %s", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
