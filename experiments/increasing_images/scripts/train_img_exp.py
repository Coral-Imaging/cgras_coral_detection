#!/usr/bin/env python3
"""
Bulk-train and evaluate YOLO models for the incremental-image experiment.

Example
-------
python run_experiment.py \
       --seed-dirs img_exp/output/seed_42 img_exp/output/seed_73 \
       --train-cfg amag_train.yaml \
       --num-runs 3 \
       --conf-levels auto,0.25,0.50,0.75
"""
from __future__ import annotations
import click, csv, time, uuid
from pathlib import Path
from ultralytics import YOLO
import yaml

# ───────────────────────── helpers ─────────────────────────
def parse_conf_list(s: str) -> list[float | None]:
    out: list[float | None] = []
    for item in s.split(","):
        item = item.strip().lower()
        if item in ("auto", "none", ""):
            out.append(None)
        else:
            out.append(float(item))
    return out

def log_results(csv_path: Path, row: dict):
    first = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if first:
            w.writeheader()
        w.writerow(row)

# ───────────────────────── CLI ─────────────────────────────
@click.command()
@click.option("--seed-dirs", multiple=True, required=True, type=click.Path(exists=True, file_okay=False),
              help="One or more seed_X directories (output of data_img_exp.py script)")
@click.option("--train-cfg", required=True, type=click.Path(exists=True, dir_okay=False),
              help="YAML file with training hyper-parameters")
@click.option("--num-runs", default=1, show_default=True, 
              help="Times to retrain each data split")
@click.option("--out-dir", default="/media/java/RRAP03/data/outputs/experiments/num_image_exp/train_img_exp/", show_default=True, type=click.Path(file_okay=False),
              help="Base directory where models/ and result CSVs are written")
def main(seed_dirs, train_cfg, num_runs, out_dir):
    seed_dirs   = [Path(p).resolve() for p in seed_dirs]
    train_cfg   = yaml.safe_load(Path(train_cfg).read_text())
    num_runs    = int(num_runs)

    device = str(train_cfg.get("device", 0))
    print(f"Using device: {device}")

    out_base = Path(out_dir).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved under: {out_base}")

    for seed_dir in seed_dirs:
        print(f"\n==== {seed_dir.name} ====")
        data_yamls = sorted(seed_dir.glob("cgras_data_*.yaml"),
                            key=lambda p: int(p.stem.split('_')[-1]))
        csv_path = out_base / seed_dir.name / f"results_{seed_dir.name}.csv"

        for data_yaml in data_yamls:
            train_size = int(data_yaml.stem.split('_')[-1])

            for run_idx in range(1, num_runs + 1):
                run_tag  = f"{train_size}_run{run_idx}"
                name_tag = f"{train_cfg['name']}_{run_tag}"
                project  = out_base / seed_dir.name / "models"

                # 1️⃣  build / load model
                model = YOLO(train_cfg["model_path"])

                # 2️⃣  train
                print(f"[Train] N={train_size} run={run_idx}")
                model.train(
                    data=str(data_yaml),
                    project=str(project),
                    name=name_tag,
                    epochs=train_cfg["epochs"],
                    batch=train_cfg["batch_size"],
                    imgsz=train_cfg["image_size"],
                    patience=train_cfg["patience"],
                    workers=train_cfg["workers"],
                    pretrained=train_cfg["pretrained"],
                    classes=train_cfg["classes"],
                    device=device,
                    scale=train_cfg.get("scale", 0.0),
                    flipud=train_cfg.get("flipud", 0.0),
                    fliplr=train_cfg.get("fliplr", 0.0),
                    overlap_mask=train_cfg.get("mask_overlap", False),
                    save_period=train_cfg["save_period"],
                    verbose=False,
                )

                best_weights = (project / name_tag / "weights" / "best.pt")

                conf_levels = train_cfg.get("conf", [0.001])
                # 3️⃣  evaluate at each conf
                for conf in conf_levels:
                    metrics = model.val(
                        data=str(data_yaml),
                        split=train_cfg.get("split", "test"),
                        iou=train_cfg.get("iou", 0.5),
                        conf=conf,
                        device=device,
                        plots=train_cfg.get("plots", False),
                        model=str(best_weights),
                    )
                    m = metrics.metrics_d
                    row = dict(
                        seed=seed_dir.name,
                        train_size=train_size,
                        run=run_idx,
                        conf=("auto" if conf is None else conf),
                        map50=m.seg.map50,
                        precision=m.seg.p,
                        recall=m.seg.r,
                        f1=m.seg.f1,
                        weights=str(best_weights),
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        uuid=uuid.uuid4().hex[:8],
                    )
                    log_results(csv_path, row)
                    print(f"  conf={row['conf']:>4}  mAP50={row['map50']:.3f}")

    print("\n✓ All training & evaluations complete.")

# entry-point
if __name__ == "__main__":
    main()
