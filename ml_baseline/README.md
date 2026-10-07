# YOLO Nano baseline

The saved reference artifacts remain in `runs/baseline_yolo_nano/`. The commands below create a separate run.

The existing split in `Trash Detection Dataset/CUSTOM_DATASET` is used as-is. The source `data.yaml` defines six classes and points to `train`, `valid`, and `test`. Labels contain both YOLO `class x_center y_center width height` rows and YOLO polygon rows. Ultralytics derives detection boxes from the polygon coordinates; the source annotations are never rewritten.

## Run

Use Python 3.13 with the packages in `requirements-ml.txt`. From the repository root:

```sh
python -m pip install -r requirements-ml.txt
python -m ml_baseline.audit --dataset 'Trash Detection Dataset/CUSTOM_DATASET' --out runs/academic_3_epoch_example
python -m ml_baseline.train --dataset 'Trash Detection Dataset/CUSTOM_DATASET' --out runs/academic_3_epoch_example --epochs 3 --batch 32 --imgsz 320 --patience 5 > runs/academic_3_epoch_example/training.log 2>&1
python -m ml_baseline.report --out runs/academic_3_epoch_example
python -m ml_baseline.verify_run --out runs/academic_3_epoch_example
```

The audit runs before training. It writes `dataset_summary.json`, `audit_issues.json`, object and image CSV tables, and EDA figures. It checks box rows and polygon rows separately, hashes all image bytes for exact duplicates, and does not change the dataset. If exact cross-split leakage is found, training stops.

The training command fine-tunes pretrained `yolo11n.pt` for detection. This short academic baseline uses 3 epochs and 320 px input, as requested by the user. The script default, if `--imgsz` is omitted, derives size from the median source width rounded to a multiple of 32 and capped at 640. Seed is 42. Ultralytics chooses the optimizer and uses its standard training augmentation. On this host, MPS is unavailable to PyTorch, so the recorded run uses CPU. CodeCarbon starts at Ultralytics `on_train_start` and stops after the final fit epoch, including per-epoch validation. It excludes EDA and final evaluation. Component energy values are null when CodeCarbon does not report them.

The saved run began with a 10-epoch schedule and was interrupted after its third saved epoch at the user's request. Running the command above with `--epochs 3` is a comparable academic run, but its learning-rate schedule may differ, so it will not reproduce byte-identical weights.

After training, the script saves the native epoch CSV as `training_metrics.csv`, evaluates `best.pt` first on validation and then on test, and writes `final_metrics.json`. It copies native plots into `figures/` and saves sample predictions with true boxes in `predictions/`. The test split is evaluated only after training and validation evaluation.

`ML_baseline.ipynb` presents the audit, charts, metrics, and conclusions and contains reproducible commands. The existing `EDA.ipynb` and dataset are left untouched.
