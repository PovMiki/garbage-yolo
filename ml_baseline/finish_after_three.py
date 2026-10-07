"""Finalize the user-shortened three-epoch run from its saved checkpoint."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import pandas as pd
from ultralytics import YOLO
import yaml

from .train import copy_native_figures, dump_json, metrics_dict, nullable_float, save_predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    dataset=args.dataset.resolve();out=args.out.resolve()
    summary=json.loads((out/'dataset_summary.json').read_text())
    config=yaml.safe_load((out/'config.yaml').read_text())
    table=pd.read_csv(out/'results.csv');table.columns=table.columns.str.strip()
    if len(table)!=3:
        raise RuntimeError(f'Expected exactly 3 completed epochs, found {len(table)}')
    best=out/'weights'/'best.pt';last=out/'weights'/'last.pt'
    if not best.is_file() or not last.is_file():raise FileNotFoundError('Missing best.pt or last.pt')
    shutil.copy2(out/'results.csv',out/'training_metrics.csv')
    map_col=next(c for c in table if 'mAP50-95' in c)
    best_epoch=int(table.loc[table[map_col].idxmax(),'epoch'])
    log=(out/'training.log').read_text(errors='replace')
    match=re.search(r'optimizer:.*?(AdamW|SGD|Adam|RMSprop)\(lr=([0-9.]+)',log)
    config.update({'epochs_executed':3,'user_requested_stop_after_epoch':3,
                   'best_epoch':best_epoch,'stop_reason':'user_requested_after_3_epochs',
                   'optimizer_actual':match.group(1) if match else None,
                   'learning_rate_actual':float(match.group(2)) if match else None})
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    with (out/'emissions.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise RuntimeError('CodeCarbon emissions.csv has no completed measurement')
    carbon=rows[-1]
    with (out/'emissions_baseline.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(carbon));writer.writeheader();writer.writerow(carbon)
    duration=nullable_float(carbon.get('duration'))
    green={'training_time_seconds':duration,'training_time_minutes':duration/60 if duration else None,
           'energy_consumed_kwh':nullable_float(carbon.get('energy_consumed')),
           'co2_emissions_kg':nullable_float(carbon.get('emissions')),
           'cpu_energy_kwh':nullable_float(carbon.get('cpu_energy')),
           'gpu_energy_kwh':nullable_float(carbon.get('gpu_energy')),
           'ram_energy_kwh':nullable_float(carbon.get('ram_energy')),
           'codecarbon_run_id':carbon.get('run_id'),
           'method':'CodeCarbon machine tracker from training start to interruption after the third saved epoch; includes epoch validation and a short part of the next batch before interruption, excludes final evaluation. CPU power estimated from load/TDP on macOS.',
           'limitations':'No direct CPU energy sensor was available; component values are CodeCarbon estimates, not wattmeter readings.'}
    dump_json(out/'green_ai.json',green)
    names=summary['classes'];imgsz=config['image_size'];batch=config['batch_size'];device=config['device']
    model=YOLO(str(best))
    print('VALIDATION_START',datetime.now(timezone.utc).isoformat(),flush=True)
    val=model.val(data=str(dataset/'data.yaml'),split='val',imgsz=imgsz,batch=batch,device=device,
                  plots=True,project=str(out),name='validation_eval',exist_ok=True)
    print('TEST_START',datetime.now(timezone.utc).isoformat(),flush=True)
    test=model.val(data=str(dataset/'data.yaml'),split='test',imgsz=imgsz,batch=batch,device=device,
                   plots=True,project=str(out),name='test_eval',exist_ok=True)
    result=metrics_dict(test,names)
    result.update({'validation':metrics_dict(val,names),'best_epoch':best_epoch,'epochs_executed':3,
                   'training_time_seconds':duration,'model_size_mb':best.stat().st_size/1_000_000,
                   'best_model':str(best),'test_evaluated_at_utc':datetime.now(timezone.utc).isoformat(),
                   'run_note':'Training deliberately stopped after 3 epochs at user request.'})
    dump_json(out/'final_metrics.json',result)
    copy_native_figures(out)
    images=pd.read_csv(out/'image_statistics.csv');boxes=pd.read_csv(out/'object_statistics.csv')
    for split in ('validation','test'):
        target=out/'predictions'/split
        if target.exists():shutil.rmtree(target)
        save_predictions(model,out,split,images,boxes,names,imgsz,device)
    print('FINISHED',datetime.now(timezone.utc).isoformat(),flush=True)
    print(json.dumps({k:result[k] for k in ['precision','recall','f1_score','map50','map50_95']}),flush=True)

if __name__=='__main__':main()
