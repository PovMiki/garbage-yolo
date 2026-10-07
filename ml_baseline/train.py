"""Pretrained YOLO Nano baseline, CodeCarbon training window, and final evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import ultralytics
from ultralytics import YOLO
import yaml
from codecarbon import EmissionsTracker
import codecarbon
from .report import generate_report


def dump_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=str))


def nullable_float(value):
    try:
        number=float(value)
        return number if np.isfinite(number) else None
    except (TypeError,ValueError):
        return None


def metrics_dict(metrics,names):
    box=metrics.box
    p,r=nullable_float(box.mp),nullable_float(box.mr)
    result={'precision':p,'recall':r,'f1_score':2*p*r/(p+r) if p is not None and r is not None and p+r else None,
            'map50':nullable_float(box.map50),'map50_95':nullable_float(box.map),
            'per_class':{}}
    for i,cls_id in enumerate(box.ap_class_index):
        pp,rr,ap50,ap=box.class_result(i)
        pp,rr=nullable_float(pp),nullable_float(rr)
        result['per_class'][names[int(cls_id)]]={
            'precision':pp,'recall':rr,'f1_score':2*pp*rr/(pp+rr) if pp is not None and rr is not None and pp+rr else None,
            'ap50':nullable_float(ap50),'ap50_95':nullable_float(ap)}
    return result


def copy_native_figures(out):
    figures=out/'figures';figures.mkdir(exist_ok=True)
    for p in out.glob('*.png'):
        shutil.copy2(p,figures/p.name)
    for split_dir in [out/'validation_eval',out/'test_eval']:
        if split_dir.exists():
            for p in split_dir.rglob('*.png'):
                shutil.copy2(p,figures/(split_dir.name+'_'+p.name))


def save_predictions(model,out,split,images,boxes,names,imgsz,device):
    """Draw actual predictions and original GT; classify examples by greedy IoU=0.5 matching."""
    target=out/'predictions'/split;target.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    image_paths=images.loc[images.split==split,'image'].to_numpy()
    selected=rng.permutation(image_paths)[:min(300,len(image_paths))]
    category_counts={key:0 for key in ['correct','false_positive','false_negative','misclassification','low_confidence']}
    for path in selected:
        if all(count>=6 for count in category_counts.values()):
            break
        gt=boxes[boxes.image==path]
        im=cv2.imread(path)
        if im is None:continue
        h,w=im.shape[:2]
        pred=model.predict(path,verbose=False,conf=.05,imgsz=imgsz,device=device)[0]
        xyxy=pred.boxes.xyxy.cpu().numpy() if len(pred.boxes) else np.empty((0,4))
        cls=pred.boxes.cls.cpu().numpy().astype(int) if len(pred.boxes) else np.array([],dtype=int)
        conf=pred.boxes.conf.cpu().numpy() if len(pred.boxes) else np.array([])
        gt_arr=[]
        for b in gt.itertuples():
            gt_arr.append((np.array([(b.xc-b.bw/2)*w,(b.yc-b.bh/2)*h,(b.xc+b.bw/2)*w,(b.yc+b.bh/2)*h]),b.class_id))
        matches=set();fp=fn=wrong=low=correct=0
        for j in np.argsort(-conf):
            box=xyxy[j]
            overlaps=[]
            for k,(truth,_) in enumerate(gt_arr):
                inter=np.maximum(0,np.minimum(box[2:],truth[2:])-np.maximum(box[:2],truth[:2]))
                ia=inter[0]*inter[1]
                union=np.prod(np.maximum(0,box[2:]-box[:2]))+np.prod(np.maximum(0,truth[2:]-truth[:2]))-ia
                overlaps.append(ia/union if union>0 else 0)
            k=int(np.argmax(overlaps)) if overlaps else -1
            hit=k>=0 and overlaps[k]>=.5 and k not in matches
            if hit:
                matches.add(k)
                if gt_arr[k][1]!=cls[j]:wrong+=1
                else:correct+=1
            elif conf[j]>=.25:fp+=1
            if conf[j]<.25:low+=1
        fn=len(gt_arr)-len(matches)
        categories=[name for name,count in [('correct',correct),('false_positive',fp),('false_negative',fn),
                                           ('misclassification',wrong),('low_confidence',low)]
                    if count and category_counts[name]<6]
        if not categories:continue
        vis=im.copy()
        for truth,c in gt_arr:
            x1,y1,x2,y2=truth.astype(int)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,220,0),1)
            cv2.putText(vis,'GT '+names[int(c)],(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,150,0),1)
        for box,c,score in zip(xyxy,cls,conf):
            x1,y1,x2,y2=box.astype(int)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.putText(vis,f'{names[int(c)]} {score:.2f}',(x1,min(h-4,y2+13)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
        for category in categories:
            name=f'{category}_{Path(path).name}'
            cv2.imwrite(str(target/name),vis)
            category_counts[category]+=1
    dump_json(target/'example_counts.json',category_counts)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--imgsz',type=int,default=None)
    ap.add_argument('--patience',type=int,default=7)
    ap.add_argument('--device',default='auto')
    args=ap.parse_args()
    dataset=args.dataset.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((out/'dataset_summary.json').read_text())
    if summary['cross_split_duplicate_groups']:
        raise RuntimeError('Cross-split identical images found. Resolve leakage before training.')
    names=summary['classes']
    imgsz=args.imgsz or int(min(640,max(320,round(summary['image_statistics']['width']['50%']/32)*32)))
    device=('mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    config={'model':'YOLO11n detection','pretrained_weights':'yolo11n.pt','dataset_path':str(dataset),'data_yaml':str(dataset/'data.yaml'),
            'num_classes':len(names),'class_names':names,'image_size':imgsz,'batch_size':args.batch,
            'max_epochs':args.epochs,'epochs_executed':None,'optimizer_requested':'auto','optimizer_actual':None,
            'learning_rate_requested':None,'learning_rate_actual':None,'weight_decay':0.0005,
            'patience':args.patience,'seed':42,'device':device,
            'split_counts':{s:{'images':v['images'],'objects':v['objects']} for s,v in summary['splits'].items()},
            'python_version':platform.python_version(),'pytorch_version':str(torch.__version__),
            'torchvision_version':str(torchvision.__version__),'ultralytics_version':str(ultralytics.__version__),
            'codecarbon_version':codecarbon.__version__,
            'data_caveats':summary['issue_counts'],
            'annotation_note':'Mixed true YOLO bbox and polygon labels; Ultralytics derives detection boxes from polygons.'}
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    tracker=EmissionsTracker(project_name='baseline_yolo_nano',output_dir=str(out),output_file='emissions.csv',
                             save_to_file=True,log_level='warning',measure_power_secs=15,tracking_mode='machine')
    measurement={'started':False,'stopped':False,'start_clock':None,'seconds':None,'emissions':None}
    def start_training(trainer):
        if not measurement['started']:
            measurement['start_clock']=time.monotonic()
            tracker.start();measurement['started']=True
            print('CODECARBON_START',datetime.now(timezone.utc).isoformat(),flush=True)
    def stop_training(trainer):
        if measurement['started'] and not measurement['stopped'] and trainer.stop:
            measurement['emissions']=tracker.stop()
            measurement['seconds']=time.monotonic()-measurement['start_clock']
            measurement['stopped']=True
            print('CODECARBON_STOP',datetime.now(timezone.utc).isoformat(),flush=True)
    model=YOLO('yolo11n.pt')
    config['model_parameters']=sum(p.numel() for p in model.model.parameters())
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    model.add_callback('on_train_start',start_training)
    model.add_callback('on_fit_epoch_end',stop_training)
    started=datetime.now(timezone.utc)
    print('TRAINING_START',started.isoformat(),flush=True)
    print('DEVICE',device,'PARAMETERS',config['model_parameters'],'CONFIG',json.dumps(config,ensure_ascii=False),flush=True)
    warnings.simplefilter('default')
    try:
        model.train(data=str(dataset/'data.yaml'),epochs=args.epochs,imgsz=imgsz,batch=args.batch,
                    device=device,patience=args.patience,seed=42,deterministic=True,optimizer='auto',
                    project=str(out.parent),name=out.name,exist_ok=True,plots=True,save=True,
                    workers=2,verbose=True)
    except BaseException:
        print('TRAINING_ERROR',traceback.format_exc(),flush=True)
        raise
    finally:
        if measurement['started'] and not measurement['stopped']:
            measurement['emissions']=tracker.stop()
            measurement['seconds']=time.monotonic()-measurement['start_clock']
            measurement['stopped']=True
        print('TRAINING_CALL_END',datetime.now(timezone.utc).isoformat(),flush=True)
    results=out/'results.csv'
    if results.exists():
        shutil.copy2(results,out/'training_metrics.csv')
        table=pd.read_csv(results);table.columns=table.columns.str.strip()
        config['epochs_executed']=len(table)
        col=next((c for c in table if 'mAP50-95' in c),None)
        config['best_epoch']=int(table.loc[table[col].idxmax(),'epoch']) if col and len(table) else None
        config['learning_rate_actual']=nullable_float(table.filter(regex='lr/pg').iloc[0,0]) if len(table) and not table.filter(regex='lr/pg').empty else None
    else:
        config['best_epoch']=None
    trainer=model.trainer
    config['optimizer_actual']=type(trainer.optimizer).__name__ if trainer and trainer.optimizer else None
    if trainer and trainer.optimizer:
        config['weight_decay_actual']=[group.get('weight_decay') for group in trainer.optimizer.param_groups]
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    carbon_csv=out/'emissions.csv';row={}
    if carbon_csv.exists():
        with carbon_csv.open(newline='') as f:
            records=list(csv.DictReader(f))
        if records:row=records[-1]
    green={'training_time_seconds':measurement['seconds'],'training_time_minutes':measurement['seconds']/60 if measurement['seconds'] else None,
           'energy_consumed_kwh':nullable_float(row.get('energy_consumed')),
           'co2_emissions_kg':nullable_float(row.get('emissions')) or nullable_float(measurement['emissions']),
           'cpu_energy_kwh':nullable_float(row.get('cpu_energy')),
           'gpu_energy_kwh':nullable_float(row.get('gpu_energy')),
           'ram_energy_kwh':nullable_float(row.get('ram_energy')),
           'method':'CodeCarbon 3.3.1 machine tracker; estimated energy where hardware sensors unavailable. Window: on_train_start to final on_fit_epoch_end, including epoch validation; excludes final_eval and post-training plots.',
           'limitations':'Null component values mean CodeCarbon did not report them on this device.'}
    dump_json(out/'green_ai.json',green)
    best=out/'weights'/'best.pt';last=out/'weights'/'last.pt'
    if not best.exists() or not last.exists():raise FileNotFoundError('Missing best.pt or last.pt')
    print('EVALUATION_VALIDATION_START',datetime.now(timezone.utc).isoformat(),flush=True)
    best_model=YOLO(str(best))
    val=best_model.val(data=str(dataset/'data.yaml'),split='val',imgsz=imgsz,batch=args.batch,
                       device=device,plots=True,project=str(out),name='validation_eval',exist_ok=True)
    print('EVALUATION_TEST_START',datetime.now(timezone.utc).isoformat(),flush=True)
    test=best_model.val(data=str(dataset/'data.yaml'),split='test',imgsz=imgsz,batch=args.batch,
                        device=device,plots=True,project=str(out),name='test_eval',exist_ok=True)
    result=metrics_dict(test,names)
    result.update({'validation':metrics_dict(val,names),'best_epoch':config['best_epoch'],
                   'epochs_executed':config['epochs_executed'],'training_time_seconds':measurement['seconds'],
                   'model_size_mb':best.stat().st_size/1_000_000,'best_model':str(best),
                   'test_evaluated_at_utc':datetime.now(timezone.utc).isoformat()})
    dump_json(out/'final_metrics.json',result)
    copy_native_figures(out)
    images=pd.read_csv(out/'image_statistics.csv')
    boxes=pd.read_csv(out/'object_statistics.csv')
    save_predictions(best_model,out,'validation',images,boxes,names,imgsz,device)
    save_predictions(best_model,out,'test',images,boxes,names,imgsz,device)
    generate_report(out)
    print('FINISHED',datetime.now(timezone.utc).isoformat(),flush=True)
    print(json.dumps({'test':{k:result[k] for k in ['precision','recall','f1_score','map50','map50_95']},'green_ai':green},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
