"""Check completeness and consistency of baseline artifacts without retraining."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

REQUIRED = [
    'config.yaml','dataset_summary.json','training_metrics.csv','final_metrics.json',
    'green_ai.json','training.log','weights/best.pt','weights/last.pt',
    'audit_issues.json','image_statistics.csv','object_statistics.csv','REPORT.md',
]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve()
    missing=[name for name in REQUIRED if not (out/name).is_file() or (out/name).stat().st_size==0]
    if missing:raise AssertionError(f'Missing or empty artifacts: {missing}')
    config=yaml.safe_load((out/'config.yaml').read_text())
    summary=json.loads((out/'dataset_summary.json').read_text())
    final=json.loads((out/'final_metrics.json').read_text())
    green=json.loads((out/'green_ai.json').read_text())
    history=pd.read_csv(out/'training_metrics.csv')
    assert len(history)==config['epochs_executed']==final['epochs_executed']
    assert final['best_epoch'] in history.iloc[:,0].to_numpy()
    assert summary['total_images']==sum(v['images'] for v in summary['splits'].values())
    assert summary['total_objects']==sum(v['objects'] for v in summary['splits'].values())
    assert set(final['per_class'])<=set(summary['classes'])
    assert all(final[key] is not None for key in ['precision','recall','f1_score','map50','map50_95'])
    assert green['training_time_seconds'] is not None and green['training_time_seconds']>0
    assert (out/'figures').is_dir() and any((out/'figures').glob('*.png'))
    assert (out/'predictions'/'test').is_dir() and any((out/'predictions'/'test').glob('*.jpg'))
    print('Baseline artifacts verified:',len(REQUIRED),'required files,',len(history),'epochs,',len(final['per_class']),'classes')

if __name__=='__main__':main()
