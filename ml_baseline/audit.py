"""Read-only dataset audit and object-detection EDA for the existing YOLO split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from ultralytics.data.utils import check_det_dataset

SPLITS = {'train': 'train', 'validation': 'val', 'test': 'test'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
sns.set_theme(style='whitegrid')


def issue(issues, kind, path, detail):
    issues.append({'kind': kind, 'path': str(path), 'detail': str(detail)})


def box_from_row(parts, n_classes, path, line_no, issues):
    """Return class and xywh derived from actual YOLO box or polygon coordinates."""
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        issue(issues, 'non_numeric_annotation', path, line_no)
        return None
    if not all(np.isfinite(vals)):
        issue(issues, 'nan_or_inf', path, line_no)
        return None
    if not vals or vals[0] != int(vals[0]) or not 0 <= vals[0] < n_classes:
        issue(issues, 'invalid_class_id', path, f'line {line_no}: {vals[0] if vals else "missing"}')
        return None
    cls = int(vals[0])
    if len(vals) == 5:
        x, y, w, h = vals[1:]
        kind = 'bbox'
    elif len(vals) >= 7 and len(vals) % 2 == 1:
        xy = np.asarray(vals[1:], dtype=float).reshape(-1, 2)
        if np.any((xy < 0) | (xy > 1)):
            issue(issues, 'polygon_coordinate_out_of_range', path, line_no)
            return None
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        x, y, w, h = (x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0
        kind = 'polygon'
    else:
        issue(issues, 'wrong_number_of_values', path, f'line {line_no}: {len(vals)} values')
        return None
    for name, value in [('x_center', x), ('y_center', y), ('width', w), ('height', h)]:
        if name in ('x_center', 'y_center') and not 0 <= value <= 1:
            issue(issues, name + '_out_of_range', path, line_no)
        if name in ('width', 'height') and value <= 0:
            issue(issues, name + '_non_positive', path, line_no)
        if name in ('width', 'height') and value > 1:
            issue(issues, name + '_over_one', path, line_no)
    if x-w/2 < -1e-6 or x+w/2 > 1+1e-6 or y-h/2 < -1e-6 or y+h/2 > 1+1e-6:
        issue(issues, 'box_outside_image', path, line_no)
    return {'class_id': cls, 'xc': x, 'yc': y, 'bw': w, 'bh': h,
            'area': w*h, 'box_aspect_ratio': w/h if h > 0 else np.nan,
            'annotation_type': kind}


def scan(dataset):
    meta = yaml.safe_load((dataset/'data.yaml').read_text())
    names = meta['names']
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if meta.get('nc') != len(names):
        raise ValueError('data.yaml: nc differs from names length')
    resolved = check_det_dataset(str(dataset/'data.yaml'), autodownload=False)
    images, boxes, issues = [], [], []
    digests = defaultdict(list)
    source_ids = defaultdict(set)
    for split, yaml_key in SPLITS.items():
        im_dir = Path(resolved[yaml_key])
        lb_dir = im_dir.parent/'labels'
        if not im_dir.is_dir() or not lb_dir.is_dir():
            raise FileNotFoundError(f'Missing image/label directory for {yaml_key}: {im_dir}, {lb_dir}')
        im_paths = sorted(p for p in im_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        lb_paths = sorted(lb_dir.glob('*.txt'))
        im_by_stem = {p.stem: p for p in im_paths}
        im_stems, lb_stems = set(im_by_stem), {p.stem for p in lb_paths}
        for stem in sorted(im_stems-lb_stems): issue(issues, 'image_without_label', im_by_stem[stem], '')
        for stem in sorted(lb_stems-im_stems): issue(issues, 'label_without_image', lb_dir/(stem+'.txt'), '')
        for p in im_paths:
            source_ids[p.stem.split('.rf.')[0]].add(split)
            label = lb_dir/(p.stem+'.txt')
            raw = p.read_bytes()
            digests[hashlib.sha256(raw).hexdigest()].append((split, str(p)))
            im = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                issue(issues, 'unreadable_image', p, '')
                continue
            height, width = im.shape[:2]
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            this_boxes = []
            if label.exists():
                lines = label.read_text(encoding='utf-8').splitlines()
                if not any(line.strip() for line in lines): issue(issues, 'empty_label', label, '')
                for line_no, line in enumerate(lines, 1):
                    if not line.strip(): continue
                    b = box_from_row(line.split(), len(names), label, line_no, issues)
                    if b is not None:
                        b.update(split=split, image=str(p), label=str(label), class_name=names[b['class_id']])
                        boxes.append(b)
                        this_boxes.append(b)
            images.append({'split': split, 'image': str(p), 'label': str(label),
                           'width': width, 'height': height, 'aspect_ratio': width/height,
                           'file_size_kb': len(raw)/1024,
                           'brightness': float(gray.mean()), 'contrast': float(gray.std()),
                           'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                           'n_objects': len(this_boxes),
                           'class_names': ', '.join(sorted({b['class_name'] for b in this_boxes}))})
    duplicate_groups = [members for members in digests.values() if len(members)>1]
    cross_split = [members for members in duplicate_groups if len({s for s,_ in members})>1]
    for group in duplicate_groups:
        issue(issues, 'identical_image_duplicate', group[0][1], group)
    for group in cross_split:
        issue(issues, 'cross_split_identical_image', group[0][1], group)
    cross_split_source_ids = sorted(k for k, splits in source_ids.items() if len(splits) > 1)
    return meta, names, pd.DataFrame(images), pd.DataFrame(boxes), issues, duplicate_groups, cross_split, cross_split_source_ids


def save_plot(path, title, xlabel, ylabel):
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_gallery(rows, boxes, names, path, title, limit=12):
    rows = rows.head(limit)
    if rows.empty: return
    cols = min(4,len(rows)); nrows = int(np.ceil(len(rows)/cols))
    fig, axs = plt.subplots(nrows,cols,figsize=(4*cols,3.4*nrows),squeeze=False)
    for ax in axs.flat: ax.axis('off')
    for ax, row in zip(axs.flat, rows.itertuples()):
        im = cv2.imread(row.image)
        if im is None: continue
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        h,w = im.shape[:2]
        image_boxes = boxes[boxes.image==row.image]
        dense = len(image_boxes) > 15
        for b in image_boxes.itertuples():
            x0,y0 = int((b.xc-b.bw/2)*w),int((b.yc-b.bh/2)*h)
            rect=plt.Rectangle((x0,y0),b.bw*w,b.bh*h,fill=False,edgecolor='lime',linewidth=.7 if dense else 1.3)
            ax.add_patch(rect)
            if not dense:
                ax.text(x0,max(0,y0-2),b.class_name,fontsize=7,color='black',backgroundcolor='lime')
        ax.imshow(im); ax.set_title(f'{Path(row.image).name[:24]} ({len(image_boxes)} objects)',fontsize=8)
    fig.suptitle(title,fontsize=14)
    fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)


def make_figures(images, boxes, names, out):
    out.mkdir(parents=True,exist_ok=True)
    if boxes.empty: return
    counts = boxes.groupby('class_name').size().sort_values(ascending=False)
    plt.figure(figsize=(10,5)); sns.barplot(x=counts.index,y=counts.values)
    plt.xticks(rotation=30,ha='right'); save_plot(out/'objects_per_class.png','Objects per class','Class','Objects')
    shares = (boxes.groupby('class_name').size() / len(boxes) * 100).reindex(names).reset_index(name='share_pct')
    plt.figure(figsize=(9,5)); sns.barplot(data=shares,x='class_name',y='share_pct')
    plt.xticks(rotation=30,ha='right'); save_plot(out/'class_share.png','Class share','Class','Share of objects [%]')
    split_class=boxes.groupby(['split','class_name']).size().reset_index(name='objects')
    plt.figure(figsize=(10,5)); sns.barplot(data=split_class,x='class_name',y='objects',hue='split')
    plt.xticks(rotation=30,ha='right'); save_plot(out/'classes_by_split.png','Objects by class and split','Class','Objects')
    for col,label,units in [('width','Image width','px'),('height','Image height','px'),('aspect_ratio','Image aspect ratio','ratio'),('file_size_kb','File size','kB')]:
        plt.figure(figsize=(8,5)); sns.histplot(data=images,x=col,hue='split',bins=35,element='step')
        save_plot(out/f'image_{col}.png',label+' distribution',label+f' [{units}]','Images')
    for col,label in [('brightness','Brightness [gray level 0–255]'),('contrast','Contrast [gray-level standard deviation]'),('sharpness','Sharpness [Laplacian variance]')]:
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        sns.histplot(data=images,x=col,hue='split',bins=40,ax=axs[0],element='step')
        axs[0].set(title=label+' histogram',xlabel=label,ylabel='Images')
        sns.boxplot(data=images,x='split',y=col,ax=axs[1],showfliers=False)
        axs[1].set(title=label+' by split',xlabel='Split',ylabel=label)
        fig.tight_layout();fig.savefig(out/f'image_{col}.png',dpi=150);plt.close(fig)
        joined=boxes[['image','class_name']].drop_duplicates().merge(images[['image',col]],on='image')
        plt.figure(figsize=(10,5));sns.boxplot(data=joined,x='class_name',y=col,showfliers=False)
        plt.xticks(rotation=30,ha='right');save_plot(out/f'{col}_by_class.png',label+' by class (multi-class images repeated)','Class',label)
    for col,title in [('bw','Box width'),('bh','Box height'),('area','Box area')]:
        plt.figure(figsize=(8,5));sns.histplot(data=boxes,x=col,bins=40)
        save_plot(out/f'box_{col}.png',title+' distribution',title+' [fraction of image]','Objects')
    for x,y,title,file in [('bw','bh','Box width vs height','box_width_height.png'),('xc','yc','Box center positions','box_centers.png')]:
        plt.figure(figsize=(9,6));sns.scatterplot(data=boxes,x=x,y=y,hue='class_name',alpha=.12,s=6)
        plt.legend(title='Class',bbox_to_anchor=(1.02,1),loc='upper left')
        save_plot(out/file,title,x+' [fraction]',y+' [fraction]')
    plt.figure(figsize=(10,5));sns.boxplot(data=boxes,x='class_name',y='area',showfliers=False)
    plt.xticks(rotation=30,ha='right');save_plot(out/'box_area_by_class.png','Relative box area by class','Class','Area [fraction of image]')
    rng=np.random.default_rng(42)
    for name in names:
        candidates=images[images.image.isin(boxes.loc[boxes.class_name==name,'image'])]
        if len(candidates):
            chosen=candidates.iloc[rng.choice(len(candidates),size=min(8,len(candidates)),replace=False)]
            plot_gallery(chosen,boxes,names,out/f'examples_{name.lower()}.png',f'Real annotations: {name}',8)
    plot_gallery(images.sort_values('n_objects',ascending=False),boxes,names,out/'most_objects.png','Most objects')
    minimum_area=boxes.groupby('image').area.min().sort_values()
    maximum_area=boxes.groupby('image').area.max().sort_values(ascending=False)
    plot_gallery(images.set_index('image').loc[minimum_area.index[:12]].reset_index(),boxes,names,out/'smallest_objects.png','Smallest annotated objects')
    plot_gallery(images.set_index('image').loc[maximum_area.index[:12]].reset_index(),boxes,names,out/'largest_objects.png','Largest annotated objects')
    for metric,ascending,name in [('brightness',True,'dark_scenes'),('brightness',False,'bright_scenes'),('sharpness',True,'blurry_scenes')]:
        plot_gallery(images.sort_values(metric,ascending=ascending),boxes,names,out/(name+'.png'),name.replace('_',' ').title())
    hard=images[(images.n_objects>=images.n_objects.quantile(.9)) | (images.brightness<=images.brightness.quantile(.05)) | (images.sharpness<=images.sharpness.quantile(.05))]
    plot_gallery(hard.sample(min(12,len(hard)),random_state=42),boxes,names,out/'challenging_scenes.png','Potentially challenging scenes')


def summarize(meta,names,images,boxes,issues,duplicates,cross_split,cross_split_source_ids):
    total=len(images)
    split_info={}
    for split in SPLITS:
        im=images[images.split==split];bx=boxes[boxes.split==split]
        label_dir = Path(im.label.iloc[0]).parent if len(im) else None
        split_info[split]={'images':len(im),'annotation_files':len(list(label_dir.glob('*.txt'))) if label_dir else 0,
            'objects':len(bx),'share_pct':100*len(im)/total if total else None,
            'mean_objects_per_image':float(im.n_objects.mean()) if len(im) else None,
            'min_objects_per_image':int(im.n_objects.min()) if len(im) else None,
            'max_objects_per_image':int(im.n_objects.max()) if len(im) else None,
            'images_without_objects':int((im.n_objects==0).sum()),
            'objects_per_class':{name:int((bx.class_name==name).sum()) for name in names}}
    counts={name:int((boxes.class_name==name).sum()) for name in names}
    positive=[x for x in counts.values() if x>0]
    class_images={name:int(boxes.loc[boxes.class_name==name,'image'].nunique()) for name in names}
    issue_groups=Counter(x['kind'] for x in issues)
    return {'dataset_path':str(images.image.iloc[0]) if total else None,'classes':names,'n_classes':len(names),
        'total_images':total,'total_objects':len(boxes),'splits':split_info,
        'objects_per_class':counts,'class_share_pct':{n:100*v/len(boxes) if len(boxes) else None for n,v in counts.items()},
        'images_per_class':class_images,'imbalance_ratio':max(positive)/min(positive) if positive else None,
        'unique_resolutions':int(images[['width','height']].drop_duplicates().shape[0]),
        'image_statistics':{c:images[c].describe(percentiles=[.05,.25,.5,.75,.95]).dropna().to_dict() for c in ['width','height','aspect_ratio','file_size_kb','brightness','contrast','sharpness']},
        'box_statistics':{c:boxes[c].describe(percentiles=[.05,.25,.5,.75,.95]).dropna().to_dict() for c in ['bw','bh','area','box_aspect_ratio','xc','yc']},
        'annotation_types':dict(Counter(boxes.annotation_type)),
        'small_objects_area_lt_1pct':int((boxes.area<.01).sum()),
        'large_objects_area_gt_50pct':int((boxes.area>.5).sum()),
        'central_boxes_center_in_middle_half':int(((boxes.xc.between(.25,.75))&(boxes.yc.between(.25,.75))).sum()),
        'duplicate_groups':duplicates,'cross_split_duplicate_groups':cross_split,
        'cross_split_source_ids_by_filename':cross_split_source_ids,
        'issue_counts':dict(issue_groups),'issue_examples':issues[:30]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    dataset=args.dataset.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    meta,names,images,boxes,issues,duplicates,cross_split,cross_split_source_ids=scan(dataset)
    summary=summarize(meta,names,images,boxes,issues,duplicates,cross_split,cross_split_source_ids)
    summary['dataset_path']=str(dataset)
    (out/'dataset_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str))
    (out/'audit_issues.json').write_text(json.dumps(issues,indent=2,ensure_ascii=False))
    images.to_csv(out/'image_statistics.csv',index=False)
    boxes.to_csv(out/'object_statistics.csv',index=False)
    make_figures(images,boxes,names,out/'figures')
    print(json.dumps({k:summary[k] for k in ['total_images','total_objects','splits','objects_per_class','annotation_types','issue_counts','imbalance_ratio']},indent=2,ensure_ascii=False))
    print('cross_split_duplicate_groups:',len(cross_split))
    print('cross_split_source_ids_by_filename:',len(cross_split_source_ids))

if __name__=='__main__':main()
