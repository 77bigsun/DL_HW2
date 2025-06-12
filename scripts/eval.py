# ===================== eval.py (revised for your model) =====================
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# --- 從 model.py 匯入您的模型和後處理函式 ---
from model import UnifiedModel, postprocess_detection

# ---------------------------------------------------
# 公用工具 (與助教版本相同)
# ---------------------------------------------------
def _find_dir(root: Path, *candidates: str) -> Path:
    names = {c.lower() for c in candidates}
    for p in root.rglob('*'):
        if p.is_dir() and p.name.lower() in names:
            return p
    raise FileNotFoundError(f"Folder {candidates} not found under {root}")

def _find_json(root: Path) -> Path:
    js = list(root.rglob('*.json'))
    if not js: raise FileNotFoundError(f'No annotation json under {root}')
    if len(js) > 1: warnings.warn(f'Multiple json under {root}, picking {js[0].name}')
    return js[0]

# ---------------------------------------------------
# Dataset 定義 (與助教版本相同)
# ---------------------------------------------------
class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, img_size: int = 512):
        subdirs = [p for p in root.rglob('*') if p.is_dir() and list(p.glob('*.*'))]
        if not subdirs: raise FileNotFoundError(f'No image folders under {root}')
        self.samples = [(img, img.parent.name) for folder in subdirs for img in folder.glob('*.*')]
        self.cls2idx = {c: i for i, c in enumerate(sorted({cls for _, cls in self.samples}))}
        self.tfm = T.Compose([
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        path, cls = self.samples[idx]
        img = self.tfm(Image.open(path).convert('RGB'))
        return img, self.cls2idx[cls]

class CocoHiddenDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, img_size: int = 512):
        img_dir = _find_dir(root, 'images', 'imgs')
        ann_path = _find_json(root)
        from pycocotools.coco import COCO
        self.coco = COCO(str(ann_path))
        self.ids = list(self.coco.imgs)
        self.img_dir = img_dir
        self.tfm = T.Compose([
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        meta = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / meta['file_name']
        img = self.tfm(Image.open(img_path).convert('RGB'))
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(a['category_id'])
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}
        return img, target

class VocHiddenDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, img_size: int = 512):
        img_dir = _find_dir(root, 'jpegimages', 'images', 'img')
        mask_dir = _find_dir(root, 'segmentationclass', 'masks', 'mask')
        self.imgs = sorted(img_dir.glob('*.jpg'))
        self.masks = sorted(mask_dir.glob('*.png'))
        self.tfm_img = T.Compose([
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.tfm_mask = T.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx: int):
        img = self.tfm_img(Image.open(self.imgs[idx]).convert('RGB'))
        mask = np.array(self.tfm_mask(Image.open(self.masks[idx])))
        mask = torch.as_tensor(mask, dtype=torch.int64)
        return img, mask

# ---------------------------------------------------
# Metric helpers (torchmetrics v1.3+)
# ---------------------------------------------------
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.segmentation import MulticlassJaccardIndex
from torchmetrics.classification import MulticlassAccuracy

def eval_det(model, loader, device):
    metric = MeanAveragePrecision(box_format='xyxy').to('cpu')
    model.eval()
    for imgs, targets in loader:
        imgs = imgs.to(device)
        _, det_raw, _ = model(imgs)
        
        # 使用您自訂的後處理函式
        preds = postprocess_detection(det_raw.cpu()) 
        
        # 更新 metric
        metric.update(preds, targets)
    
    map_results = metric.compute()
    return map_results['map'].item()

def eval_seg(model, loader, device):
    metric = MulticlassJaccardIndex(num_classes=21, ignore_index=255).to(device)
    model.eval()
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        _, _, seg_logits = model(imgs)
        metric.update(seg_logits.argmax(1), masks)
    return metric.compute().item()

def eval_cls(model, loader, device):
    metric = MulticlassAccuracy(num_classes=10, top_k=1).to(device)
    model.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        cls_logits, _, _ = model(imgs)
        metric.update(cls_logits.argmax(1), labels)
    return metric.compute().item()

# ---------------------------------------------------
# CLI & main
# ---------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, type=Path, help="Path to your trained model (.pt or .pth)")
    ap.add_argument('--data_root', required=True, type=Path, help="Path to the hidden test set root directory")
    ap.add_argument('--tasks', default='all', help='Tasks to evaluate: det,seg,cls or all')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--cpu', action='store_true')
    return ap.parse_args()

def collate_det(batch):
    return tuple(zip(*batch))

def main():
    args = parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Using device: {device}")

    # 載入您的 UnifiedModel 模型
    model = UnifiedModel(num_classes=10, num_det_classes=10, num_seg_classes=21).to(device)
    try:
        state_dict = torch.load(args.weights, map_location=device)
        # 處理可能的 'model_state_dict' 鍵
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {args.weights}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model.eval()
    
    selected_tasks = {'det', 'seg', 'cls'} if args.tasks == 'all' else {t.strip() for t in args.tasks.split(',')}
    metrics: dict[str, float] = {}

    if 'det' in selected_tasks:
        print("\nEvaluating Object Detection...")
        det_ds = CocoHiddenDataset(args.data_root / 'mini_coco_det')
        det_loader = DataLoader(det_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate_det, pin_memory=True)
        metrics['mAP'] = round(eval_det(model, det_loader, device), 4)

    if 'seg' in selected_tasks:
        print("\nEvaluating Semantic Segmentation...")
        seg_ds = VocHiddenDataset(args.data_root / 'mini_voc_seg')
        seg_loader = DataLoader(seg_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
        metrics['mIoU'] = round(eval_seg(model, seg_loader, device), 4)

    if 'cls' in selected_tasks:
        print("\nEvaluating Image Classification...")
        cls_ds = ImagenetteDataset(args.data_root / 'imagenette_160')
        cls_loader = DataLoader(cls_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
        metrics['Top-1'] = round(eval_cls(model, cls_loader, device), 4)
    
    print("\n--- Evaluation Results ---")
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    main()