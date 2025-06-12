# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ultralytics import YOLO

# --- 從 TIL.ipynb 複製的模型定義 ---

class UnifiedModel(nn.Module):
    def __init__(self, num_classes=10, num_det_classes=10, num_seg_classes=21):
        super(UnifiedModel, self).__init__()
        
        # --- 骨幹網路 (Backbone) ---
        yolo_base = YOLO('yolov8n.pt') 
        self.backbone = yolo_base.model.model[:10]
        backbone_out_channels = 256
        
        # --- 統一頭部 (Unified Head) ---
        self.unified_head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # --- 任務專用輸出層 (Task-specific Output Layers) ---
        self.cls_output = nn.Linear(256, num_classes)
        self.det_output = nn.Conv2d(256, 5 + num_det_classes, kernel_size=1)
        self.seg_output = nn.Conv2d(256, num_seg_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        unified_features = self.unified_head(features)

        # 分類輸出
        cls_pooled = F.adaptive_avg_pool2d(unified_features, 1).view(unified_features.size(0), -1)
        cls_out = self.cls_output(cls_pooled)
        
        # 偵測輸出
        det_out = self.det_output(unified_features)
        
        # 分割輸出
        seg_out_small = self.seg_output(unified_features)
        seg_out = F.interpolate(seg_out_small, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 返回順序：cls, det, seg
        return cls_out, det_out, seg_out

# --- 從 TIL.ipynb 複製的後處理函式 ---

def postprocess_detection(det_out, conf_threshold=0.01, nms_threshold=0.5, img_size=512):
    batch_size = det_out.shape[0]
    grid_h, grid_w = det_out.shape[2], det_out.shape[3]
    
    det_out = det_out.permute(0, 2, 3, 1).contiguous()
    det_out = det_out.view(batch_size, grid_h * grid_w, -1)
    
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    grid_xy = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 2).to(det_out.device)
    
    # 解碼預測
    det_out[..., :2] = torch.sigmoid(det_out[..., :2])
    det_out[..., 4] = torch.sigmoid(det_out[..., 4])
    det_out[..., 5:] = torch.sigmoid(det_out[..., 5:])
    
    box_xy = (det_out[..., :2] + grid_xy) / torch.tensor([grid_w, grid_h], device=det_out.device)
    
    # 統一訓練與推論的解碼方式
    box_wh = torch.sigmoid(det_out[..., 2:4]) 
    
    box_xy1 = box_xy - box_wh / 2
    box_xy2 = box_xy + box_wh / 2
    boxes = torch.cat((box_xy1, box_xy2), dim=-1)
    
    obj_conf = det_out[..., 4:5]
    class_conf, class_labels = torch.max(det_out[..., 5:], dim=-1, keepdim=True)
    final_conf = obj_conf * class_conf
    
    output_list = []
    
    for i in range(batch_size):
        mask = final_conf[i].squeeze() >= conf_threshold
        
        batch_boxes = boxes[i][mask]
        batch_scores = final_conf[i][mask].flatten()
        batch_labels = class_labels[i][mask].flatten()
        
        if batch_boxes.shape[0] == 0:
            output_list.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.long)})
            continue

        batch_boxes *= img_size

        nms_indices = torchvision.ops.nms(batch_boxes, batch_scores, nms_threshold)
        
        final_boxes = batch_boxes[nms_indices]
        final_scores = batch_scores[nms_indices]
        final_labels = batch_labels[nms_indices]
        
        output_list.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels,
        })
        
    return output_list