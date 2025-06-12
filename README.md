# Unified-OneHead 多任務模型評估腳本 (eval.py)

## 1. 總覽

本腳本 (`eval.py`) 用於評估一個**單一、已預訓練好**的 `UnifiedModel` 模型在指定任務上的效能。它可以獨立或同時評估以下三項任務：

* **物件偵測 (Object Detection)**：計算 mAP (mean Average Precision)。
* **語意分割 (Semantic Segmentation)**：計算 mIoU (mean Intersection over Union)。
* **圖片分類 (Image Classification)**：計算 Top-1 正確率 (Accuracy)。

腳本的設計符合 DL Assignment 2 的最終評估要求，即使用**單一模型權重**對所有任務進行測試。

## 2. 前置要求

在執行此腳本前，請確保滿足以下條件：

1.  **Python 環境**:
    確認已安裝所有必要的函式庫。主要包括：
    * `torch` & `torchvision`
    * `ultralytics`
    * `torchmetrics`
    * `pycocotools`
    * `numpy`
    * `Pillow`

2.  **`model.py` 檔案**:
    此評估腳本依賴於一個名為 `model.py` 的檔案，該檔案必須與 `eval.py` 位於同一目錄下。`model.py` 中必須包含您在 `TIL.ipynb` 中定義的 `UnifiedModel` 類別以及 `postprocess_detection` 函式。

3.  **資料集目錄結構**:
 ```
data/                   # <-- --data_root 參數指向這裡
  ├── mini_coco_det/
  │   └── ... (包含 images 和 annotations 的資料夾)
  ├── mini_voc_seg/
  │   └── ... (包含 images 和 masks 的資料夾)
  └── imagenette_160/
  └── ... (包含 train/val 和各類別的資料夾)
```

## 3. 使用方法

### 指令格式
```bash
python eval.py --weights [權重路徑] --data_root [資料根目錄] --tasks [任務名稱]
```

參數說明
* --weights (必須): 請使用與eval.py同路徑之**checkpoint_cls_best.pt**。
* --data_root (必須): 指向包含所有迷你資料集的根目錄（例如 ./data）。
* --tasks: (可選) 指定要評估的任務，以逗號分隔。可選值為 det, seg, cls。若設定為 all，則會評估所有三個任務。預設值為 all。
* --batch: (可選) 設定驗證時的批次大小 (batch size)。預設值為 8。
* --cpu: (可選) 加上此參數會強制使用 CPU 進行評估，即使有可用的 GPU。
