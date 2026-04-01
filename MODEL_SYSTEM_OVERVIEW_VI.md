# Mo ta chi tiet he thong model DeepFake Detection

## 1) Muc tieu he thong

He thong duoc xay dung de phat hien video deepfake theo huong **video-level classification** (phan loai tren ca chuoi frame), thay vi chi danh gia tung frame rieng le.

Mo hinh hien tai trong codebase la:

- `TemporalTriStreamDetector` (mo hinh chinh dung cho train/test/inference)
- Nen tang khong gian: `TriStreamDeepFakeDetector`
- Nen tang thoi gian: `TemporalTransformer`

Kien truc nay ket hop:

- dac trung khong gian (RGB + tan so + SRM),
- hop nhat bang channel attention,
- va suy luan thoi gian bang Transformer tren chuoi frame.

---

## 2) Cong nghe va thanh phan ky thuat

### 2.1 Framework va thu vien

- **PyTorch**: xay dung model, train loop, AMP, gradient scaling.
- **efficientnet-pytorch**: backbone EfficientNet (B1/B4,...).
- **Albumentations**: pipeline augmentation va preprocessing.
- **Transformers scheduler**: `get_cosine_schedule_with_warmup`.
- **scikit-learn + scipy**: metrics, softmax, confusion matrix.
- **OpenCV**: doc anh/frame.

### 2.2 Backbone va kieu dac trung

Model su dung 3 stream song song:

1. **RGB stream**: hoc thong tin mau/surface truc tiep.
2. **Frequency stream**:
   - tao 3 kenh tan so tu frame:
     - FFT log-magnitude,
     - FFT phase,
     - DCT log-magnitude.
3. **SRM stream**:
   - ap dung high-pass filters (SRM-style) de lam noi artefacts vi mo phong/gan/synthesis.

Ca 3 stream deu su dung EfficientNet encoder khong gian dac trung (khong dung FC head goc), sau do fusion o muc vector dac trung.

### 2.3 Suy luan thoi gian

- Chuoi frame `[B, T, 3, H, W]` duoc ma hoa thanh `[B, T, D]`.
- Them CLS token va positional encoding.
- Dua qua Transformer Encoder (Pre-LN).
- Lay vector CLS de phan loai video.

---

## 3) Luong hoat dong end-to-end

## 3.1 Luong du lieu

1. **Trich xuat khuon mat** (scripts ngoai train):
   - `scripts/extract_faces.py` / `scripts/extract_from_manifest.py`
   - Ten frame theo convention:
     - `{video_stem}-{timestamp}.jpg`
2. **Dataset grouping theo video**:
   - `VideoSequenceDataset` group frame theo `video_stem`.
   - Moi sample tra ve `sequence [T, 3, H, W] + label`.
3. **Sampling frame**:
   - `uniform` (deu theo thoi gian) hoac `random`.
   - Neu it frame hon `T`: lap frame cuoi de pad.
4. **Train/Val/Test loader**:
   - `CombinedVideoDataset` ghep real + fake.
   - Co the bat `WeightedRandomSampler` de can bang lop.

## 3.2 Luong train

1. Parse arg trong `scripts/train.py`.
2. Build transforms theo muc augmentation (`light/medium/heavy`).
3. Build dataset + dataloader.
4. Khoi tao `TemporalTriStreamDetector`.
5. Chay train theo **2 phase**:
   - Phase 1: spatial-only.
   - Phase 2: full temporal end-to-end.
6. Log metrics va luu checkpoint moi epoch.
7. Luu `best_model.pth` theo metric chon (`acc/auc/f1`).

## 3.3 Luong test

1. `scripts/test.py` load checkpoint.
2. Tu checkpoint, phuc hoi:
   - backbone,
   - `n_frames`,
   - che do output (BCE/CE).
3. Chay evaluate tren test split.
4. Xuat:
   - metrics tong hop,
   - confusion matrix,
   - ROC/EER,
   - predictions CSV (neu bat).

## 3.4 Luong inference

- `scripts/inference.py` nhan:
  - 1 thu muc frame cua 1 video, hoac
  - thu muc cha chua nhieu video subdir.
- Chon/pad frame ve dung `T`.
- Tra ve:
  - nhan REAL/FAKE,
  - `prob_real`, `prob_fake`, confidence.

---

## 4) Kien truc model chi tiet

## 4.1 Tri-stream spatial encoder (`TriStreamDeepFakeDetector`)

Input frame (da normalize ImageNet):

- `x_rgb -> RGB encoder -> f_rgb [B, D]`
- `x_rgb -> freq transform (FFT/DCT) -> freq encoder -> f_freq [B, D]`
- `x_rgb -> SRM filters -> srm encoder -> f_srm [B, D]`

### Channel Attention Fusion

- Noi 3 vector: `[f_rgb, f_freq, f_srm]`.
- MLP hoc trong so mem `w_rgb, w_freq, w_srm` (softmax).
- Vector hop nhat:
  - `f_fused = w_rgb*f_rgb + w_freq*f_freq + w_srm*f_srm`

### Classifier spatial

- `LayerNorm -> Dropout -> Linear(512) -> GELU -> Linear(out_dim)`.

Luu y:

- Stream encoder **khong weight-share** de moi stream hoc bieu dien rieng.
- Feature fusion o cap vector (`D`), khong phai fusion logits.

## 4.2 Temporal module (`TemporalTriStreamDetector`)

Voi sequence:

- Frame-wise encoding -> `[B, T, D]`
- Them CLS token -> `[B, T+1, D]`
- Them sinusoidal positional encoding
- Transformer Encoder (multi-head attention, Pre-LN)
- Lay CLS output `[B, D]`
- Qua classifier ra logits

Che do output:

- Mac dinh: `FC(1) + BCEWithLogitsLoss` (binary)
- Legacy: `FC(2) + CrossEntropyLoss`

---

## 5) Ly thuyet cot loi duoc ap dung

### 5.1 Spatial forensics + Frequency forensics

- Deepfake thuong de lai dau vet:
  - texture khong tu nhien,
  - nhiu tan so cao/thap bat thuong,
  - artefacts do blending/generation.
- RGB stream hoc semantics.
- Frequency stream (FFT/DCT) hoc dau vet tan so.
- SRM stream nhan manh high-frequency residual va edge inconsistencies.

=> Ket hop 3 stream giup model nhin cung luc:

- noi dung thi giac,
- mo hinh tan so,
- va residual traces.

### 5.2 Attention-based fusion

Thay vi cong/coi trong so co dinh, mo hinh hoc trong so theo tung mau.
Neu mau co artefact tan so ro, trong so frequency stream co the cao hon.

### 5.3 Temporal reasoning

Deepfake video khong chi loi tren tung frame ma con:

- flicker theo thoi gian,
- bat dong bo giua frame,
- artifact xuat hien theo pattern ngan han.

Transformer voi self-attention cho phep:

- theo doi quan he frame-frame,
- tap trung vao frame nghi ngo nhat (qua CLS attention),
- tong hop thong tin sequence de ra quyet dinh video-level.

### 5.4 Focal loss va class imbalance

Trong train script co:

- `FocalBCELoss` (hoac Focal CE),
- `pos_weight` tinh theo ti le am/duong,
- tuy chon `WeightedRandomSampler`.

Muc tieu:

- giam anh huong class imbalance,
- tap trung hoc mau kho (hard examples),
- cai thien generalization.

### 5.5 Scheduler va on dinh toi uu

- Optimizer: `AdamW`
- Scheduler: cosine + warmup
- AMP mixed precision cho toc do/VRAM
- Gradient checkpointing (dac biet cho B4 va VRAM han che)

---

## 6) Chien luoc train 2 phase (diem rat quan trong)

## Phase 1: Spatial-only pretraining

- Dong bang temporal transformer.
- Forward dung mean-pool frame features.
- Muc tieu: lam cho bo trich xuat khong gian hoc du tot truoc.

## Phase 2: Full temporal fine-tuning

- Mo bang toan bo tham so.
- Bat temporal transformer day du.
- Dung LR thap hon cho spatial backbone (`--lr-backbone`), LR cao hon cho temporal head.

### Loi ich

- Giam bat on khi train end-to-end tu dau.
- Transformer hoc temporal tren nen dac trung spatial da "sach".
- Thuong cho ket qua val ben vung hon.

---

## 7) To chuc he thong trong codebase

### 7.1 Tang model

- `deepfake_detector/models/multistream.py`
  - `TriStreamDeepFakeDetector`
  - `ChannelAttentionFusion`
  - xu ly frequency + SRM + fusion.
- `deepfake_detector/models/temporal.py`
  - `TemporalTransformer`
  - `TemporalTriStreamDetector`
  - quan ly phase 1/2, checkpointing.

### 7.2 Tang du lieu

- `deepfake_detector/data/dataset.py`
  - group frame theo video id,
  - sampling theo sequence,
  - ghep real/fake.
- `deepfake_detector/data/transforms.py`
  - train transforms (`light/medium/heavy`)
  - val/test transforms
  - test-time augmentation helper.

### 7.3 Tang huan luyen va danh gia

- `scripts/train.py`
  - parser tham so,
  - 2-phase training,
  - loss/sampler/optimizer/scheduler,
  - log metrics va luu checkpoint.
- `scripts/test.py`
  - evaluate toan bo test set,
  - xuat metrics, confusion matrix, ROC.
- `scripts/inference.py`
  - suy luan tren 1 hoac nhieu video.

### 7.4 Tang utility

- `deepfake_detector/utils/metrics.py`
  - Accuracy, Precision, Recall, F1,
  - ACER/APCER/NPCER,
  - EER, HTER, AUC.
- `deepfake_detector/utils/logger.py`
- `deepfake_detector/utils/visualization.py`

---

## 8) Cac gia dinh va quy uoc trong he thong

- Label convention trong dataset:
  - `real = 1`, `fake = 0`.
- Input cho model temporal:
  - shape `[B, T, 3, H, W]`.
- So frame `T` phai dong nhat giua train/test/infer va checkpoint.
- Frame extraction naming phai dung convention:
  - `{video_stem}-{timestamp}.jpg`.

---

## 9) Uu diem ky thuat cua model hien tai

- Ket hop 3 mien dac trung (RGB + Frequency + SRM), khong phu thuoc 1 loai signal.
- Temporal Transformer xu ly duoc pattern theo chuoi frame.
- Chien luoc train 2 phase giup on dinh va hoi tu tot hon.
- Pipeline train/test/inference phan tach ro, de tai lap.
- Ho tro imbalance handling (focal + pos_weight + balanced sampler).
- Ho tro VRAM han che (AMP + grad checkpointing).

---

## 10) Han che va huong nang cap

### Han che

- Van phu thuoc chat luong face extraction.
- Domain gap giua dataset nguon va dataset dich (cross-dataset) co the lon.
- Quy uoc dat ten frame/thu muc can dung de grouping video khong loi.

### Huong nang cap de nghiem tuc

- Them validation theo cross-dataset trong qua trinh chon best checkpoint.
- Them temporal augmentation manh hon (frame dropout, jitter theo time).
- Ensemble nhieu backbone/nhieu `T`.
- Them calibration threshold rieng cho moi domain du lieu.
- Them monitoring drift khi deploy thuc te.

---

## 11) Tong ket

Model ban da xay dung la mot he thong **multimodal theo mien dac trung + temporal reasoning**:

- dung 3 stream de bat dau vet frame-level,
- dung Transformer de tong hop video-level,
- va dung 2-phase training de toi uu tinh on dinh va hieu qua.

Day la mot thiet ke co co so ly thuyet ro, kha manh cho bai toan deepfake detection trong boi canh du lieu thuc te co nhieu bien dong.
