# BSQAv2 — Tổng Quan Đồ Án: Data Mining + Deep Learning

> Tài liệu này dùng để nắm cốt lõi đồ án, giải thích với cô/teammates, chuẩn bị báo cáo, và biết cách chạy các phần chính.

---

## 1. Đồ án này làm gì?

**BSQAv2** là hệ thống nhận diện và đánh giá cú đánh cầu lông dựa trên skeleton pose.

Input chính:

```text
Video một cú đánh cầu lông
```

Output chính:

```text
1. Loại cú đánh: smash / clear / drop_shot / net_shot / lift
2. Độ tin cậy dự đoán
3. Pose quality / reliability
4. Technique quality score 0-100
5. Feedback kỹ thuật dạng rule-based
6. Visualization để giải thích model
```

Pipeline tổng quát:

```text
Video
→ MediaPipe Pose
→ COCO-17 keypoints
→ Pose quality control
→ Preprocessing skeleton sequence
→ Data Mining branch hoặc Deep Learning branch
→ Stroke classification
→ Quality assessment bằng DTW + biomechanics rules
→ Streamlit Observatory demo
```

5 class của bài toán:

| Class | Ý nghĩa |
|---|---|
| `smash` | Đập cầu |
| `clear` | Phông cầu |
| `drop_shot` | Bỏ nhỏ từ cuối sân |
| `net_shot` | Bỏ nhỏ trên lưới |
| `lift` | Nâng cầu |

---

## 2. Dữ liệu và biểu diễn skeleton

### 2.1 Từ video sang skeleton

Video được xử lý bằng **MediaPipe Pose**.

MediaPipe ban đầu cho ra 33 landmarks. Project map về format **COCO-17** để thống nhất với pipeline.

Mỗi frame có:

```text
17 keypoints × 2 coordinates = 34 values
```

Một sequence sau preprocessing có shape:

```text
(T, N, C) = (64, 17, 2)
```

Trong đó:

| Ký hiệu | Giá trị | Ý nghĩa |
|---|---:|---|
| `T` | 64 | số frame cố định sau padding/truncation |
| `N` | 17 | số keypoint COCO |
| `C` | 2 | tọa độ x, y |

File liên quan:

```text
src/utils/video_to_csv.py
src/data/preprocessing.py
src/data/dataset.py
```

---

### 2.2 Preprocessing skeleton

Pipeline preprocessing:

```text
raw keypoints
→ replace NaN/inf bằng 0
→ interpolate missing joints
→ normalize hip-centered, torso-scaled
→ pad/truncate về 64 frames
```

Chi tiết:

1. **Interpolate missing joints**
   - Nếu một joint bị missing ở vài frame, nội suy từ frame trước/sau.

2. **Normalize hip-centered**
   - Lấy midpoint của left hip và right hip làm tâm.
   - Mọi tọa độ được trừ đi hip center.

3. **Torso-scaled**
   - Scale skeleton theo khoảng cách hip center → shoulder center.
   - Giúp giảm ảnh hưởng của camera distance/người to nhỏ khác nhau.

4. **Pad/truncate về 64 frames**
   - Nếu sequence dài hơn 64: center crop.
   - Nếu ngắn hơn 64: lặp frame đầu/cuối.

Lý do cần 64 frames:

> Deep Learning model cần input có shape cố định. 64 frames đủ để bao phủ một cú đánh ngắn, nhưng không quá dài để training nặng.

---

## 3. Hai hướng chính của đồ án

Đồ án có 2 nhánh chính:

```text
A. Data Mining branch
B. Deep Learning branch
```

So sánh nhanh:

| Tiêu chí | Data Mining | Deep Learning |
|---|---|---|
| Input | feature thủ công | raw skeleton sequence |
| Đại diện | Random Forest, Decision Tree | GCN + BiLSTM + Attention |
| Ưu điểm | dễ giải thích, kết quả mạnh | học spatial-temporal tự động |
| Nhược điểm | phụ thuộc feature engineering | cần nhiều data, nhạy noise |
| Kết quả hiện tại | tốt nhất numerically | tốt cho research/attention |

---

# PART A — DATA MINING

## 4. Mục tiêu Data Mining

Phần Data Mining trả lời câu hỏi:

> Nếu ta trích xuất các đặc trưng chuyển động có ý nghĩa sinh cơ học từ skeleton, các model cổ điển có phân loại cú đánh tốt không?

Thay vì đưa raw `(64, 17, 2)` vào model, nhánh này chuyển skeleton thành bảng feature.

Ví dụ feature:

```text
contact_height
impact_frame
swing_phase_ratio
speed_right_wrist_max
speed_left_wrist_max
angle_right_elbow_mean
angle_left_elbow_mean
hip_center_speed
wrist displacement
elbow velocity
shoulder movement
```

File chính:

```text
src/data/biomechanics.py
src/data/rf_baseline.py
src/data/dm_analysis.py
```

---

## 5. Biomechanical features là gì?

**Biomechanical features** là các đặc trưng mô tả chuyển động cơ thể.

Ví dụ:

### 5.1 Wrist speed

Tốc độ cổ tay.

Ý nghĩa:

- Smash thường có wrist speed cao.
- Net shot thường có wrist speed thấp.

### 5.2 Contact height

Độ cao điểm tiếp xúc ước lượng từ wrist/arm position.

Ý nghĩa:

- Smash và clear thường tiếp xúc cao.
- Lift/net shot có thể thấp hơn hoặc compact hơn.

### 5.3 Elbow angle

Góc khuỷu tay.

Ý nghĩa:

- Smash/clear cần arm extension tốt.
- Nếu khuỷu tay không duỗi đủ, cú đánh có thể yếu hoặc không đúng kỹ thuật.

### 5.4 Impact frame

Frame có chuyển động mạnh nhất, thường gần thời điểm impact.

Ý nghĩa:

- Giúp mô tả timing của cú đánh.

### 5.5 Swing phase ratio

Tỉ lệ pha vung tay trong toàn bộ sequence.

Ý nghĩa:

- Một số cú đánh có preparation dài hơn, một số cú đánh compact hơn.

---

## 6. Data Mining models

### 6.1 Random Forest

Random Forest là model chính của nhánh Data Mining.

Ý tưởng:

```text
Nhiều decision trees
→ mỗi tree học một phần pattern
→ voting / averaging
→ prediction cuối cùng ổn định hơn một cây đơn
```

Lý do phù hợp:

- Feature đã có ý nghĩa domain.
- Dataset không quá lớn.
- RF xử lý feature nonlinear tốt.
- RF ít overfit hơn Decision Tree đơn.
- Có feature importance để giải thích.

Kết quả hiện tại:

```text
Accuracy:    0.7172
F1 macro:    0.7134
F1 weighted: 0.7161
```

Giải thích các số:

| Metric | Giá trị | Ý nghĩa |
|---|---:|---|
| Accuracy | 0.7172 | khoảng 71.72% samples được phân loại đúng |
| F1 macro | 0.7134 | trung bình F1 của 5 class, coi class nào cũng quan trọng như nhau |
| F1 weighted | 0.7161 | F1 trung bình có xét số lượng sample mỗi class |

Vì sao cần F1 macro?

> Dataset có thể lệch class. Accuracy cao chưa chắc tốt nếu model chỉ học class đông. F1 macro giúp xem model có công bằng hơn giữa các class không.

Per-class RF F1:

| Class | RF F1 | Nhận xét |
|---|---:|---|
| smash | 0.8708 | tốt nhất, chuyển động mạnh và rõ |
| clear | 0.7016 | khá tốt |
| drop_shot | 0.7335 | khá tốt |
| net_shot | 0.6486 | trung bình |
| lift | 0.6126 | khó hơn, dễ nhầm với clear/net |

---

### 6.2 Decision Tree

Decision Tree dùng chủ yếu để giải thích.

Ý tưởng:

```text
if feature_A > threshold:
    predict class X
else:
    check feature_B
```

Ưu điểm:

- Dễ vẽ cây.
- Dễ giải thích rule.
- Phù hợp phần Data Mining: entropy, information gain, interpretability.

Nhược điểm:

- Dễ overfit.
- Kém ổn định hơn Random Forest.

Kết quả hiện tại trong report:

```text
Accuracy khoảng 0.5427 ± 0.0092
```

Cách giải thích:

> Decision Tree không phải model mạnh nhất, nhưng giúp phân tích feature nào tạo ra split tốt và giúp báo cáo phần explainability.

---

## 7. Feature importance / entropy / mutual information

Nhánh Data Mining có thêm phân tích:

```text
feature importance
entropy analysis
mutual information
correlation
decision tree rules
```

Mục tiêu:

> Không chỉ biết model dự đoán gì, mà còn biết vì sao feature đó quan trọng.

Ví dụ top RF features trong artifact hiện tại:

```text
speed_hip_center_maxframe
speed_right_elbow_maxframe
speed_left_elbow_maxframe
velx_left_elbow_mean
velx_right_elbow_mean
vely_right_wrist_netdisp
speed_left_wrist_maxframe
impact_frame
contact_height
swing_phase_ratio
```

Cách nói khi báo cáo:

> Các feature quan trọng chủ yếu liên quan đến timing, tốc độ cổ tay/khuỷu tay, độ cao tiếp xúc và chuyển động thân người. Điều này hợp lý với domain cầu lông vì các cú đánh khác nhau khác nhau ở tốc độ, điểm tiếp xúc và pha vung tay.

---

## 8. Cách train / chạy Data Mining

### 8.1 Train Random Forest và export artifact

```bash
cd BSQAv2
../.venv/Scripts/python.exe src/data/rf_baseline.py --export-artifact
```

Output quan trọng:

```text
results/rf_baseline/rf_results.json
results/rf_baseline/rf_confusion_matrix.png
results/rf_baseline/rf_confusion_matrix_norm.png
results/rf_baseline/rf_feature_importance.png
webapp/artifacts/models/rf_baseline/rf_model_bundle.joblib
```

### 8.2 Chạy Data Mining analysis

Nếu cần regenerate analysis:

```bash
cd BSQAv2
../.venv/Scripts/python.exe src/data/dm_analysis.py
```

Output:

```text
results/dm_analysis/decision_tree_results.json
results/dm_analysis/decision_tree_rules.txt
results/dm_analysis/entropy_analysis.json
results/dm_analysis/mutual_information.csv
results/dm_analysis/*.png
```

---

# PART B — DEEP LEARNING

## 9. Mục tiêu Deep Learning

Phần Deep Learning trả lời câu hỏi:

> Nếu đưa trực tiếp skeleton sequence vào neural network, model có tự học được quan hệ không gian-thời gian của cú đánh không?

Input:

```text
(B, T, N, C) = (batch_size, 64, 17, 2)
```

Model chính:

```text
GCN + BiLSTM + Temporal Attention
```

File chính:

```text
src/models/gcn.py
src/models/bilstm.py
src/models/attention.py
src/models/gcn_bilstm_attn.py
train.py
```

---

## 10. Kiến trúc Deep Learning chính

Pipeline model:

```text
Input skeleton sequence (B, 64, 17, 2)
→ Add velocity channels
→ Spatial GCN
→ BiLSTM
→ Temporal Attention
→ Classifier head
→ Stroke logits (B, 5)
```

Full architecture:

```text
(B, 64, 17, 2)
→ add_velocity_torch
→ (B, 64, 17, 4)
→ SpatialGCN
→ (B, 64, 128)
→ TemporalBiLSTM
→ (B, 64, 256)
→ TemporalAttention
→ context vector (B, 256)
→ classifier
→ logits (B, 5)
```

---

## 11. Vì sao thêm velocity channel?

Input ban đầu chỉ có position:

```text
x, y
```

Model thêm velocity:

```text
dx = x[t] - x[t-1]
dy = y[t] - y[t-1]
```

Sau đó mỗi joint có:

```text
x, y, dx, dy
```

Shape đổi từ:

```text
(B, 64, 17, 2)
```

thành:

```text
(B, 64, 17, 4)
```

Lý do:

> Trong cầu lông, tốc độ chuyển động rất quan trọng. Smash nhanh, net shot chậm/compact. Nếu chỉ dùng position, model khó nhận ra speed pattern.

---

## 12. Spatial GCN

File:

```text
src/models/gcn.py
```

Mục tiêu:

> Học quan hệ giữa các khớp trong cùng một frame.

Skeleton là graph, không phải vector thường.

Ví dụ edge:

```text
shoulder → elbow → wrist
hip → knee → ankle
left shoulder ↔ right shoulder
left hip ↔ right hip
```

Công thức graph convolution:

```text
H_out = ReLU(A_norm @ H_in @ W)
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| `H_in` | feature của joints |
| `A_norm` | adjacency matrix đã normalize |
| `W` | learnable weight |
| `H_out` | joint features sau GCN |

Adjacency matrix:

```text
A_norm = D^(-1/2) × (A + I) × D^(-1/2)
```

Ý nghĩa:

- `A`: kết nối xương giữa các joints.
- `I`: self-loop để joint giữ thông tin của chính nó.
- Normalize để tránh node nhiều cạnh làm scale quá lớn.

Hyperparameters GCN:

| Parameter | Value | Ý nghĩa |
|---|---:|---|
| `GCN_HIDDEN_DIM` | 128 | vector feature sau GCN cho mỗi frame |
| `GCN_NUM_LAYERS` | 3 | số lớp graph convolution |
| `GCN_POOL` | `joint_attn` | attention pooling qua 17 joints |

Output GCN:

```text
(B, 64, 128)
```

Nghĩa là mỗi frame được tóm tắt thành vector 128 chiều.

---

## 13. BiLSTM

File:

```text
src/models/bilstm.py
```

Mục tiêu:

> Học chuyển động theo thời gian.

Một cú đánh có nhiều phase:

```text
preparation
→ swing
→ impact
→ follow-through
```

BiLSTM đọc sequence theo hai chiều:

```text
forward:  frame 1 → frame 64
backward: frame 64 → frame 1
```

Lý do dùng bidirectional:

> Khi phân loại toàn bộ clip, model có thể dùng cả thông tin trước và sau impact. Ví dụ follow-through cũng giúp phân biệt smash/clear/drop.

Hyperparameters BiLSTM:

| Parameter | Value | Ý nghĩa |
|---|---:|---|
| `BILSTM_HIDDEN_DIM` | 128 | hidden size mỗi chiều |
| `BILSTM_NUM_LAYERS` | 2 | số layer LSTM |
| `BILSTM_DROPOUT` | 0.2 | giảm overfitting |
| Bidirectional | True | đọc 2 chiều |

Vì bidirectional:

```text
Output dim = 128 × 2 = 256
```

Output BiLSTM:

```text
(B, 64, 256)
```

---

## 14. Temporal Attention

File:

```text
src/models/attention.py
```

Mục tiêu:

> Tìm frame quan trọng nhất trong sequence.

Input:

```text
(B, 64, 256)
```

Output:

```text
context vector: (B, 256)
attention weights: (B, 64, 64)
```

Hyperparameters:

| Parameter | Value | Ý nghĩa |
|---|---:|---|
| `ATTENTION_HEADS` | 4 | multi-head attention |
| `ATTENTION_DIM` | 256 | bằng output BiLSTM |
| Dropout | 0.2 | giảm overfitting |

Ý nghĩa attention:

> Attention cho phép model tập trung vào các frame quan trọng như impact frame, wrist acceleration, contact moment. Đây cũng là phần dùng để visualize và giải thích model trong demo.

---

## 15. Classifier head và quality head

Full model có 2 head:

```text
1. Classifier head → 5 stroke classes
2. Quality head → scalar quality score
```

Classifier:

```text
context (256)
→ Dropout
→ Linear 256 → 128
→ ReLU
→ Dropout
→ Linear 128 → 5
```

Output:

```text
logits shape = (B, 5)
```

Trong training hiện tại, wrapper `_DualHeadWrapper` dùng **classification logits** là chính:

```text
logits, quality, attention = model(x)
training loss dùng logits
```

Quality score trong app hiện tại chủ yếu đến từ:

```text
DTW + biomechanics rules
```

chứ không dựa hoàn toàn vào supervised quality head, vì dataset chưa có expert quality labels.

---

## 16. Hyperparameters huấn luyện Deep Learning

Từ `src/config.py` và `train.py`:

| Hyperparameter | Value | Ý nghĩa |
|---|---:|---|
| `SEQUENCE_LENGTH` | 64 | số frame mỗi sample |
| `NUM_KEYPOINTS` | 17 | COCO-17 skeleton |
| `COORD_DIM` | 2 | x, y |
| `COORD_DIM_VELOCITY` | 4 | x, y, dx, dy |
| `NUM_CLASSES` | 5 | 5 stroke types |
| `BATCH_SIZE` | 16 | số sample mỗi batch |
| `LEARNING_RATE` | 5e-4 | learning rate Adam |
| `NUM_EPOCHS` | 100 | số epoch tối đa |
| `EARLY_STOPPING_PATIENCE` | 15 | dừng sớm nếu val loss không cải thiện |
| `K_FOLDS` | 5 | 5-fold stratified cross-validation |
| `SEED` | 42 | reproducibility |
| Optimizer | Adam | optimizer chính |
| Scheduler | ReduceLROnPlateau | giảm LR khi val loss không giảm |
| Scheduler factor | 0.5 | LR giảm còn một nửa |
| Scheduler patience | 7 | chờ 7 epoch trước khi giảm LR |
| Minimum LR | 1e-6 | learning rate thấp nhất |
| Loss | CrossEntropyLoss | classification loss |
| AMP | enabled on CUDA | mixed precision nếu có GPU |

---

## 17. Loss function và class weights

Training dùng:

```python
nn.CrossEntropyLoss(weight=class_weights)
```

Cross Entropy Loss dùng cho multi-class classification.

Với 5 classes, model output logits:

```text
[logit_smash, logit_clear, logit_drop_shot, logit_net_shot, logit_lift]
```

Sau softmax thành probabilities.

---

### 17.1 Vì sao cần class weights?

Dataset có thể không cân bằng class.

Ví dụ class `clear` nhiều hơn `net_shot`, model có xu hướng học class đông hơn.

Để giảm bias, training tính weight theo inverse frequency:

```python
class_counts = np.bincount(all_train_labels, minlength=NUM_CLASSES)
class_weights = len(all_train_labels) / (NUM_CLASSES * class_counts)
```

Trong code có thêm `1e-8` để tránh chia 0:

```python
inv_freq = len(all_train_labels) / (NUM_CLASSES * (class_counts + 1e-8))
```

Ý nghĩa:

| Class frequency | Weight |
|---|---|
| class nhiều sample | weight thấp hơn |
| class ít sample | weight cao hơn |

Tác dụng:

> Nếu model sai ở class ít sample, loss bị phạt nặng hơn. Điều này giúp model không bỏ qua class nhỏ.

---

## 18. Optimizer, scheduler, early stopping

### 18.1 Adam optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
```

Adam tự điều chỉnh learning rate theo từng parameter, phù hợp training neural network.

---

### 18.2 ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=7,
    min_lr=1e-6,
)
```

Cách hoạt động:

```text
Nếu validation loss không giảm trong 7 epoch
→ learning rate = learning rate × 0.5
→ không thấp hơn 1e-6
```

Lý do:

> Khi model gần hội tụ, LR lớn có thể làm loss dao động. Giảm LR giúp model fine-tune tốt hơn.

---

### 18.3 Early stopping

```text
EARLY_STOPPING_PATIENCE = 15
```

Nếu validation loss không cải thiện trong 15 epoch:

```text
stop training fold hiện tại
```

Lý do:

> Tránh overfitting và tiết kiệm thời gian.

---

## 19. Cross-validation

Training dùng:

```text
5-Fold Stratified Cross-Validation
```

Ý nghĩa:

```text
Dataset chia thành 5 fold
Mỗi lần dùng 4 fold train, 1 fold validation
Lặp 5 lần
```

Stratified nghĩa là:

> Tỉ lệ class trong mỗi fold được giữ tương đối giống toàn dataset.

Lý do dùng 5-fold:

- Đánh giá ổn định hơn một train/val split duy nhất.
- Giảm phụ thuộc may rủi vào cách chia data.
- Phù hợp báo cáo academic.

Output mỗi fold:

```text
best_model_fold0.pth
best_model_fold1.pth
best_model_fold2.pth
best_model_fold3.pth
best_model_fold4.pth
cv_summary.json
```

---

## 20. Augmentation trong Deep Learning

Khi train có flag:

```bash
--augment
```

Augmentation chỉ áp dụng cho train split, không áp dụng validation.

Lý do:

> Validation phải phản ánh dữ liệu thật, không bị biến đổi nhân tạo.

Các augmentation:

| Augmentation | Probability | Ý nghĩa |
|---|---:|---|
| time warp | 0.5 | thay đổi tốc độ clip |
| mirror flip | 0.5 | lật trái-phải |
| joint noise | 0.7 | thêm noise nhỏ vào joints |
| frame dropout | 0.3 | giả lập mất frame |

Tất cả output vẫn giữ shape:

```text
(64, 17, 2)
```

---

## 21. Kết quả Deep Learning

Final report metrics:

```text
Accuracy:    0.6563 ± 0.0372
F1 macro:    0.6483 ± 0.0408
F1 weighted: 0.6517 ± 0.0400
```

Giải thích:

| Metric | Ý nghĩa |
|---|---|
| Accuracy 0.6563 | trung bình khoảng 65.63% sample đúng |
| ± 0.0372 | độ dao động giữa 5 folds |
| F1 macro 0.6483 | trung bình F1 của 5 class |
| F1 weighted 0.6517 | F1 có xét support từng class |

Per-class DL F1:

| Class | DL F1 | Nhận xét |
|---|---:|---|
| smash | 0.824 | tốt nhất, motion rõ |
| clear | 0.644 | trung bình |
| drop_shot | 0.690 | khá |
| net_shot | 0.758 | tốt |
| lift | 0.434 | yếu nhất, dễ nhầm |

Cách giải thích với cô:

> DL model học trực tiếp skeleton sequence nên khó hơn RF vì nó phải tự học feature từ data. Với dataset skeleton còn nhiễu và chưa quá lớn, RF với engineered features đang mạnh hơn. Tuy nhiên DL vẫn có giá trị vì nó học spatial-temporal pattern và cho attention visualization.

---

# PART C — QUALITY ASSESSMENT

## 22. Mục tiêu quality assessment

Sau khi model biết loại cú đánh, hệ thống đánh giá chất lượng kỹ thuật.

Pipeline:

```text
Predicted stroke type
→ lấy reference cùng stroke type
→ DTW similarity
→ biomechanics rules
→ hybrid score
→ feedback
```

File:

```text
src/quality/dtw_scorer.py
src/quality/rules.py
src/quality/hybrid.py
src/observatory/quality_references.py
```

---

## 23. DTW similarity

DTW = Dynamic Time Warping.

Mục tiêu:

> So sánh hai chuỗi chuyển động dù tốc độ khác nhau.

Ví dụ:

```text
User smash sequence
vs
Curated reference smash sequence
```

Nếu giống:

```text
DTW distance thấp
DTW similarity cao
```

Score:

```text
0-100
```

100 nghĩa là rất giống reference.

---

## 24. Biomechanics rules

Rule scorer đánh giá các tiêu chí kỹ thuật theo từng stroke.

Ví dụ:

### Smash

```text
contact height cao
wrist speed mạnh
elbow extension tốt
impact timing hợp lý
```

### Net shot

```text
wrist speed thấp
body movement compact
wrist travel ngắn
```

### Lift

```text
wrist upward motion
controlled power
body support
plausible timing
```

Mỗi rule trả về:

```text
rule name
score 0-100
observed value
target range
feedback text
```

---

## 25. Hybrid quality score

Hybrid score kết hợp:

```text
DTW similarity: 40%
Biomechanics rules: 60%
```

Công thức:

```text
quality_score = 0.4 × dtw_score + 0.6 × rule_score
```

Nếu không có reference DTW:

```text
quality_score = rule_score
```

Lý do rule weight cao hơn:

> Reference DTW phụ thuộc clip mẫu và camera angle. Rule-based score dễ giải thích hơn và ổn định hơn trong demo. Vì vậy rules chiếm 60%, DTW chiếm 40%.

Cần nói thật:

> Quality score là heuristic 2D-pose indicator, không phải điểm kỹ thuật được chuyên gia chứng nhận.

---

# PART D — PHASE 5 EVALUATION REPORT

## 26. Phase 5 làm gì?

Phase 5 tạo report tổng hợp từ artifacts có sẵn.

Command:

```bash
cd BSQAv2
../.venv/Scripts/python.exe evaluate.py --output results/evaluation_report --all-models --kfold 5
```

Output:

```text
results/evaluation_report/evaluation_summary.json
results/evaluation_report/evaluation_report.md
results/evaluation_report/model_comparison.csv
results/evaluation_report/per_class_metrics.csv
results/evaluation_report/quality_validation.csv
results/evaluation_report/model_accuracy_comparison.png
```

Phase 5 hiện report:

```text
Models: 4
Per-class rows: 5
Quality validation pairs: 9
Positive quality-drop pass rate: 1.0
```

4 model trong comparison:

```text
GCN + BiLSTM + Attention final report metrics
GCN + BiLSTM + Attention local checkpoint summary
Random Forest
Decision Tree
```

---

## 27. Quality validation trong Phase 5

Mục tiêu:

> Kiểm tra quality scorer có phản ứng đúng khi skeleton bị degrade hay không.

Cách làm:

```text
Curated reference skeleton
→ score baseline
→ tạo degraded version bằng noise + dampen arm motion
→ score degraded
→ kỳ vọng degraded score thấp hơn baseline
```

Kết quả hiện tại:

```text
Quality validation pairs: 9
Positive-drop pass rate: 1.0
```

Ý nghĩa:

> Trong 9 cặp reference-vs-degraded, score degraded đều thấp hơn baseline. Đây là sanity check tốt cho quality scoring.

---

# PART E — STREAMLIT OBSERVATORY

## 28. Streamlit app dùng để làm gì?

Streamlit Observatory là demo tổng hợp toàn pipeline.

Chạy:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m streamlit run webapp/Home.py
```

Các page:

| Page | Mục đích |
|---|---|
| Home | overview và chọn curated sample |
| Full Pipeline Demo | demo end-to-end cached pipeline |
| Pose Inspector | xem skeleton, pose QC, missing joints |
| Deep Learning Inspector | xem DL probabilities, attention |
| Data Mining Motion Lab | xem RF, feature importance, entropy, MI |
| Error Analysis Lab | phân tích lỗi RF/DL |
| Training & Evaluation | metrics, confusion matrix, checkpoint inventory |
| Dataset Explorer | distribution dataset/curated samples |
| Robustness Experiment | thử degrade skeleton xem model nhạy không |
| Custom Upload | upload video thật, chạy MediaPipe + RF/DL optional + quality |

---

## 29. Demo path nên dùng khi báo cáo

Thứ tự demo khuyến nghị:

```text
1. Home
2. Full Pipeline Demo
3. Pose Inspector
4. Data Mining Motion Lab
5. Deep Learning Inspector
6. Training & Evaluation
7. Robustness Experiment
8. Custom Upload nếu còn thời gian
```

Lưu ý:

```text
Curated mode = ổn định nhất cho bảo vệ
Custom Upload = bonus/beta path
```

---

# PART F — CÁCH CHẠY

## 30. Cài môi trường

Từ repo root:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m pip install -r requirements.txt
```

---

## 31. Chạy app

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m streamlit run webapp/Home.py
```

---

## 32. Chạy test

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m unittest tests/test_observatory_artifacts.py tests/test_observatory_pipeline.py tests/test_scan_curated_candidates.py tests/test_observatory_dl_inference.py tests/test_build_curated_manifest.py tests/test_webapp_components.py tests/test_deep_learning_viz.py tests/test_data_mining_viz.py tests/test_error_analysis_viz.py tests/test_eval_viz.py tests/test_dataset_viz.py tests/test_robustness_viz.py tests/test_upload_pipeline.py tests/test_quality_dtw.py tests/test_quality_rules.py tests/test_quality_hybrid.py tests/test_quality_references.py tests/test_evaluation_report.py
```

Latest:

```text
Ran 73 tests
OK
```

---

## 33. Chạy Phase 5 report

```bash
cd BSQAv2
../.venv/Scripts/python.exe evaluate.py --output results/evaluation_report --all-models --kfold 5
```

---

## 34. Train Deep Learning model

Full proposed model:

```bash
cd BSQAv2
../.venv/Scripts/python.exe train.py --model gcn_bilstm_attn --epochs 100 --augment
```

Train một fold để test:

```bash
cd BSQAv2
../.venv/Scripts/python.exe train.py --model gcn_bilstm_attn --epochs 5 --quick-test --fold 0
```

Các model choices:

```text
lstm_baseline
bilstm_baseline
gcn_lstm
gcn_bilstm
gcn_bilstm_attn
```

---

## 35. Train Data Mining Random Forest

```bash
cd BSQAv2
../.venv/Scripts/python.exe src/data/rf_baseline.py --export-artifact
```

---

## 36. TensorBoard

Sau khi train DL:

```bash
tensorboard --logdir=runs/<run_name>
```

Ví dụ:

```bash
tensorboard --logdir=runs/gcn_bilstm_attn_20260528_095136
```

---

# PART G — CÁCH GIẢI THÍCH KHI BÁO CÁO

## 37. Nếu cô hỏi: “Tại sao có cả Data Mining và Deep Learning?”

Trả lời:

> Data Mining giúp khai thác các feature sinh cơ học dễ hiểu và cho kết quả mạnh với Random Forest. Deep Learning giúp học trực tiếp skeleton sequence, mô hình hóa quan hệ không gian-thời gian bằng GCN + BiLSTM + Attention. Hai hướng bổ sung cho nhau: một hướng mạnh về hiệu năng và interpretability feature, một hướng mạnh về kiến trúc học biểu diễn và attention visualization.

---

## 38. Nếu cô hỏi: “Tại sao Random Forest tốt hơn Deep Learning?”

Trả lời:

> Vì feature thủ công đã encode sẵn domain knowledge như wrist speed, elbow angle, contact height và impact timing. Dataset skeleton còn nhiễu và chưa đủ lớn để Deep Learning vượt trội. Deep Learning phải tự học các feature đó từ raw skeleton nên khó hơn. Tuy nhiên Deep Learning vẫn quan trọng vì học spatial-temporal pattern tự động và có attention để giải thích.

---

## 39. Nếu cô hỏi: “Attention có ý nghĩa gì?”

Trả lời:

> Attention giúp model gán trọng số cao hơn cho các frame quan trọng, ví dụ impact frame hoặc giai đoạn tăng tốc cổ tay. Vì vậy ngoài classification, attention còn giúp visualize model đang tập trung vào phần nào của cú đánh.

---

## 40. Nếu cô hỏi: “Quality score có đáng tin không?”

Trả lời:

> Quality score là heuristic indicator dựa trên DTW similarity và biomechanics rules. Vì dataset chưa có nhãn chất lượng do huấn luyện viên gán, hệ thống không claim đây là điểm chuẩn chuyên gia. Nó là chỉ báo kỹ thuật từ 2D pose, dùng để feedback tương đối và giải thích được.

---

## 41. Nếu cô hỏi: “DTW dùng để làm gì?”

Trả lời:

> DTW so sánh hai chuỗi chuyển động cùng loại cú đánh ngay cả khi tốc độ thực hiện khác nhau. Ví dụ một người smash nhanh hơn reference, DTW vẫn align được các phase tương ứng như preparation, swing, impact, follow-through.

---

## 42. Nếu cô hỏi: “Custom Upload có thật sự chạy live không?”

Trả lời:

> Có. Custom Upload lưu video tạm, chạy MediaPipe frame-by-frame, convert sang COCO-17, pose QC, preprocessing, RF prediction, optional DL inference, DTW/rule quality scoring. Nhưng nhóm đánh dấu beta vì live MediaPipe và DL CPU có thể chậm. Curated cached mode là path ổn định hơn cho bảo vệ.

---

## 43. Nếu cô hỏi: “Hạn chế của hệ thống?”

Các hạn chế nên nói thật:

```text
1. Skeleton-only, chưa có racket/shuttle/court context.
2. 2D pose phụ thuộc camera angle.
3. Quality score chưa có expert-labeled ground truth.
4. Custom Upload chậm vì MediaPipe chạy live.
5. Deep Learning cần thêm data sạch/lớn để vượt RF.
```

---

## 44. Nếu cô hỏi: “Đóng góp chính của đồ án?”

Có thể nói:

```text
1. Xây dựng pipeline video → skeleton → classification → quality feedback.
2. So sánh Data Mining và Deep Learning cho badminton stroke recognition.
3. Thiết kế biomechanical features và RF baseline mạnh.
4. Xây dựng GCN + BiLSTM + Attention cho skeleton spatial-temporal learning.
5. Thêm quality assessment bằng DTW + biomechanics rules.
6. Xây dựng Streamlit Observatory để demo, visualize, inspect errors, robustness, upload.
```

---

# PART H — TÓM TẮT MỘT PHÚT

Nếu cần nói cực ngắn trong 1 phút:

> Đồ án BSQAv2 nhận diện và đánh giá cú đánh cầu lông từ video. Video được chuyển thành skeleton COCO-17 bằng MediaPipe, sau đó normalize và đưa vào hai nhánh. Nhánh Data Mining trích xuất biomechanical features như wrist speed, elbow angle, contact height và dùng Random Forest/Decision Tree; Random Forest đạt accuracy 71.7% và là model mạnh nhất. Nhánh Deep Learning dùng GCN + BiLSTM + Attention để học trực tiếp skeleton sequence: GCN học quan hệ khớp, BiLSTM học chuyển động theo thời gian, Attention tìm frame quan trọng; accuracy final khoảng 65.6%. Sau classification, hệ thống có quality assessment bằng DTW similarity với reference và biomechanics rules, tạo quality score 0-100 và feedback. Toàn bộ pipeline được tích hợp vào Streamlit Observatory để demo video, skeleton, prediction, attention, data mining analysis, evaluation, robustness và custom upload.

---

# PART I — Checklist trước khi báo cáo

Trước buổi báo cáo nên kiểm tra:

```bash
cd BSQAv2
../.venv/Scripts/python.exe -m streamlit run webapp/Home.py
```

Mở các page chính:

```text
Home
Full Pipeline Demo
Pose Inspector
Data Mining Motion Lab
Deep Learning Inspector
Training & Evaluation
Custom Upload nếu cần
```

Chạy report:

```bash
../.venv/Scripts/python.exe evaluate.py --output results/evaluation_report --all-models --kfold 5
```

Chạy tests nếu cần:

```bash
../.venv/Scripts/python.exe -m unittest tests/test_observatory_artifacts.py tests/test_observatory_pipeline.py tests/test_scan_curated_candidates.py tests/test_observatory_dl_inference.py tests/test_build_curated_manifest.py tests/test_webapp_components.py tests/test_deep_learning_viz.py tests/test_data_mining_viz.py tests/test_error_analysis_viz.py tests/test_eval_viz.py tests/test_dataset_viz.py tests/test_robustness_viz.py tests/test_upload_pipeline.py tests/test_quality_dtw.py tests/test_quality_rules.py tests/test_quality_hybrid.py tests/test_quality_references.py tests/test_evaluation_report.py
```

Expected:

```text
Ran 73 tests
OK
```
