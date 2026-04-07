# BSQAv2 — Phase 0 Completion + Phase 1 Implementation

## Tổng Quan

Hoàn tất các điểm còn thiếu trong Phase 0 (Project Setup), sau đó xây dựng toàn bộ Phase 1 (Data Expansion pipeline + Data Collection Guide).

---

## Phase 0 — Gap Analysis & Fixes

Sau khi kiểm tra kỹ, Phase 0 **gần hoàn tất** nhưng vẫn thiếu vài thành phần theo plan:

| File Cần Có (theo plan) | Status | Hành Động |
|---|---|---|
| `src/config.py` | ✅ Có, đầy đủ | — |
| `src/data/dataset.py` | ✅ Có, K-Fold + augment-aware | — |
| `src/data/preprocessing.py` | ✅ Có, hip-center + interpolation | — |
| `src/data/skeleton.py` | ✅ Có, adjacency matrix builder | — |
| `src/models/lstm_baseline.py` | ✅ Có | — |
| `src/utils/visualization.py` | ❌ **Thiếu** | Copy từ v1 + mở rộng |
| `src/utils/metrics.py` | ❌ **Thiếu** | Tạo mới |
| `train.py` | ✅ Có (skeleton) | — |
| `predict.py` | ✅ Có (skeleton) | — |
| `evaluate.py` | ✅ Có (skeleton) | — |
| `data/metadata.csv` | ✅ Có (header) | — |
| `requirements.txt` | ✅ Có | — |
| `.gitignore` | ✅ Có | — |

### Fixes cần thực hiện

#### [NEW] `src/utils/visualization.py`
- Copy từ BSQAv1 `visualization.py`
- Giữ nguyên `plot_skeleton()` và `plot_training_curves()`
- Thêm stub cho `plot_attention_weights()` (sẽ implement ở Phase 5)
- Fix import path (`..data.skeleton` → dùng đúng package v2)

#### [NEW] `src/utils/metrics.py`
- Classification metrics: accuracy, f1-score (macro/weighted), confusion matrix
- Utility wrapper quanh sklearn dùng cho evaluate.py sau này

---

## Phase 1 — Data Expansion Pipeline

### Proposed Changes

---

### Data Collection Tools

#### [NEW] `tools/download_clips.py`
- Đọc file CSV timestamp log (`tools/timestamp_log.csv`) do người dùng tạo thủ công
- Format CSV: `video_url, stroke_type, start_time, end_time, player_name, notes`
- Sử dụng `yt-dlp` để tải video gốc → `data/raw_videos/`
- Caching: nếu video đã được tải (dựa trên video_id), bỏ qua
- Error handling: log các video không tải được

#### [NEW] `tools/trim_clips.py`
- Đọc cùng file timestamp log
- Sử dụng `ffmpeg` (subprocess) để cắt video theo start/end time
- Output: `data/clips/{stroke_type}/{video_id}_{index}.mp4`
- Naming convention rõ ràng để truy vết nguồn gốc
- Hỗ trợ batch processing toàn bộ log

#### [NEW] `tools/timestamp_log.csv` (Template)
- File mẫu với header và 1-2 ví dụ
- Người dùng sẽ điền thủ công khi xem YouTube

---

### MediaPipe Extraction Pipeline

#### [NEW] `src/utils/video_to_csv.py`
- Input: thư mục chứa video clips (`data/clips/`)
- Process: chạy MediaPipe Pose trên từng clip
- Map MediaPipe 33 keypoints → COCO-17 format
- Quality Control filters:
  - Confidence threshold: `visibility > 0.5` cho critical joints (wrists, elbows, shoulders)
  - Jump detection: displacement > 100px giữa 2 frames → flag
  - Missing joint threshold: > 30% missing → discard clip
- Output: CSV files matching v1 schema → `data/youtube/{stroke_type}_v2.csv`
  - Columns: `id, type_of_shot, frame_count, kpt_0_x, kpt_0_y, ..., kpt_16_x, kpt_16_y`

> [!IMPORTANT]
> **MediaPipe 33-to-COCO-17 Mapping** rất quan trọng. MediaPipe có 33 landmarks nhưng chúng ta chỉ dùng 17 joints tương ứng với COCO format. Mapping table sẽ được định nghĩa rõ trong code.

---

### Data Augmentation

#### [NEW] `src/data/augmentation.py`
- 4 kỹ thuật augmentation (chỉ áp dụng cho train set):
  1. **Time warping**: resample sequence ở tốc độ 0.8x-1.2x → **resample lại về đúng 64 frames** sau khi warp
  2. **Mirror flip**: lật ngang (đổi left ↔ right keypoints)
  3. **Joint noise**: Gaussian noise nhỏ (σ=0.01-0.02) trên tọa độ
  4. **Frame dropout**: loại bỏ ngẫu nhiên 5-10% frames → **interpolate + resample lại về đúng 64 frames**

> [!IMPORTANT]
> **Output shape bất biến:** Mọi augmentation function đều nhận `(64, 17, 2)` và **luôn trả về `(64, 17, 2)`**.
> `time_warp` và `frame_dropout` thay đổi số frames trung gian, nhưng **bắt buộc resample về 64** trước khi return.
> Điều này đảm bảo tương thích với model input size mà không cần xử lý thêm ở downstream.

- Có thể compose nhiều augmentation: `compose_augmentations([time_warp, mirror, noise])`

---

### Data Collection Guide

#### [NEW] `BSQAv2/DATA_COLLECTION_GUIDE.md`
- Hướng dẫn chi tiết tìm kiếm video trên YouTube
- Từ khóa tìm kiếm hiệu quả cho từng loại cú đánh
- Tiêu chuẩn chọn video (góc quay, chất lượng, visibility)
- Cách ghi timestamp log hiệu quả nhất
- Tiêu chí cắt video: bắt đầu từ đâu, kết thúc ở đâu
- Ví dụ mẫu cho từng stroke type

---

## Open Questions — All Resolved ✅

| # | Question | Decision |
|---|----------|----------|
| 1 | `download_clips.py` — tải lại hay cache? | ✅ **Cache video đã tải** — nếu video_id đã tồn tại trong `data/raw_videos/`, skip. Dựa vào video_id extract từ URL. |
| 2 | `video_to_csv.py` — QC failed clips? | ✅ **Giữ lại flagged clips** để review thủ công. Di chuyển vào `data/flagged/` kèm log lý do fail. Không tự động discard. |

---

## Verification Plan

### Automated Tests
```bash
# Phase 0 fixes
python -c "from src.utils.visualization import plot_skeleton; print('✓ viz ok')"
python -c "from src.utils.metrics import compute_metrics; print('✓ metrics ok')"

# Phase 1 - Tools
python tools/download_clips.py --help  # Verify args parse
python tools/trim_clips.py --help      # Verify args parse

# Phase 1 - video_to_csv
python src/utils/video_to_csv.py --input data/clips/smash/ --output /tmp/test_out.csv --verify

# Phase 1 - augmentation
python -c "
from src.data.augmentation import time_warp, mirror_flip, joint_noise, frame_dropout
import numpy as np
x = np.random.randn(64, 17, 2)
for fn in [time_warp, mirror_flip, joint_noise, frame_dropout]:
    out = fn(x.copy())
    assert out.shape == (64, 17, 2), f'{fn.__name__} shape mismatch'
print('✓ All augmentation tests passed!')
"
```

### Manual Verification
- Chạy `video_to_csv.py` trên 2-3 clip thật → mở CSV kiểm tra xem skeleton data có hợp lý
- So sánh output CSV format với `data/kaggle/smash_v1.csv` → đảm bảo column names giống hệt
