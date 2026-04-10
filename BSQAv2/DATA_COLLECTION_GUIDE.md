# 🏸 Hướng Dẫn Thu Thập Dữ Liệu Video Cầu Lông — BSQAv2

> **Mục tiêu:** Thu thập 500-750 video clips mới từ YouTube để mở rộng dataset lên 5 class (smash, clear, drop_shot, net_shot, lift), tổng 650-900 samples.

---

## 1. Tìm Video Trên YouTube

### 1.1 Từ khóa tìm kiếm hiệu quả

| Cú đánh | Từ khóa chính | Từ khóa bổ sung |
|----------|---------------|-----------------|
| **Smash** | `badminton smash slow motion`, `badminton jump smash compilation` | `BWF smash highlights`, `Lee Zii Jia smash` |
| **Clear** | `badminton clear shot technique`, `badminton defensive clear` | `badminton overhead clear drill`, `coaching clear stroke` |
| **Drop shot** | `badminton drop shot slow motion`, `badminton drop shot technique` | `badminton deceptive drop`, `BWF drop shot rally` |
| **Net shot** | `badminton net shot technique`, `badminton net play compilation` | `badminton tumbling net shot`, `badminton net kill` |
| **Lift** | `badminton lift shot technique`, `badminton underarm clear` | `badminton defensive lift`, `badminton net lift` |

### 1.2 Kênh YouTube chất lượng cao

| Kênh | Ưu điểm | Link |
|------|---------|------|
| **BWF TV** | Góc quay broadcast chuẩn, VĐV đẳng cấp | youtube.com/@BWFBadmintonWorld |
| **Badminton Insight** | Video kỹ thuật, slow motion, góc quay rõ | youtube.com/@BadmintonInsight |
| **Shuttle Amazing** | Compilation highlights, nhiều cú đánh đa dạng | youtube.com/@ShuttleAmazing |
| **Coaching channels** | Drill videos, góc quay side-view tốt | Tìm "badminton coaching" |

### 1.3 Tiêu chuẩn chọn video (QUAN TRỌNG)

✅ **NÊN chọn:**
- Góc quay **side-view** hoặc **3/4 angle** — thấy rõ toàn thân
- Chất lượng **720p trở lên** — MediaPipe cần resolution tốt
- Nền tương đối sạch, **ít che khuất**
- VĐV mặc áo khác màu với nền
- **1 VĐV rõ ràng** trong khung hình (hoặc VĐV chính nổi bật)

❌ **KHÔNG chọn:**
- Góc quay từ **trên cao bird's-eye** — mất thông tin depth
- Video **quá xa** — VĐV quá nhỏ trong frame
- Nhiều VĐV **chồng chéo** che khuất nhau
- Video **đen trắng** hoặc chất lượng thấp dưới 480p
- Slow motion **quá chậm** (< 10 fps) — ít frame hữu ích

---

## 2. Ghi Timestamp Log

### 2.1 Workflow thực tế

```
1. Mở video YouTube
2. Xem và xác định từng cú đánh cần cắt
3. Ghi vào tools/timestamp_log.csv theo format bên dưới
4. Repeat cho video tiếp theo
```

### 2.2 Format file `tools/timestamp_log.csv`

```csv
video_url,stroke_type,start_time,end_time,player_name,notes
https://www.youtube.com/watch?v=abc123,smash,01:23,01:25,Viktor Axelsen,Jump smash from right court
https://www.youtube.com/watch?v=abc123,clear,02:10,02:13,Viktor Axelsen,Defensive clear to back
https://www.youtube.com/watch?v=def456,drop_shot,00:45,00:47,Tai Tzu Ying,Deceptive drop from mid-court
```

### 2.3 Quy tắc ghi timestamp

| Quy tắc | Giải thích |
|---------|------------|
| **Start time** | Bắt đầu từ lúc VĐV **chuẩn bị swing** (tay bắt đầu đưa lên/ra sau) |
| **End time** | Kết thúc sau khi **follow-through** hoàn tất (tay đã hạ xuống) |
| **Thời lượng lý tưởng** | **1.5 - 3 giây** cho mỗi clip |
| **Dư hơn thiếu** | Thà lấy dài hơn 0.5s mỗi bên, preprocessing sẽ center-crop |
| **Một URL có thể có nhiều dòng** | Cùng 1 video có thể có nhiều cú đánh khác nhau |

### 2.4 Mẹo ghi nhanh

1. **Dùng YouTube playback speed 0.5x** để xem chậm và xác định chính xác thời điểm
2. **Tạm dừng video** tại điểm bắt đầu, ghi timestamp, rồi làm tương tự cho điểm kết thúc
3. **Phím tắt YouTube**: `,` (lùi 1 frame), `.` (tiến 1 frame) — rất hữu ích khi cần chính xác
4. **Ghi notes** ngắn gọn để sau này biết clip nào tốt/xấu

---

## 3. Tiêu Chí Cắt Video — Chi Tiết Từng Loại Cú Đánh

### 3.1 Smash 💥

```
Timeline:    [───Prep───][─Swing─][Impact][─Follow─]
Start here ──┘                                    └── End here
Duration: 1.5 - 2.5s
```

- **Start:** VĐV bắt đầu đưa tay + vợt lên cao (preparation phase)
- **End:** Sau khi vợt đã đánh qua cầu và tay hạ xuống (follow-through)
- **Key:** Bao gồm cả bước nhảy (nếu jump smash)
- **Chú ý:** Nếu VĐV nhảy, bắt đầu từ lúc bắt đầu nhún chân

### 3.2 Clear 🌈

```
Timeline:    [───Prep───][──Full Swing──][Impact][──Follow──]
Start here ──┘                                            └── End here
Duration: 2.0 - 3.0s
```

- **Start:** VĐV xoay người + đưa vợt ra sau (đặc trưng: full body rotation)
- **End:** Sau khi hoàn tất follow-through (tay đung đưa phía trước)
- **Key:** Clear thường có swing dài hơn smash, arm fully extended
- **Phân biệt với smash:** Clear đánh cầu lên cao + xa, smash đánh xuống mạnh

### 3.3 Drop Shot 🪶

```
Timeline:    [───Prep (giống smash)───][──Decel──][Impact][─Short Follow─]
Start here ──┘                                                         └── End here
Duration: 1.5 - 2.5s
```

- **Start:** Giống preparation của smash (đây là yếu tố deceptive!)
- **End:** Ngay sau khi wrist decelerate và cầu rơi gần lưới
- **Key:** Phase quan trọng nhất là sự **giảm tốc đột ngột** của cổ tay
- **Chú ý:** Preparation phase CỐ TÌNH giống smash — đây là đặc trưng cần giữ

### 3.4 Net Shot 🕸️

```
Timeline:    [─Short Prep─][──Gentle Touch──][─Minimal Follow─]
Start here ──┘                                               └── End here
Duration: 1.0 - 2.0s
```

- **Start:** VĐV bắt đầu đưa tay ra phía trước gần lưới
- **End:** Sau khi hoàn tất cú chạm nhẹ
- **Key:** Động tác nhỏ, ít amplitude, chủ yếu cổ tay + ngón tay
- **Chú ý:** Clip ngắn hơn các loại khác, nhưng vẫn cần đủ frames

### 3.5 Lift ⬆️

```
Timeline:    [─Low Prep─][───Upward Swing───][Impact][─Follow─]
Start here ──┘                                              └── End here
Duration: 1.5 - 2.5s
```

- **Start:** VĐV ở tư thế thấp (gần lưới hoặc phòng thủ), tay bắt đầu đưa lên
- **End:** Sau khi cầu đã được đẩy lên cao về phía sau sân đối phương
- **Key:** Chuyển động từ dưới lên, wrist supination
- **Phân biệt với clear:** Lift đánh từ thấp lên cao, clear đánh từ overhead

---

## 4. Quy Trình Chạy Pipeline

> **Lưu ý quan trọng:** Tất cả các lệnh dưới đây đều yêu cầu bạn phải đứng ở thư mục gốc của project (vd: `F:\CODE\BSQAv2`) và kích hoạt môi trường ảo.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING='utf-8'
```

### Bước 0: Vào đúng thư mục
- Sau khi activate phải ```cd .\BSQAv2\```

### Bước 1: Ghi timestamp log
- Mở file `BSQAv2/tools/timestamp_log.csv` và nhập thông tin các cú đánh.

### Bước 2: Tải video gốc
```powershell
python .\tools\download_clips.py --log .\tools\timestamp_log.csv --dry-run  # Kiểm tra trước
python .\tools\download_clips.py --log .\tools\timestamp_log.csv           # Tải thật
```

### Bước 3: Cắt clips
```powershell
python python .\tools\trim_clips.py --log .\tools\timestamp_log.csv --dry-run # Kiểm tra trước
python .\tools\trim_clips.py --log .\tools\timestamp_log.csv                 # Cắt thật
```

### Bước 4: Trích xuất skeleton keypoints (Tự động QC)
```powershell
python BSQAv2/src/utils/video_to_csv.py --input BSQAv2/data/clips --output BSQAv2/data/youtube --flagged BSQAv2/data/flagged --verify
```

### Bước 5: Manual Check & Chỉnh sửa (Đặc biệt quan trọng)
Dù thuật toán QC tự động có tốt đến đâu, bạn VẪN CẦN kiểm tra bằng mắt để loại bỏ các clip bị che khuất hoặc MediaPipe bắt nhầm khung xương của đối thủ (đây là điểm mù của mọi QC).

**Cách 1: Càn quét hàng loạt bằng `review_clips.py` (Khuyên dùng)**
Công cụ trình chiếu liên tục các clip cùng với bộ khung xương overlay. Rất hữu ích duyệt nhanh hàng loạt video.
```powershell
python .\review_clips.py --folder .\data\clips\smash\ --approved-dir .\data\clips\ --rejected-dir .\data\flagged\
```
- Sử dụng phím tắt: `Y` (Giữ lại), `N` (Loại bỏ), `R` (Xem lại 1 đoạn), `Q` (Thoát/Lưu tiến độ ngang lưng chừng)

**Cách 2: Điều tra chuyên sâu 1 video bằng `test_skeleton.py`**
- Mở file `BSQAv2/test_skeleton.py`, sửa `VIDEO_PATH` thành đường dẫn clip bạn nghi vấn.
- Chạy: `python BSQAv2/test_skeleton.py` (Phím `q` thoát, `Space` tạm dừng để xem kỹ frame).

### Bước 6: Xoá file rác và Cập nhật lại CSV (Re-run)
Nếu ở Bước 5 bạn đánh rớt vài clips lỗi nhưng lỡ chạy Bước 4 sinh ra CSV trước đó, thì data của đoạn CSV đó đã "bẩn".
1. Xoá thẳng tay các file `.mp4` lỗi khỏi `BSQAv2/data/clips/<tên_cú_đánh>/`. (Dùng `review_clips.py` thì công cụ tự chuyển sang rejected tự động).
2. Xoá file CSV cũ tương ứng (hoặc xoá hết để làm lại cho chắc chắn):
```powershell
Remove-Item BSQAv2\data\youtube\smash_v2.csv -Force
```
3. Chạy lại lệnh ở **Bước 4**. Script sẽ tự động scan lại thư mục clips_đã_sạch để compile ra file CSV tinh khiết nhất.

---

## 5. Mục Tiêu Số Lượng

| Stroke | Cần thu thập (YouTube) | Đã có (Kaggle v1) | Tổng mục tiêu |
|--------|----------------------|-------------------|---------------|
| Smash | 100-150 clips | 50 | 150-200 |
| Lift | 100-150 clips | 50 | 150-200 |
| Net shot | 100-150 clips | 50 | 150-200 |
| Clear | 100-150 clips | 0 (mới) | 100-150 |
| Drop shot | 100-150 clips | 0 (mới) | 100-150 |
| **Total** | **500-750** | **150** | **650-900** |

> [!TIP]
> **Chiến lược hiệu quả:** Tìm 1 video highlight dài (5-10 phút) → trích xuất 10-20 clips từ đó. Hiệu quả hơn nhiều so với tìm từng video ngắn riêng lẻ.

---

## 6. Checklist Trước Khi Chạy Pipeline

- [ ] `tools/timestamp_log.csv` đã có đủ entries
- [ ] Đã cài `yt-dlp`: `pip install yt-dlp`
- [ ] Đã cài `ffmpeg`: `brew install ffmpeg`
- [ ] Đã cài `mediapipe`: `pip install mediapipe opencv-python`
- [ ] Đã test dry-run cho download và trim
- [ ] Kiểm tra flagged clips sau khi extraction
