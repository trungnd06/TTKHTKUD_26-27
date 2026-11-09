# Dữ liệu EEG

Repo kèm `epochs.npz` (~40 MiB) để mọi thành viên huấn luyện lại ngay sau khi clone.
Đây là dữ liệu thật đã tiền xử lý, không phải dữ liệu mô phỏng.

## Nguồn và giấy phép

- Bộ gốc: **The Nencki-Symfonia EEG/ERP dataset**.
- Bài báo: Dzianok et al. (2022), [GigaScience, giac015](https://doi.org/10.1093/gigascience/giac015).
- Dữ liệu gốc: [GigaDB 100990](https://doi.org/10.5524/100990).
- Bản OpenNeuro: [ds004621](https://openneuro.org/datasets/ds004621), DOI `10.18112/openneuro.ds004621.v1.0.4` trong metadata nguồn.
- Giấy phép dữ liệu: **CC0**, theo file `dataset_description.json` của bên phát hành.
- URL và SHA-256 từng file đã tải: `download_manifest.json`. S3 URL không chứa snapshot bất biến;
  dùng checksum để nhận diện chính xác dữ liệu đã dùng.

Trích dẫn bài báo và bộ dữ liệu khi sử dụng lại. Nhãn subject là mã nghiên cứu, không phải thành viên nhóm.

## File đã xử lý

```python
import numpy as np

with np.load('data/epochs.npz', allow_pickle=False) as data:
    X = data['X']          # (4543, 16, 154), microvolts
    y = data['y']          # (4543,), 0=standard, 1=target
    subjects = data['subjects']
    times = data['times']  # giây, tương đối với stimulus onset
    channels = data['channels']
```

| Thuộc tính | Giá trị |
|---|---|
| Người tham gia | sub-01, sub-03, sub-04, sub-05, sub-06, sub-07, sub-08, sub-09 |
| Standard (S5), class 0 | 3.904 |
| Target (S6), class 1 | 639 |
| Tổng trial giữ lại | 4.543 |
| Trial bị loại | 89 trên 4.632 event S5/S6 |
| Tần số sau resample | 128 Hz |
| Kênh | 16; tên nằm trong trường `channels` |

`processed_manifest.json` chứa SHA-256, kích thước, shape và cấu trúc của bản pilot được commit.
Chạy `python check_data.py` để kiểm tra sau khi clone.

## Tái tạo từ bản ghi gốc

```powershell
python download_data.py
python preprocess.py
```

Script tải 44 file (~5,61 GB) vào `data/raw/`. Thư mục raw được Git bỏ qua.
Preprocessing: chọn 16 kênh, average reference trên 16 kênh, Butterworth 0.5–30 Hz,
epoch −0.2 đến 1 s, baseline −0.2 đến 0 s, loại peak-to-peak >150 µV hoặc <0.1 µV, resample 128 Hz.
Distractor S7 và response marker bị loại. Không áp dụng ICA.

`preprocessing_audit.json` ghi số trial trước và sau cho từng người. Chuỗi lọc không fit từ nhãn.
Scaler dùng trong mô hình chỉ fit trên train của từng fold.
Tái tạo ở phiên bản thư viện khác có thể tạo byte/checksum khác; dùng `run_pipeline.py --mode full`
để đánh giá dữ liệu mới đó, thay vì coi nó là bản pilot đã đóng băng.

Không ghi đè `epochs.npz` để thử cấu hình khác rồi commit nhầm vào kết quả pilot.
Dùng `python preprocess.py --output data/epochs_experiment.npz` và
`python experiment.py --data data/epochs_experiment.npz --output results_experiment` cho các thử nghiệm riêng.
