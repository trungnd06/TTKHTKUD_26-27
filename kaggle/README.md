# Chạy toàn bộ oddball Nencki–Symfonia trên Kaggle

Thư mục này độc lập với bản pilot 8 người ở thư mục cha. **Chưa có kết quả 42 người đi kèm**:
bạn chạy notebook để tạo kết quả mới. Không sử dụng điểm pilot 0.863 làm kết quả full.

## Có public dataset trên Kaggle không?

Ngày kiểm tra: **11/11/2026**. Tìm kiếm công khai với `Nencki`, `Nencki-Symfonia`, `Symfonia`
và `ds004621` chưa tìm được bản Kaggle có thể xác nhận đúng bộ này. Điều đó không chứng minh
Kaggle hoàn toàn không có bản sao. Không thay bằng dataset EEG/P300 khác nếu cần làm đúng bộ trong slide.

Nguồn công khai xác nhận được là [OpenNeuro ds004621](https://openneuro.org/datasets/ds004621)
và [repository của bên phát hành](https://github.com/OpenNeuroDatasets/ds004621).
Các notebook tải trực tiếp file từ S3 public của OpenNeuro, không cần token dữ liệu.
Nguồn không chứa snapshot trong URL; SHA-256, ETag và URL của từng file được lưu để nhận diện bản đã dùng.
Metadata gốc nêu CC0; cần trích dẫn [bài báo](https://doi.org/10.1093/gigascience/giac015)
và [GigaDB](https://doi.org/10.5524/100990) khi dùng dữ liệu.

## Cách chạy khuyến nghị: hai notebook

### 1. Tải và tiền xử lý, dùng CPU

1. Tạo Kaggle Notebook → Import Notebook → chọn **`01_prepare_full.ipynb`**.
2. Bật **Internet**, để Accelerator = None, rồi Run All.
3. Notebook mặc định yêu cầu đủ **42 người, chỉ task oddball**.
4. Mỗi lượt chỉ giữ bản ghi gốc của một người trong thư mục tạm do script sở hữu, xử lý xong sẽ dọn.
   Không tải cả bộ raw cùng lúc và không xóa các file bạn gắn dưới `/kaggle/input`.
5. Kiểm tra `preparation_summary.json`: bao nhiêu người được yêu cầu, giữ lại, bị loại và lý do.
6. **Save Version** với output, hoặc tải folder `nencki_full/processed` về máy và tạo Kaggle Dataset từ đó.

Output chính:

```text
/kaggle/working/nencki_full/processed/
  epochs_full.npz
  preparation_summary.json
  sub-01.npz, sub-01.json, ...
  dataset_description.json
  task-oddball_events.json
  task-oddball_eeg.json
```

Nếu phiên chạy dừng, các subject có output hợp lệ có thể được dùng lại. Để tiếp tục ở phiên khác,
gắn output đã Save làm Input và điền `RESUME_PROCESSED` trong notebook 01. Script kiểm tra checksum
của các subject được giữ lại khi ghép dữ liệu cuối cùng. File đang ghi dở không được coi là hoàn tất.

Kaggle cho lưu tối đa **20 GB output của notebook**, theo [tài liệu](https://www.kaggle.com/docs/notebooks).
Output ở đây là EEG đã xử lý và metadata, không chứa toàn bộ raw. Dữ liệu ở phiên tương tác phải được
Save Version hoặc tải xuống; đừng coi việc để file trong working là đã lưu bền vững.

### 2. Huấn luyện, bật GPU

1. Import **`02_train_full.ipynb`** vào notebook thứ hai.
2. Add Input → thêm output đã lưu của notebook 01 hoặc Kaggle Dataset bạn vừa tạo.
3. Bật **GPU** và Internet để clone mã/cài MNE nếu thiếu. Không cần cài lại PyTorch của Kaggle.
4. Notebook tìm `epochs_full.npz` dưới input. Nếu có nhiều bản, điền `DATA_PATH` rõ ràng.
5. Run All. CNN chạy CUDA khi có GPU; LDA/SVM/RF vẫn chạy CPU.
6. Sau khi kiểm tra thành công, Save Version có output hoặc tải `nencki_full_results.zip`.

Notebook huấn luyện kiểm tra metadata và checksum để tránh vô tình dùng file pilot 8 người.
Điểm mới, dự đoán, checkpoint, giải thích CNN và tóm tắt tiếng Anh nằm trong output `results/`.
**Không tự ghi đè report/PowerPoint pilot của nhóm**. Cần cập nhật các tài liệu đó sau khi có kết quả full.

Thời gian tải và huấn luyện tùy mạng, CPU/GPU và tài nguyên của phiên. RBF SVM có calibration
và tìm ba giá trị C nên có thể chạy lâu trên toàn bộ cohort. Không có thời gian hoàn thành được đảm bảo.

## Thiết kế nghiên cứu

- S5 = standard (0), S6 = target (1), loại distractor S7 và response markers.
- 16 sensor, tham chiếu trung bình 16 kênh, Butterworth 0.5–30 Hz, epoch -0.2 đến 1 s,
  baseline -0.2 đến 0 s, reject >150 µV hoặc flat <0.1 µV, resample 128 Hz. Không ICA.
- Đánh dấu gián đoạn ghi bên trong bản ghi, không lọc xuyên qua ranh giới và loại epoch đi qua chúng.
- Yêu cầu xử lý tất cả 42 người. Nếu một người còn dưới 10 trial ở một lớp sau QC, ghi rõ exclusion;
  lỗi tải hoặc đọc dữ liệu thì **dừng**, không tự bỏ qua như một ca QC.
- sub-02, sub-14, sub-17 được bài gốc nêu vấn đề tuân thủ hành vi. Phiên bản này vẫn giữ họ nếu đạt QC,
  đồng thời ghi flag để nhóm có thể làm phân tích nhạy cảm sau này. Không khẳng định 42 người đều đạt QC.
- 5 outer GroupKFold theo người. Validation là khoảng 20% subject trong development set, ít nhất 2;
  train/validation/test hoàn toàn khác người. Scaler và class weights chỉ fit train.
- Validation AUC chọn SVM C, độ sâu RF và CNN checkpoint (tối đa 50 epoch, patience 8).
- Giữ cùng cách tạo 320 ERP features và CNN như pilot. One base seed, fixed threshold 0.5.
- GPU dùng các thiết lập deterministic khi được hỗ trợ; `warn_only=True` cho phép phép toán CUDA
  chưa có implementation deterministic. Kết quả có thể thay đổi nhẹ theo thiết bị/phần mềm.
- Sau mỗi fold, lưu checkpoint và cache có fingerprint. Có thể tiếp tục fold đã hoàn tất nếu cấu hình,
  dữ liệu và checksum checkpoint trùng. Muốn tiếp tục ở phiên khác, gắn output cũ rồi điền `RESUME_RESULTS`.

## Chạy bằng CLI ngoài Kaggle

Không cần copy các file của project cha; chỉ dùng nội dung folder này và các dependencies đã cài.

```bash
python prepare_data.py --output ./work/processed
python train.py --data ./work/processed/epochs_full.npz --output ./work/results --device auto
python verify.py --data ./work/processed/epochs_full.npz --output ./work/results
```

Dùng raw đã tải trên máy hoặc Kaggle Input (không xóa raw):

```bash
python prepare_data.py --raw-root /path/to/bids-root --output ./work/processed
```

Smoke test có chủ đích trên ít người:

```bash
python prepare_data.py --raw-root ../data/raw --subjects 1 3 4 5 6 7 --output ./work/smoke
python train.py --data ./work/smoke/epochs_full.npz --output ./work/smoke_results --folds 3 --epochs 1 --device cpu --allow-partial
python verify.py --data ./work/smoke/epochs_full.npz --output ./work/smoke_results
```

`--allow-partial` chỉ dành cho kiểm tra kỹ thuật; metadata vẫn ghi đây không phải lần chạy đủ cohort.
Thay đổi cấu hình hoặc phiên bản thư viện: dùng output mới để không trộn fold cũ và mới.

## Khi có kết quả

Đọc `preparation_summary.json`, `run_metadata.json`, `verification.json`, `fold_metrics.csv`,
`summary.csv`, `RESULTS_EN.md`, `attribution.png` và `occlusion.png` trước khi kết luận.
Báo cáo số người thực tế giữ lại, số trial mỗi lớp, các exclusions và split; không ghi đủ 42 nếu QC loại người.
Mean ± SD theo fold không phải confidence interval. Attribution không chứng minh nguồn thần kinh hoặc nhân quả.
