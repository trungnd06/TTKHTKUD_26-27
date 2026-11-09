# Hướng dẫn hoàn thiện và bảo vệ miniproject

## Những gì đã thực hiện

Dữ liệu thật: oddball, 8 người của Nencki–Symfonia, bản OpenNeuro ds004621.
Đã chạy tiền xử lý, majority baseline, shrinkage LDA, RBF SVM, Random Forest, CNN; đánh giá 4 fold theo người.
Kết quả chính: Compact EEG CNN có ROC-AUC trung bình cao nhất trong các mô hình đã thử.
CNN trừ ML mạnh nhất (RBF SVM) = +0.027 AUC. Xem số chi tiết trong Report_EN.pdf và results/summary.csv.
Kết quả là pilot 8 người, không phải toàn bộ 42 người hay kỷ lục của bộ dữ liệu.

## Mỗi thành viên cần tự làm và hiểu

Bảng dưới là **đề xuất**, không xác nhận ai đã thực hiện công việc. Mỗi người phải kiểm tra, chạy lại,
ghi nhận phần việc thực tế và điều chỉnh tỷ lệ trong contributions.csv (tổng 100%).
Việc được giao tên trong tài liệu không tự chứng minh đóng góp.

| Thành viên | Phần đề xuất | Minh chứng nên chuẩn bị |
|---|---|---|
| Lê Minh Đức — 24022291 | Dữ liệu, tiền xử lý | Giải thích S5/S6/S7, chạy preprocess.py, đọc audit |
| Nguyễn Đức Trung — 24022473 | Đặc trưng và ML | Giải thích 320 đặc trưng, LDA/SVM/RF, train-only scaling |
| Phạm Sỹ Toàn — 24022467 | CNN và so sánh | Vẽ lại kiến trúc, giải thích loss và validation checkpoint |
| Đỗ Hoàng Minh — 24022399 | Hình và giải thích | Đọc attribution/occlusion, nêu hạn chế suy luận |
| Đào Văn Việt — 24022489 | Tích hợp, báo cáo, tái lập | Chạy verify_results.py, tổng hợp báo cáo, kiểm tra nguồn |

Tạm đề xuất mỗi người 20%; nhóm cần xác nhận hoặc sửa theo thực tế. Có thể đổi người trình bày trong slide.

## Câu hỏi vấn đáp

1. **Single trial là gì?** Một đáp ứng EEG sau một kích thích. Mỗi trial sinh một dự đoán. ERP là trung bình nhiều trial để nhìn dạng sóng.
2. **Nhãn lấy ở đâu?** Marker S5 = thường gặp, S6 = mục tiêu; S7 = distractor bị loại. Không dùng marker bấm nút làm nhãn.
3. **Vì sao không chỉ dùng accuracy?** Lớp standard chiếm đa số. Đoán toàn standard vẫn có accuracy cao nhưng balanced accuracy và AUC chỉ 0.5.
4. **Balanced accuracy tính sao?** (Sensitivity của target + specificity của standard) / 2.
5. **ROC-AUC là gì?** Khả năng xếp điểm target cao hơn standard khi thay đổi ngưỡng. Không phải tỷ lệ dự đoán đúng ở ngưỡng 0.5.
6. **Có rò rỉ dữ liệu không?** Trong từng fold, train/validation/test là các người khác nhau. Scaler chỉ fit train; không dùng test chọn C hay epoch.
7. **Vì sao chia theo người?** Trial của cùng một người có đặc điểm tương quan. Trộn trial vào train/test sẽ không đo đúng khái quát sang người mới.
8. **Vì sao có validation?** Chọn cấu hình mà không nhìn điểm test. 2 người test, 2 validation, 4 train ở mỗi fold.
9. **Đặc trưng ML là gì?** Trung bình biên độ từng cửa sổ 50 ms trong 0–1 s, 20 cửa sổ × 16 kênh = 320.
10. **CNN học gì?** Lọc theo thời gian rồi phối hợp không gian giữa điện cực; lớp cuối tạo logit target. Kiến trúc lấy ý tưởng EEGNet, không sao chép chính xác EEGNet.
11. **Vì sao weighting?** Target hiếm; trọng số trong loss tăng chi phí bỏ sót target. Tính tỷ lệ từ train, không từ test.
12. **Deep learning có tốt hơn không?** Trả lời theo bảng kết quả: The CNN had the higher mean held-out AUC in this pilot. Không khái quát sang mọi dữ liệu từ một pilot.
13. **Integrated gradients là gì?** Tích phân gradient logit dọc đường từ baseline đến input. Baseline là trung bình kênh của train; đồ thị lấy trị tuyệt đối nên không cho chiều tác động.
14. **Occlusion là gì?** Che một cửa sổ 100 ms bằng trung bình train, đo AUC giảm bao nhiêu. Che có thể tạo tín hiệu không tự nhiên.
15. **Có chứng minh P300 không?** Không. Có thể so với ERP nhưng còn nhiễu, khác biệt thị giác và vận động bấm nút. Attribution trên da đầu không định vị nguồn não.
16. **Đã loại nhiễu mắt bằng ICA chưa?** Chưa. Có lọc, tham chiếu, baseline và loại epoch theo biên độ. Báo cáo ghi rõ giới hạn này.
17. **Tại sao không chạy 42 người?** Chọn 8 người để phù hợp tài nguyên/thời gian miniproject. Có mã mở rộng, nhưng điểm báo cáo chỉ thuộc 8 người đã chạy.
18. **Mean ± SD có phải khoảng tin cậy không?** Không. SD cho độ dao động qua 4 fold; các fold còn dùng chung một số người trong train.
19. **Dùng online được chưa?** Chưa. Lọc hai chiều dùng mẫu tương lai, và chưa đo độ trễ hay thử hệ thống online.
20. **Cải tiến tiếp theo?** Toàn bộ cohort, nhiều seed, lặp validation, kiểm soát chỉ trước kích thích và trước phản hồi, kiểm tra nhạy cảm với nhiễu.

## Trình bày và nộp

- 10 slide chính có kịch bản tổng 540 giây; 2 slide dự phòng về nguồn và đóng góp. Luyện nói và đo giờ thực tế.
- Mọi người cần hiểu toàn bộ pipeline, không chỉ phần được giao. Slide gốc cho biết điểm cá nhân phụ thuộc đóng góp và hiểu biết.
- Gửi báo cáo tiếng Anh ít nhất 1 tuần trước lịch thuyết trình cho guy.nagels@vub.be và hai thầy Trần Quốc Sang, Nguyễn Tiến Dũng (lấy email từ lớp).
- Chưa có ngày thuyết trình, tên lớp và mẫu bìa riêng. Bổ sung nếu giảng viên yêu cầu.
- Mở Word/PDF/PPTX, kiểm tra tên, tỷ lệ, biểu đồ; không nộp phần phân công đề xuất như đóng góp đã xác nhận.
- Công cụ không tự gửi email cho giảng viên.
