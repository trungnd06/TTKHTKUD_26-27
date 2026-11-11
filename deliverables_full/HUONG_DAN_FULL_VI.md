# Kết quả full cohort — thay cho các số liệu pilot trong hướng dẫn cũ

42 người; 24.200 trial (20.801 standard, 3.399 target); loại 240 trial; không loại người theo QC.

CNN: AUC 0,936 ± 0,031; balanced accuracy 0,866 ± 0,039; accuracy 0,887. SVM: AUC 0,907; accuracy 0,912. CNN tốt hơn theo AUC và balanced accuracy, SVM cao hơn theo accuracy.

5 fold theo người: 26–27 train, 7 validation, 8–9 test. Một base seed, chưa kiểm định ý nghĩa thống kê. Pz/P3/O2 có tổng attribution lớn; che 0,4–0,5 s giảm AUC trung bình khoảng 0,0195. Không chứng minh tín hiệu chỉ là P300; còn nhiễu và thông tin vận động.

Sử dụng Report_EN.pdf và Presentation_EN.pptx trong deliverables_full/. Giữ epochs_full.npz và preparation_summary.json từ notebook 01 để tái lập. Tỷ lệ đóng góp 20% mỗi người vẫn cần xác nhận theo công việc thực tế.
