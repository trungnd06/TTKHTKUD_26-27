# Đưa dự án lên GitHub và làm việc nhóm

Repo của nhóm: **[TTKHTKUD_26-27](https://github.com/trungnd06/TTKHTKUD_26-27)**.

## Tạo repo và push lần đầu

Trên GitHub, tạo repo trống. Không chọn tạo sẵn README, .gitignore hoặc LICENSE vì dự án đã có nội dung.
Repo private hay public do nhóm chọn. Copy URL HTTPS của repo, rồi chạy từ thư mục dự án:

```powershell
git init -b main
git add .
git status
git commit -m "Add reproducible EEG oddball miniproject"
git remote add origin https://github.com/trungnd06/TTKHTKUD_26-27.git
git push -u origin main
```

Nếu dự án đã được khởi tạo hoặc đã commit, bỏ qua các bước tương ứng. Nếu remote `origin` đã có,
kiểm tra bằng `git remote -v` trước khi đổi. Không dùng force-push để xử lý repo có sẵn nội dung.
Git có thể mở trình duyệt yêu cầu đăng nhập GitHub trong lần push đầu tiên.
Không ghi mật khẩu hoặc token vào README, URL remote hay mã nguồn.

Dữ liệu đã xử lý ~40 MiB được commit bình thường. Dữ liệu gốc 5,6 GB, `.vendor/`, `.venv/`,
cache và ZIP đã được loại bằng `.gitignore`. Các bản ghi EEG gốc riêng lẻ vượt giới hạn 100 MiB
của GitHub Git thông thường, nên thành viên tải bằng `download_data.py` khi cần.
Tham khảo [giới hạn file của GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Thêm thành viên

Vào repo trên GitHub → **Settings → Collaborators** (hoặc **Collaborators and teams**, tùy loại repo)
→ **Add people**, rồi nhập GitHub username hoặc email của từng người. Mỗi người cần chấp nhận lời mời.
Chủ repo tự giữ quyền quản trị; cấp quyền phù hợp để các bạn push branch và tạo pull request.
Nhóm chưa cung cấp tài khoản GitHub của từng thành viên, nên danh sách họ tên/MSSV trong báo cáo không thay thế bước mời này.
Xem [hướng dẫn mời collaborator của GitHub](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository).

## Quy trình mỗi thành viên

```powershell
git clone https://github.com/trungnd06/TTKHTKUD_26-27.git
cd TTKHTKUD_26-27
git switch -c feature/ten-phan-viec
```

Làm theo README để cài môi trường và chạy kiểm tra. Sau khi thay đổi:

```powershell
git status
git add <CAC_FILE_DA_SUA>
git commit -m "Describe the completed change"
git push -u origin feature/ten-phan-viec
```

Tạo pull request vào `main`, ghi rõ thay đổi, cách chạy và kết quả kiểm tra.
Chủ repo xem rồi merge. Trước khi bắt đầu việc mới, cập nhật `main`:

```powershell
git switch main
git pull --ff-only
```

Không sửa trực tiếp cùng một file Word/PDF/PPTX ở nhiều nhánh vì khó merge.
Thảo luận trước ai giữ bản tài liệu chính. Sửa mã hoặc Markdown trước, rồi xuất tài liệu khi thống nhất.
Không chạy lại generator sau khi sửa tay Word/PPTX hoặc contributions.csv nếu chưa sao lưu.

Trong `contributions.csv`, mỗi người cần ghi nhận đóng góp thực tế; 20% hiện tại là đề xuất.
Lịch sử commit và nội dung pull request có thể giúp lưu minh chứng công việc của nhóm.
