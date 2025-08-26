# HAM10000 Skin Lesion Detection & Classification

Ứng dụng Flask phát hiện vùng tổn thương da (YOLO) và phân loại 7 bệnh lý trên bộ dữ liệu HAM10000 bằng CNN tùy biến.

---

### Badges
- CI/CD: (chưa cấu hình)
- License: MIT

---

### Mục lục
- [Giới thiệu](#giới-thiệu)
- [Cài đặt](#cài-đặt)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [API Usage](#api-usage)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Dữ liệu & Huấn luyện](#dữ-liệu--huấn-luyện)
- [Đóng góp](#đóng-góp)
- [License](#license)
- [Liên hệ](#liên-hệ)

---

### Giới thiệu
Backend cung cấp API:
- Phát hiện vùng nghi ngờ bằng `YOLO` và cắt ảnh tổn thương.
- Phân loại 7 lớp bệnh: `akiec, bcc, bkl, df, mel, nv, vasc` bằng `CustomCNN`.
Frontend tĩnh nằm trong `static/` (nếu có `index.html`, server sẽ phục vụ ở `/`).

### Cài đặt
Yêu cầu: Python 3.9+ (khuyến nghị), pip, virtualenv.

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```

Lưu ý: Model YOLO (`routes/object_detection.pt`) và trọng số phân loại (`routes/classification_model.pth`) cần được đặt đúng đường dẫn trước khi chạy.

### Chạy ứng dụng
```bash
python main.py
```
Mặc định chạy tại `http://localhost:5000`.

### API Usage
- Sức khỏe hệ thống:
```http
GET /api/object-detection/health
```

- Dò bằng base64:
```http
POST /api/object-detection/detect
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

- Dò bằng upload file:
```http
POST /api/object-detection/detect/file
Content-Type: multipart/form-data
file=@lesion.jpg
```

Phản hồi mẫu:
```json
{
  "status": "success",
  "objects": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.87,
      "class": "mel",
      "classification_confidence": 0.91,
      "disease_probabilities": {"mel": {"percentage": 76.5, "description": "Melanoma ..."}, "nv": {"percentage": 12.3, "description": "..."}}
    }
  ]
}
```

### Cấu trúc thư mục
```
HAM10000-skin/
  main.py                # Flask app, serve static và đăng ký blueprint
  routes/
    object_detection.py  # API, nạp YOLO + CNN, xử lý ảnh
    train_FN.py          # Huấn luyện CNN (ví dụ)
  static/                # Frontend tĩnh (index.html, favicon, ...)
  random_samples/        # Ảnh mẫu thử nghiệm
  requirements.txt
  README.md
```

### Dữ liệu & Huấn luyện
- Bộ dữ liệu: HAM10000 (tham khảo giấy phép và nguồn phát hành chính thức).
- Script huấn luyện mẫu: `routes/train_FN.py` (sửa đường dẫn `data_split/...`).
- Kết quả huấn luyện lưu: `best_model_finetuned_v4.pth` (ví dụ), khi triển khai API chuyển sang `routes/classification_model.pth` (theo định dạng load hiện tại: `checkpoint['model_state_dict']`).

### Đóng góp
1. Fork repo, tạo nhánh tính năng: `feat/ten-tinh-nang`
2. Commit theo Conventional Commits: `feat: ...`, `fix: ...`
3. Mở Pull Request mô tả rõ thay đổi, kèm hướng dẫn test.

### License
MIT. Xem file `LICENSE`.

### Liên hệ
- Mở issue trên repository này
- Email: (điền email của bạn)
