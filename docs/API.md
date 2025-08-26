## API Reference

Base URL: `http://localhost:5000`

### Health Check
GET `/api/object-detection/health`

Response
```json
{
  "status": "healthy",
  "yolo_model_loaded": true,
  "classification_model_loaded": true,
  "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
  "disease_names": {"akiec": "..."}
}
```

### Detect (base64)
POST `/api/object-detection/detect`

Headers: `Content-Type: application/json`

Body
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

Response
```json
{
  "status": "success",
  "objects": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.87,
      "class": "mel",
      "predicted_class": "mel",
      "classification_confidence": 0.91,
      "disease_probabilities": {
        "mel": {"percentage": 76.5, "description": "Melanoma ..."}
      },
      "description": "Melanoma (ung thư hắc tố)"
    }
  ]
}
```

### Detect (file upload)
POST `/api/object-detection/detect/file`

Form-Data: `file=@lesion.jpg`

Response: như trên.

### Lỗi thường gặp
- `Models not loaded`: kiểm tra `routes/object_detection.pt` và `routes/classification_model.pth` tồn tại.
- `Invalid image format`: đảm bảo base64 hợp lệ hoặc file ảnh hợp lệ.


