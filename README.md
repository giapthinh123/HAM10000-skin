# HAM10000 Skin Lesion Detection & Classification

Flask application for detecting skin lesion regions (YOLO) and classifying seven diseases on the HAM10000 dataset using a custom CNN.

---

### Badges
- CI/CD: (not configured)
- License: MIT

---

### Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Run](#run)
- [API Usage](#api-usage)
- [Project Structure](#project-structure)
- [Data & Training](#data--training)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

### Overview
Backend provides APIs to:
- Detect suspicious regions using `YOLO` and crop the lesion area.
- Classify seven classes: `akiec, bcc, bkl, df, mel, nv, vasc` using a `CustomCNN`.
Static frontend is served from `static/` (if `index.html` exists, it is served at `/`).

### Installation
Requirements: Python 3.9+ (recommended), pip, virtualenv.

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```

Note: Place YOLO model weights at `routes/object_detection.pt` and classification weights at `routes/classification_model.pth` before running.

### Run
```bash
python main.py
```
Default URL: `http://localhost:5000`.

### API Usage
- Health check:
```http
GET /api/object-detection/health
```

- Detect via base64:
```http
POST /api/object-detection/detect
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

- Detect via file upload:
```http
POST /api/object-detection/detect/file
Content-Type: multipart/form-data
file=@lesion.jpg
```

Sample response:
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

### Project Structure
```
HAM10000-skin/
  main.py                # Flask app, serves static and registers blueprint
  routes/
    object_detection.py  # API, loads YOLO + CNN, image processing
    train_FN.py          # CNN training (example)
  static/                # Static frontend (index.html, favicon, ...)
  random_samples/        # Sample images
  requirements.txt
  README.md
```

### Data & Training
- Dataset: HAM10000 (observe the official license and distribution terms).
- Example training script: `routes/train_FN.py` (update paths like `data_split/...`).
- Model output example: `best_model_finetuned_v4.pth`. For the API, place weights at `routes/classification_model.pth` (expected format: `checkpoint['model_state_dict']`).

### Contributing
1. Fork the repo and create a feature branch: `feat/your-feature`
2. Use Conventional Commits: `feat: ...`, `fix: ...`
3. Open a Pull Request with clear description and test instructions.

### License
MIT. See `LICENSE`.

### Contact
- Open an issue on this repository
- Email: (your email)
