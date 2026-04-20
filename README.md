# Wedding Image Culling Suite

> **AI-powered backend that automatically selects the best wedding photos — in seconds.**

A production-grade Python pipeline that ingests a folder of raw wedding photographs and intelligently culls them using computer vision and deep learning: detecting blur, poor exposure, eyes closed, low aesthetics, and near-duplicate burst shots, while recognising and preserving photos of enrolled VIPs (Bride, Groom, etc.).

---

## ✨ What It Does

| Filter | Technology | Rejects |
|---|---|---|
| Blur detection | Laplacian variance | Blurry / motion-blurred shots |
| Exposure check | Mean grayscale + dynamic percentiles | Under/overexposed images |
| Person detection | YOLOv8n (COCO) | Photos with no people |
| Decor-focus detection | YOLO + MTCNN | Back-of-head / décor-only shots |
| Eyes closed | MediaPipe FaceMesh + EAR | Blink shots |
| Aesthetic scoring | CLIP ViT-L/14 + AVA regression | Low-quality composition |
| Duplicate grouping | CLIP embeddings + L2 clustering | Near-identical burst duplicates |
| VIP recognition | FaceNet / InceptionResnetV1 | *(preserves)* enrolled persons |

The output is a clean timestamped folder of kept images, a `by_vip/` subfolder tree per recognised person, and a full `log.csv` audit trail with scores and rejection reasons for every image.

---

## 🚀 Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/Chhayansh-Git/Phase2.git
cd Phase2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Enrol VIPs
python scripts/enroll_vips.py "Bride" /path/to/bride/portraits
python scripts/enroll_vips.py "Groom" /path/to/groom/portraits

# 3. Run the pipeline
python pipeline.py /path/to/wedding/photos

# 4. Or start the REST API
uvicorn api.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

---

## 🏗️ Project Structure

```
photo-filter/
├── api/fastapi_app.py          # REST API (FastAPI)
├── filters/                    # Individual filter modules
│   ├── blur_filter.py
│   ├── exposure_filter.py
│   ├── person_filter.py        # YOLOv8n
│   ├── eyes_closed_filter.py   # MediaPipe FaceMesh
│   ├── face_id_filter.py       # FaceNet / MTCNN
│   ├── aesthetic_filter.py     # CLIP + regression head
│   ├── duplicate_filter.py     # CLIP + sklearn NearestNeighbors
│   ├── cropped_filter.py       # Face-edge-crop detection (available, not yet wired)
│   └── obstruction_filter.py   # IoU obstruction detection (available, not yet wired)
├── utils/fast_loader.py        # TurboJPEG + PIL image loader
├── scripts/
│   ├── enroll_vips.py          # VIP face enrolment CLI
│   └── verify_setup.py         # End-to-end smoke test
├── data/embeddings/            # Persisted VIP face embeddings
├── pipeline.py                 # Main orchestrator
├── config.py                   # All thresholds (env-overridable)
├── logging_config.py           # Logging setup
├── requirements.txt
├── yolov8n.pt                  # Bundled YOLOv8 nano weights
└── DOCUMENTATION.md            # Full technical documentation
```

---

## ⚙️ Configuration

All thresholds live in `config.py` and are overridable via environment variables — no code change needed.

| Setting | Env Var | Default | Effect |
|---|---|---|---|
| Soft blur | `BLUR_THRESHOLD` | `30.0` | Raise to reject more soft-focus |
| Hard blur | `BLUR_FATAL` | `5.0` | Catastrophic blur cutoff |
| Eyes EAR | `EAR_THRESHOLD` | `0.35` | Lower to be less strict about blinks |
| Aesthetic | `AESTHETIC_THRESHOLD` | `0.35` | Raise to keep only the most polished |
| VIP match | `VIP_COSINE_THRESH` | `0.75` | Lower to 0.65 if VIPs aren't being recognised |
| Duplicate | `DUPLICATE_THRESH` | `0.30` | Lower to reduce false-positive duplicates |
| Output dir | `OUTPUT_BASE` | `./output` | Change to write to a different disk |

---

## 🌐 REST API Endpoints

Start: `uvicorn api.fastapi_app:app --reload`  
Docs: `http://localhost:8000/docs`

| Method | Path | Description |
|---|---|---|
| `GET` | `/ping` | Health check |
| `POST` | `/filter` | Run full pipeline on a folder |
| `POST` | `/filter-by-person` | Pipeline + return per-VIP filenames |
| `POST` | `/upload-profiles` | Enrol VIP via ZIP upload |
| `GET` | `/download-log` | Download CSV audit log |
| `GET` | `/vips` | List enrolled VIPs |

See **[DOCUMENTATION.md](DOCUMENTATION.md)** §11 for full request/response schemas.

---

## 🔌 Frontend Integration

The API is CORS-enabled for `http://localhost:3000` out of the box. Connect any React / Next.js / Electron / mobile app via standard HTTP fetch calls.

See **[DOCUMENTATION.md](DOCUMENTATION.md)** §13 for ready-to-use code snippets for React, Electron, and mobile.

---

## 🧩 Extending the System

- **Add a new filter** — create a module in `filters/`, add a threshold to `config.py`, wire into `pipeline.py`. Full step-by-step in §14.1 of the docs.
- **Wire existing stubs** — `cropped_filter.py` and `obstruction_filter.py` are complete and just need importing. See §14.2.
- **Swap YOLO model** — point `YOLO_MODEL_PATH` to any YOLOv8 checkpoint.
- **Async jobs** — add `BackgroundTasks` to the API. Boilerplate in §14.5 of the docs.

---

## 📋 Prerequisites

- Python 3.10, 3.11, or 3.12
- NVIDIA GPU + CUDA (optional — auto-detected; CPU fallback works)
- `libjpeg-turbo` (optional — auto-fallback to PIL if absent)

---

## 📄 Documentation

**→ [DOCUMENTATION.md](DOCUMENTATION.md)** — complete reference covering every filter's algorithm, the full pipeline walkthrough, all API schemas, VIP enrolment, frontend integration, extension guide, limitations, and troubleshooting.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list. Key packages:

- `fastapi` + `uvicorn` — REST API
- `ultralytics` — YOLOv8
- `facenet-pytorch` — MTCNN + InceptionResnetV1
- `mediapipe` — FaceMesh
- `transformers` — CLIP ViT-L/14
- `scikit-learn` — duplicate clustering
- `joblib` — parallel processing
- `PyTurboJPEG` — fast JPEG decode (optional)
