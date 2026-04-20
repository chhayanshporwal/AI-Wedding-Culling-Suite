# Wedding Image Culling Suite — Complete Documentation

> **Language**: Python 3.10+  
> **API Framework**: FastAPI  
> **Primary use case**: Automated, AI-powered photo culling for wedding photography studios

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [What It Can Do](#2-what-it-can-do)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [How Every Filter Works](#5-how-every-filter-works)
6. [The Pipeline — Step by Step](#6-the-pipeline-step-by-step)
7. [Output Structure](#7-output-structure)
8. [Configuration Reference](#8-configuration-reference)
9. [Installation & Setup](#9-installation--setup)
10. [CLI Usage](#10-cli-usage)
11. [REST API Reference](#11-rest-api-reference)
12. [Enrolling VIPs (Bride / Groom)](#12-enrolling-vips)
13. [Frontend Integration Guide](#13-frontend-integration-guide)
14. [Expanding the System](#14-expanding-the-system)
15. [Known Limitations & Gotchas](#15-known-limitations--gotchas)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What This Project Is

The **Wedding Image Culling Suite** is a fully automated backend system that ingests a folder of raw wedding photographs and intelligently selects only the best ones — rejecting blurry, over/underexposed, aesthetically poor, duplicate, or "bad-moment" shots — while guaranteeing that photos of important people (Bride, Groom, or any other enrolled "VIP") are preserved.

It is designed to replace hours of manual curation by a photographer's assistant. The selected photos land in a clean output folder, organised by VIP, and every image (kept **and** rejected) is logged in a detailed CSV audit file.

The system runs as either:
- A **REST API** (FastAPI) — for any frontend (React, Next.js, Electron, mobile)
- A **CLI tool** — for direct terminal usage

---

## 2. What It Can Do

### 2.1 Core Filtering Capabilities

| Capability | What It Does | Algorithm |
|---|---|---|
| **Blur Detection** | Rejects images that are too blurry | Laplacian variance, normalised by image area |
| **Fatal Blur Gate** | Hard-rejects images with catastrophic blur | Laplacian variance < 5.0 |
| **Exposure Check** | Filters images that are too dark or too bright | Mean grayscale intensity; dynamic percentile thresholds |
| **Person Detection** | Rejects photos with no people in frame | YOLOv8n (COCO class 0 = "person") |
| **Decor-Focus Rejection** | Rejects back-of-head / décor shots | Person detected + no face box found |
| **Eyes Closed Detection** | Rejects photos where any face has closed eyes | MediaPipe FaceMesh + Eye Aspect Ratio (EAR) |
| **Aesthetic Scoring** | Rejects photos with poor composition or quality | OpenAI CLIP ViT-L/14 → AVA-trained regression head |
| **Duplicate Detection** | Finds burst near-duplicates; keeps the best | CLIP embeddings → L2 radius clustering |
| **VIP Face Recognition** | Recognises enrolled persons by face | FaceNet InceptionResnetV1 (VGGFace2) + cosine similarity |

### 2.2 Output Capabilities

- **Keeps** only images that pass all active filters
- **Organises** kept images in a timestamped output folder
- **Groups** kept images by detected VIP into `by_vip/<Name>/` subfolders (hard-linked)
- **Produces** a full `log.csv` with per-image scores and rejection reasons for every image

### 2.3 REST API Capabilities

| Endpoint | Purpose |
|---|---|
| `GET /ping` | Health check |
| `POST /filter` | Run full pipeline on a server-accessible folder |
| `POST /filter-by-person` | Pipeline + return only filenames matching named VIPs |
| `POST /upload-profiles` | Live VIP enrollment via ZIP upload |
| `GET /download-log` | Download CSV audit log for a completed run |
| `GET /vips` | List all currently enrolled VIP names |

### 2.4 Performance Features

- **Parallel processing** — `joblib` threading backend, uses all CPU cores by default
- **Thread-safe model loading** — YOLO, FaceNet/MTCNN, MediaPipe, CLIP each use thread-local instances
- **Fast image loading** — TurboJPEG for JPEG decode, with automatic PIL/OpenCV fallback
- **Cheap pre-filter** — loader-level Laplacian gate discards hopeless images before any model runs
- **Embedding reuse** — face detection runs once per image; result shared with VIP matching
- **GPU support** — auto-detected via `torch.cuda.is_available()`

---

## 3. System Architecture

```
                    ┌─────────────────────────────────┐
                    │         FastAPI Server           │
                    │       api/fastapi_app.py         │
                    └──────────────┬──────────────────┘
                                   │ calls
                                   ▼
                    ┌─────────────────────────────────┐
                    │        pipeline.py               │
                    │      run_filtering()             │
                    │                                  │
                    │  1. Collect image paths          │
                    │  2. Parallel process_image()     │
                    │  3. Exposure percentile calc     │
                    │  4. Duplicate clustering         │
                    │  5. CSV write + file output      │
                    └───┬───────────┬─────────────────┘
                        │           │
           ┌────────────┘   ┌───────┘
           ▼                ▼
 ┌──────────────┐   ┌─────────────────────────────────┐
 │ fast_loader  │   │           filters/               │
 │              │   │                                  │
 │ TurboJPEG   │   │  blur_filter   (Laplacian)       │
 │ + PIL EXIF  │   │  exposure_f.   (mean gray)       │
 │ + downscale │   │  person_f.     (YOLOv8n)         │
 │ + blur gate │   │  face_id_f.    (FaceNet/MTCNN)   │
 └──────────────┘   │  eyes_f.       (MediaPipe EAR)   │
                    │  aesthetic_f.  (CLIP + MLP)      │
                    │  duplicate_f.  (CLIP + kNN)      │
                    └─────────────────────────────────┘
```

---

## 4. Project Structure

```
photo-filter/
├── api/
│   └── fastapi_app.py          # All REST API endpoints
│
├── filters/
│   ├── __init__.py
│   ├── aesthetic_filter.py     # Thin wrapper: CLIP embedding + scorer
│   ├── aesthetic_score.py      # AestheticScorer class (CLIP + regression head)
│   ├── ava_logos_linearMSE.pth # Pre-trained aesthetic regression weights
│   ├── blur_filter.py          # Laplacian variance sharpness score
│   ├── cropped_filter.py       # Face-crop detection (available, not yet wired)
│   ├── duplicate_filter.py     # DuplicateFilter: CLIP + sklearn radius neighbours
│   ├── exposure_filter.py      # Mean grayscale intensity
│   ├── eyes_closed_filter.py   # MediaPipe FaceMesh + EAR
│   ├── face_id_filter.py       # FaceNet face detection + VIP cosine matching
│   ├── obstruction_filter.py   # IoU-based obstruction detection (available, not yet wired)
│   └── person_filter.py        # YOLOv8n person detection
│
├── utils/
│   └── fast_loader.py          # High-performance image loading
│
├── scripts/
│   ├── enroll_vips.py          # CLI: register a VIP's face embeddings
│   └── verify_setup.py         # End-to-end smoke test with synthetic images
│
├── data/
│   └── embeddings/             # Persisted VIP .npy embeddings + index.json
│
├── output/                     # Runtime output (gitignored — generated per run)
│
├── pipeline.py                 # Main orchestration engine
├── config.py                   # All constants; auto-detects GPU / CPU / OS
├── logging_config.py           # Structured logging setup
├── requirements.txt            # All Python dependencies
├── yolov8n.pt                  # YOLOv8 nano weights (bundled)
├── Wedding_Culling_Colab.ipynb # Google Colab version
├── DOCUMENTATION.md            # ← You are here
└── README.md
```

### Filters not yet wired into the active pipeline

| File | What it does | Status |
|---|---|---|
| `filters/cropped_filter.py` | Detects faces clipped by image edge | Ready — see §14.2 to wire in |
| `filters/obstruction_filter.py` | IoU-based: person found but face obstructed/covered | Ready — see §14.2 to wire in |

---

## 5. How Every Filter Works

### 5.1 Fast Image Loader (`utils/fast_loader.py`)

**`fast_imread(path) → np.ndarray | None`**

Called first for every image before any model runs:
1. Decodes JPEG via **TurboJPEG** (fastest) — falls back to **PIL** with EXIF rotation correction
2. **Downscales** so the longer edge ≤ 640 px — reduces all downstream model load
3. **Cheap blur gate**: Laplacian variance < 20 → returns `None` immediately, image is skipped entirely

---

### 5.2 Blur Filter (`filters/blur_filter.py`)

**`blur_score(image) → float`**

```
score = var(Laplacian(gray)) / (H × W) × 1e6
```

Higher = sharper. Two rejection levels:

| Threshold | Config key | Default | Reject reason |
|---|---|---|---|
| Hard | `BLUR_FATAL` | 5.0 | `"fully_blurred"` |
| Soft | `BLUR_THRESHOLD` | 30.0 | `"blurry"` |

---

### 5.3 Exposure Filter (`filters/exposure_filter.py`)

**`exposure_score(image) → float`** — `mean(grayscale)`, range 0–255

Thresholds computed **dynamically per batch**:
- Below 5th percentile → `"underexposed"`
- Above 95th percentile → `"overexposed"`

> Dynamic percentiles adapt to venue lighting (dim reception vs. bright outdoor ceremony) instead of failing on entire events.

---

### 5.4 Person Filter (`filters/person_filter.py`)

**`detect_persons(image, conf_thresh=0.3) → list[tuple]`**

- YOLOv8n pretrained on COCO, class 0 = "person"
- Returns `[(x1, y1, x2, y2), ...]`
- Thread-safe — each thread owns its YOLO instance
- Rejection: `"no_person"` if result is empty

---

### 5.5 Decor-Focus Logic (in `pipeline.py`)

```python
if r["num_persons"] > 0 and not r["face_boxes"]:
    reasons.append("decor_focus")
```

Person detected by YOLO but no face found by MTCNN → back-of-head, too-distant, or décor shot. Rejection: `"decor_focus"`.

---

### 5.6 Face Detection & VIP Recognition (`filters/face_id_filter.py`)

**`detect_faces(image) → list[tuple]`**  
MTCNN from `facenet_pytorch` — returns pixel-space bounding boxes.

**`match_vips(image, thresh, known_face_boxes) → list[(name, score)]`**  
Accepts pre-computed face boxes — avoids running MTCNN twice per image.
- Crops each face, resizes to 160×160, runs through **InceptionResnetV1** (VGGFace2)
- Cosine similarity vs. each enrolled VIP's averaged + normalised embedding
- Returns all matches ≥ `VIP_COSINE_THRESH`

**Embedding index** (`data/embeddings/index.json`):
```json
{
  "Bride": ["Bride/Bride_000.npy", "Bride/Bride_001.npy"],
  "Groom": ["Groom/Groom_000.npy"]
}
```

---

### 5.7 Eyes Closed Filter (`filters/eyes_closed_filter.py`)

**`is_eyes_closed(image, ear_threshold=0.35) → bool`**

1. MediaPipe FaceMesh → 468 landmarks per face
2. Eye Aspect Ratio (EAR):
   ```
   EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
   ```
3. If mean EAR of both eyes < `EAR_THRESHOLD` for **any** face → `True`
4. Thread-safe; gracefully disabled if MediaPipe is not installed

Rejection: `"eyes_closed"`

---

### 5.8 Aesthetic Scorer (`filters/aesthetic_score.py` + `filters/aesthetic_filter.py`)

**`aesthetic_score(image) → float` (0.0–1.0)**

1. BGR → RGB → PIL Image
2. **CLIP ViT-L/14** image encoder → 768-dim embedding
3. **`AestheticHead`** regression (768 → 1024 → 128 → 64 → 16 → 1)
4. Weights (`ava_logos_linearMSE.pth`) trained on the AVA dataset
5. Output clamped to [0, 1]

Also exposes **`get_clip_embedding(image) → np.ndarray`** used for duplicate detection.

Rejection: `"low_aesthetic"` if score < `AESTHETIC_THRESHOLD`

---

### 5.9 Duplicate Filter (`filters/duplicate_filter.py`)

**`DuplicateFilter(threshold=0.3)`**

1. Collects CLIP embeddings for all images in the batch
2. sklearn `NearestNeighbors` with `radius=DUPLICATE_THRESH` (L2) → adjacency graph
3. DFS extracts connected components (duplicate clusters)
4. For each cluster: keeps the image with the highest `aesthetic_score`, marks rest as `"duplicate"`

---

### 5.10 Available (Not Yet Wired) Filters

#### `filters/cropped_filter.py`
**`is_cropped(face_boxes, img_w, img_h, edge_margin=0.05) → bool`**  
Returns `True` if any detected face comes within 5% of any image edge — catches unflattering crops.

#### `filters/obstruction_filter.py`
**`is_obstructed(person_boxes, face_boxes, min_iou=0.5) → bool`**  
Returns `True` if any person bounding box has no corresponding face that lies inside it with IoU ≥ `min_iou` — detects people whose faces are blocked by objects or other people.

See §14.2 for wiring instructions.

---

## 6. The Pipeline — Step by Step

**`run_filtering(input_folder, output_base, max_workers)`** in `pipeline.py`:

```
Step 1  Collect file paths
        └─ os.listdir(input_folder), filter by IMAGE_EXTENSIONS, sort

Step 2  Create output directory
        └─ {output_base}/{YYYYMMDD_HHMMSS}/

Step 3  Parallel per-image processing  (joblib threading)
        └─ For each image:
             a. fast_imread()           → load + downscale + cheap blur gate
             b. detect_persons()        → YOLOv8n bounding boxes
             c. detect_faces()          → MTCNN bounding boxes
             d. match_vips()            → FaceNet + cosine  (reuses face boxes)
             e. is_eyes_closed()        → MediaPipe EAR
             f. get_clip_embedding()    → CLIP ViT-L/14 vector
             g. blur_score()            → Laplacian variance
             h. exposure_score()        → mean grayscale
             i. aesthetic_score()       → CLIP + regression head
             → Returns dict with all scores + boxes

Step 4  Exposure percentile thresholds
        └─ np.percentile([all exposure_scores], [EXP_PCTL_LOW, EXP_PCTL_HIGH])

Step 5  Duplicate clustering
        └─ Feed all CLIP embeddings into DuplicateFilter
        └─ Mark lower-aesthetic duplicates

Step 6  Decision + output
        └─ For each result:
             - Evaluate all rejection conditions → build reasons list
             - Write every image to log.csv  (kept AND rejected)
             - If reasons is empty → hard-link to {output}/{stamp}/{filename}
             - If VIP matched → also hard-link to {output}/{stamp}/by_vip/{Name}/
```

---

## 7. Output Structure

```
output/
└── 20240920_143022/          ← timestamped run folder
    ├── log.csv               ← full per-image audit log
    ├── photo_001.jpg         ← kept photos (flat, hard-linked)
    ├── photo_015.jpg
    └── by_vip/
        ├── Bride/
        │   ├── photo_001.jpg
        │   └── photo_033.jpg
        └── Groom/
            └── photo_015.jpg
```

### log.csv columns

| Column | Description |
|---|---|
| `filename` | Basename of the image file |
| `blur_score` | Normalised Laplacian variance (higher = sharper) |
| `exposure_score` | Mean grayscale 0–255 |
| `num_persons` | YOLO person count |
| `eyes_closed` | `True` / `False` |
| `aesthetic_score` | Float 0–1 |
| `is_duplicate` | `True` / `False` |
| `vip_matches` | Pipe-separated `Name:score`, e.g. `Bride:0.91\|Groom:0.83` |
| `reject_reasons` | Semicolon-separated list. **Empty string = image was kept** |

### Possible `reject_reasons` values

| Reason | Trigger |
|---|---|
| `fully_blurred` | Laplacian < BLUR_FATAL |
| `blurry` | Laplacian < BLUR_THRESHOLD |
| `underexposed` | Exposure < EXP_PCTL_LOW percentile |
| `overexposed` | Exposure > EXP_PCTL_HIGH percentile |
| `no_person` | YOLO found 0 persons |
| `decor_focus` | Person found but no face detected |
| `eyes_closed` | Any face has EAR < EAR_THRESHOLD |
| `low_aesthetic` | Aesthetic score < AESTHETIC_THRESHOLD |
| `duplicate` | Near-duplicate cluster member (not the best) |

> **VIP images are NOT exempt from any filter.** A blurry bridezilla shot with closed eyes will be rejected.

---

## 8. Configuration Reference

All settings live in `config.py` and are overridable via environment variables.

| Constant | Env Var | Default | Description |
|---|---|---|---|
| `OUTPUT_BASE` | `OUTPUT_BASE` | `./output` | Root for run outputs |
| `BLUR_THRESHOLD` | `BLUR_THRESHOLD` | `30.0` | Soft blur rejection |
| `BLUR_FATAL` | `BLUR_FATAL` | `5.0` | Hard blur rejection |
| `EAR_THRESHOLD` | `EAR_THRESHOLD` | `0.35` | Eye closure cutoff |
| `AESTHETIC_THRESHOLD` | `AESTHETIC_THRESHOLD` | `0.35` | Min aesthetic score |
| `VIP_COSINE_THRESH` | `VIP_COSINE_THRESH` | `0.75` | VIP face match strictness |
| `DUPLICATE_THRESH` | `DUPLICATE_THRESH` | `0.30` | Max L2 for duplicate |
| `EXP_PCTL_LOW` | `EXP_PCTL_LOW` | `5.0` | Under-exposure percentile |
| `EXP_PCTL_HIGH` | `EXP_PCTL_HIGH` | `95.0` | Over-exposure percentile |
| `CLIP_BATCH_SIZE` | `CLIP_BATCH_SIZE` | `32` | CLIP inference batch |
| `YOLO_MODEL_PATH` | `YOLO_MODEL_PATH` | `./yolov8n.pt` | YOLO weights path |
| `AESTHETIC_WEIGHTS` | `AESTHETIC_WEIGHTS` | `./filters/ava_logos_linearMSE.pth` | Aesthetic head weights |
| `CPU_CORES` | — | `os.cpu_count()` | Auto-detected |
| `DEVICE` | — | `"cuda"` or `"cpu"` | Auto-detected by PyTorch |

### Example: override via env vars

```bash
# Stricter sharpness, more permissive aesthetics
BLUR_THRESHOLD=50 AESTHETIC_THRESHOLD=0.25 python pipeline.py /photos

# Custom output root for the API server
OUTPUT_BASE=/Volumes/NAS/Culled uvicorn api.fastapi_app:app --reload
```

---

## 9. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU + CUDA (optional but strongly recommended)
- `libjpeg-turbo` for TurboJPEG (optional — falls back to PIL)

### Steps

```bash
# 1. Clone
git clone https://github.com/Chhayansh-Git/Phase2.git
cd Phase2

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the setup
python scripts/verify_setup.py
# Expect: ✅ Success! Output generated at: ...
```

### macOS — TurboJPEG (optional speed boost)

```bash
brew install libjpeg-turbo
```

---

## 10. CLI Usage

### Run the full culling pipeline

```bash
python pipeline.py /path/to/wedding/photos
```

```bash
python pipeline.py /path/to/photos \
  --output_base /path/to/output \
  --workers 8
```

| Argument | Default | Description |
|---|---|---|
| `input_folder` | required | Path to folder containing images |
| `--output_base` | `config.OUTPUT_BASE` | Where to write output |
| `--workers` | all CPU cores | Number of parallel threads |

### Enrol a VIP

```bash
python scripts/enroll_vips.py "Bride" /path/to/bride/portraits
python scripts/enroll_vips.py "Groom" /path/to/groom/portraits
```

### Run end-to-end smoke test

```bash
python scripts/verify_setup.py
```

---

## 11. REST API Reference

### Start the server

```bash
uvicorn api.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

---

### `GET /ping` — Health check

```json
{ "status": "ok" }
```

---

### `POST /upload-profiles` — Enrol a VIP

**Request** (multipart/form-data):

| Field | Type | Description |
|---|---|---|
| `person` | string | VIP name, e.g. `"Bride"` |
| `zipfile_upload` | file | ZIP containing face photos |

**Response:**
```json
{ "enrolled": "Bride" }
```

---

### `POST /filter` — Run full pipeline

**Request (JSON):**
```json
{
  "input_folder": "/absolute/path/to/photos",
  "workers": 4
}
```

**Response:**
```json
{ "output_folder": "/absolute/path/to/output/20240920_143022" }
```

**Errors:** `400` bad path · `500` pipeline error

---

### `POST /filter-by-person` — Filter for specific VIPs

**Request (JSON):**
```json
{
  "input_folder": "/absolute/path/to/photos",
  "persons": ["Bride", "Groom"],
  "workers": 4
}
```

**Response:**
```json
{
  "output_folder": "/absolute/path/to/output/20240920_143022",
  "filtered_images": ["photo_001.jpg", "photo_033.jpg"]
}
```

Only filenames that **passed all filters** and matched one of the requested VIPs are returned.

---

### `GET /download-log?folder=<path>` — Download CSV audit log

Returns `text/csv` (`results.csv`) for the given `output_folder`.

**Error:** `404` if log file is missing.

---

### `GET /vips` — List enrolled VIPs

```json
{ "vips": ["Bride", "Groom", "Maid of Honor"] }
```

---

## 12. Enrolling VIPs

### What you need
- 3–20 clear, front-facing portrait photos per person (JPEG or PNG)
- Varied lighting / angles improve recognition robustness

### CLI

```bash
python scripts/enroll_vips.py "Bride" /path/to/bride/portraits
```

### API (for frontends)

```http
POST /upload-profiles
Content-Type: multipart/form-data

person=Bride
zipfile_upload=<bride_portraits.zip>
```

### What happens internally

```
1. MTCNN detects face in each portrait
2. InceptionResnetV1 computes 512-dim FaceNet embedding
3. Embeddings saved as .npy → data/embeddings/Bride/Bride_000.npy …
4. index.json updated atomically (temp file + os.replace — no partial writes)
5. In-memory VIP_EMBS cache refreshed immediately
```

### Re-enrolment

Running `enroll` again for the same name **replaces** all existing embeddings for that person.

---

## 13. Frontend Integration Guide

### CORS configuration

The API is pre-configured for `http://localhost:3000`. For production, update in `api/fastapi_app.py`:

```python
allow_origins=["https://your-production-domain.com"]
```

### React / Next.js code snippets

```typescript
const BASE = "http://localhost:8000";

// Health check
await fetch(`${BASE}/ping`);

// Enrol VIP
const form = new FormData();
form.append("person", "Bride");
form.append("zipfile_upload", zipFile);        // File from <input type="file">
await fetch(`${BASE}/upload-profiles`, { method: "POST", body: form });

// Run filter
const { output_folder } = await (await fetch(`${BASE}/filter`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ input_folder: "/path/to/photos", workers: 4 }),
})).json();

// Download CSV log
const csv = await (await fetch(
  `${BASE}/download-log?folder=${encodeURIComponent(output_folder)}`
)).text();

// Filter by person
const { filtered_images } = await (await fetch(`${BASE}/filter-by-person`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ input_folder: "/path/to/photos", persons: ["Bride"] }),
})).json();
```

### Electron (Desktop App)

```javascript
// main.js — spawn the FastAPI server as a child process
const { spawn } = require("child_process");
const server = spawn("uvicorn", ["api.fastapi_app:app", "--port", "8000"], {
  cwd: "/path/to/project",
  env: { ...process.env },
});

// Renderer process — use the same fetch() calls above
```

### Mobile (React Native / Flutter)

The backend must be reachable on the network. Use the host machine's local IP or a tunnel (ngrok). Pass file paths that are accessible to the **Python server**, not the mobile device (e.g., a shared NAS share).

### Serving output images to the frontend

The API currently does not serve static files. Add this to `fastapi_app.py` to expose the output folder:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/output", StaticFiles(directory=config.OUTPUT_BASE), name="output")
# Access: http://localhost:8000/output/20240920_143022/photo_001.jpg
```

---

## 14. Expanding the System

### 14.1 Adding a Brand New Filter

**Step 1** — Create `filters/my_filter.py`:
```python
import numpy as np

def my_score(image: np.ndarray) -> float:
    """Higher = better. Return 0.0 if image is None/empty."""
    if image is None or image.size == 0:
        return 0.0
    # ... logic ...
    return score
```

**Step 2** — Add threshold to `config.py`:
```python
MY_THRESHOLD = float(os.getenv("MY_THRESHOLD", 0.5))
```

**Step 3** — Import and call in `pipeline.py` → `process_image()`:
```python
from filters.my_filter import my_score
# add to return dict:
"my_score": my_score(img),
```

**Step 4** — Add rejection logic in `run_filtering()`:
```python
if r["my_score"] < config.MY_THRESHOLD:
    reasons.append("my_filter_failed")
```

**Step 5** — Add `"my_score"` to the `headers` list in `run_filtering()`.

---

### 14.2 Wiring Up the Existing Stubs

#### `filters/cropped_filter.py`

```python
# In pipeline.py — in process_image():
from filters.cropped_filter import is_cropped

"is_cropped": is_cropped(faces, w, h),  # add to return dict
```
```python
# In run_filtering() rejection block:
if r["is_cropped"]:
    reasons.append("face_cropped")
```

#### `filters/obstruction_filter.py`

```python
# In pipeline.py — in process_image():
from filters.obstruction_filter import is_obstructed

"is_obstructed": is_obstructed(persons, faces),  # add to return dict
```
```python
# In run_filtering() rejection block:
if r["is_obstructed"]:
    reasons.append("face_obstructed")
```

---

### 14.3 Swapping the YOLO Model

```bash
# Larger, more accurate model
YOLO_MODEL_PATH=/path/to/yolov8s.pt python pipeline.py /photos

# Domain fine-tuned model
YOLO_MODEL_PATH=/path/to/wedding_person_detector.pt python pipeline.py /photos
```

### 14.4 Swapping the Aesthetic Model

1. Ensure the new regression head takes 768-dim input (CLIP ViT-L/14)
2. Update `AESTHETIC_WEIGHTS` env var or `config.py`
3. If architecture differs, update `AestheticHead` in `filters/aesthetic_score.py`

### 14.5 Adding Async / Long-Running Jobs

```python
# In api/fastapi_app.py
import uuid
from fastapi import BackgroundTasks

jobs: dict = {}   # in production: use Redis / a DB

@app.post("/filter-async")
def filter_async(req: FilterRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running"}
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def job_status(job_id: str):
    return jobs.get(job_id, {"error": "not found"})

def _run_job(job_id: str, req: FilterRequest):
    out = run_filtering(req.input_folder, config.OUTPUT_BASE, req.workers)
    jobs[job_id] = {"status": "done", "output_folder": out}
```

---

## 15. Known Limitations & Gotchas

| Area | Detail |
|---|---|
| **Input folder access** | Path passed to `/filter` must be accessible to the Python process |
| **Synchronous API** | `/filter` blocks until the run completes — can be minutes for large batches |
| **EAR checks ANY face** | One blinker in a group of 10 rejects the whole image |
| **CLIP loaded at import** | `aesthetic_score.py` loads CLIP on module import → ~3–5s cold start |
| **Loader vs. config size** | `fast_loader.py` hard-codes `MAX_EDGE = 640`; `config.IMAGE_MAX_SIZE` is not used there |
| **VIP re-enrolment replaces** | Running enrol again for the same name overwrites — does not append |
| **Output folder grows forever** | No automatic cleanup of old timestamped runs |
| **No RAW format support** | `.ARW`, `.CR2`, `.NEF`, `.DNG` are not in `IMAGE_EXTENSIONS` |

---

## 16. Troubleshooting

### "No images found in folder"
- Check `IMAGE_EXTENSIONS` in `config.py` — RAW formats not supported
- Ensure the path is absolute and accessible to the Python process

### Models are very slow
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, all inference runs on CPU. First run is always slower (model loading).

### "MediaPipe FaceMesh not available"
```bash
pip install mediapipe>=0.10.9
# Apple Silicon:
pip install mediapipe-silicon
```

### VIPs not recognised
- Lower `VIP_COSINE_THRESH` to 0.65–0.70
- Ensure enrolment portraits clearly show the face (face height ≥ 160 px)
- Confirm enrolment ran: `GET /vips`

### High duplicate rejection rate
- Lower `DUPLICATE_THRESH` (e.g., 0.15)
- Check if all photos are taken in the same tight burst

### TurboJPEG warning (non-fatal)
```bash
# macOS
brew install libjpeg-turbo && pip install PyTurboJPEG
```

### API 500 on `/filter`
- Check the terminal running `uvicorn` for the full Python stack trace
- Verify `yolov8n.pt` is in project root and `ava_logos_linearMSE.pth` is in `filters/`
