# Wedding Image Culling Suite (Phase 2)

A powerful, AI-driven backend for automating the culling of wedding photos. This system filters images based on clarity (blur), exposure, eyes open/closed status, and aesthetic quality, while preserving "VIP" (Bride/Groom) photos and grouping duplicates to select the best shots.

## 🚀 Features

-   **Multi-Stage Filtering**:
    -   **Blur Detection**: Rejects blurry images using Laplacian variance.
    -   **Exposure Check**: Filters under/overexposed images based on dynamic percentiles.
    -   **Person Detection**: Ensures photos contain people (unless configured otherwise).
    -   **Face Quality**: Detects closed eyes using MediaPipe FaceMesh.
    -   **Aesthetic Scoring**: Uses OpenAI CLIP + MLP to rate image composition and quality.
-   **VIP Protection**: specific people (Bride/Groom) can be enrolled via facial recognition (FaceNet) to ensure their photos are preserved even if borderline.
-   **Duplicate Grouping**: Identifies similar images (using CLIP embeddings) and automatically picks the best one from the group.
-   **High Performance**:
    -   Parallel processing with `joblib`.
    -   Thread-safe model loading for YOLO, FaceNet, and CLIP.
    -   Fast image loading with TurboJPEG (with Windows fallback).

## 🛠️ Setup & Installation

### Prerequisites
-   Python 3.10+
-   CUDA-capable GPU recommended (but runs on CPU).
-   Windows, macOS, or Linux.

### Installation
1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: On Windows, `TurboJPEG` may require installing `libjpeg-turbo`. If not found, the system gracefully falls back to Pillow/OpenCV.*

## 🏃 Usage

### Running the API (Recommended)
The project exposes a `FastAPI` server for frontend integration.

```bash
uvicorn api.fastapi_app:app --reload
```
The API will be available at `http://localhost:8000`.

### CLI Usage
You can also run the pipeline directly on a folder:
```bash
python pipeline.py /path/to/input/images --output_base /path/to/output
```

## 🏗️ Project Structure

-   `api/`: FastAPI application endpoints.
-   `filters/`: Modular filter implementations.
    -   `aesthetic_filter.py`: CLIP-based scoring.
    -   `face_id_filter.py`: VIP recognition (FaceNet).
    -   `person_filter.py`: Person detection (YOLO).
    -   `eyes_closed_filter.py`: Blink detection (MediaPipe).
-   `utils/`: Helper utilities (Fast image loading).
-   `pipeline.py`: Main orchestrator that runs filters in parallel.
-   `config.py`: Configuration thresholds and paths.

## 🔧 Configuration

Adjust thresholds in `config.py` or via environment variables:
-   `BLUR_THRESHOLD`: Minimum sharpness (default: 30.0).
-   `AESTHETIC_THRESHOLD`: Minimum aesthetic score (default: 0.35).
-   `EAR_THRESHOLD`: Eye Aspect Ratio for blink detection (default: 0.35).
-   `VIP_COSINE_THRESH`: Face matching strictness (default: 0.75).

## 🖥️ Frontend Integration
See `frontend_integration_guide.md` for details on connecting a React/Electron frontend.
