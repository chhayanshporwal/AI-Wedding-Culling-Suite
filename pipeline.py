#pipeline.py
import os
import csv
import shutil
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, cast

import numpy as np
from PIL import Image
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib  # type: ignore

import config
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---------- NEW FAST LOADER ----------
from utils.fast_loader import fast_imread  # returns np.ndarray | None

# ---------- FILTER IMPORTS ----------
from filters.person_filter import detect_persons
from filters.face_id_filter import load_vip_embeddings, match_vips, detect_faces
from filters.blur_filter import blur_score
from filters.exposure_filter import exposure_score
from filters.eyes_closed_filter import is_eyes_closed
from filters.aesthetic_filter import aesthetic_score, get_clip_embedding
from filters.duplicate_filter import DuplicateFilter

# ---------- WORKER ----------
def init_worker():
    load_vip_embeddings()

# ---------- PER-IMAGE ----------
def process_image(path: str) -> Dict[str, Any]:
    img = fast_imread(path)
    fname = os.path.basename(path)

    # safe defaults
    base: Dict[str, Any] = {
        "path": path,
        "filename": fname,
        "width": 0,
        "height": 0,
        "blur_score": 0.0,
        "exposure_score": 0.0,
        "num_persons": 0,
        "eyes_closed": False,
        "aesthetic_score": 0.0,
        "person_boxes": [],
        "face_boxes": [],
        "vip_matches": [],
        "clip_embedding": None,
    }

    if img is None:
        logger.warning(f"Skipping invalid image: {path}")
        return base

    h, w = img.shape[:2]
    persons = detect_persons(img) or []
    faces = detect_faces(img) or []
    # Optimization: Pass detected faces to avoid re-detection
    vips = match_vips(img, config.VIP_COSINE_THRESH, known_face_boxes=faces) or []
    eyes = is_eyes_closed(img, config.EAR_THRESHOLD)  # Now thread-safe

    # Use global embedding extractor
    emb = get_clip_embedding(img)

    return {
        **base,
        "width": w,
        "height": h,
        "blur_score": blur_score(img),
        "exposure_score": exposure_score(img),
        "num_persons": len(persons),
        "eyes_closed": eyes,
        "aesthetic_score": aesthetic_score(img),
        "person_boxes": persons,
        "face_boxes": faces,
        "vip_matches": vips,
        "clip_embedding": emb,
    }



# ---------- MAIN ----------
def run_filtering(
    input_folder: str,
    output_base: str = config.OUTPUT_BASE,
    max_workers: Optional[int] = None,
) -> Optional[str]:
    load_vip_embeddings()
    exts = tuple(config.IMAGE_EXTENSIONS or [])
    files = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder))
             if f.lower().endswith(exts)]
    if not files:
        logger.error("No images found in '%s'", input_folder)
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_base, stamp)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Processing %d images → %s", len(files), out_dir)

    t0 = time.time()

    # parallel processing
    with parallel_backend("threading"):
        with tqdm_joblib(tqdm(desc="Processing", total=len(files), unit="img")):
            results = Parallel(
                n_jobs=max_workers or config.CPU_CORES,
                prefer="threads"
            )(delayed(process_image)(p) for p in files)

    # exposure percentiles
    exp_vals = np.array([r["exposure_score"] for r in results if r is not None], dtype=float)
    low_cut, high_cut = np.percentile(exp_vals, [config.EXP_PCTL_LOW, config.EXP_PCTL_HIGH])

    # duplicate clustering
    deduper = DuplicateFilter(threshold=config.DUPLICATE_THRESH)
    for r in results:
        if r is None:
            continue
        emb = r.get("clip_embedding")
        if emb is not None:
            deduper.add(r["path"], emb)

    for grp in deduper.cluster():
        if len(grp) < 2:
            continue
        best = max(grp, key=lambda p: next((x["aesthetic_score"] for x in results if x is not None and x["path"] == p), 0.0))
        for p in grp:
            if p != best:
                next(x for x in results if x is not None and x["path"] == p)["is_duplicate"] = True

    # CSV & keepers
    csv_path = os.path.join(out_dir, "log.csv")
    headers = ["filename", "blur_score", "exposure_score", "num_persons",
               "eyes_closed", "aesthetic_score", "is_duplicate",
               "vip_matches", "reject_reasons"]

    vip_root = os.path.join(out_dir, "by_vip")
    os.makedirs(vip_root, exist_ok=True)

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=headers)
        writer.writeheader()
        for r in results:
            if r is None:
                continue
            reasons: list[str] = []
            blur = r["blur_score"]
            exp = r["exposure_score"]
            is_vip = bool(r["vip_matches"])

            # 🔴 VIPs are NOT exempt
            if blur < config.BLUR_FATAL:
                reasons.append("fully_blurred")
            if r["num_persons"] > 0 and not r["face_boxes"]:
                reasons.append("decor_focus")
            if blur < config.BLUR_THRESHOLD:
                reasons.append("blurry")
            if exp < low_cut:
                reasons.append("underexposed")
            elif exp > high_cut:
                reasons.append("overexposed")
            if r["num_persons"] == 0:
                reasons.append("no_person")
            if r["eyes_closed"]:
                reasons.append("eyes_closed")
            if r["aesthetic_score"] < config.AESTHETIC_THRESHOLD:
                reasons.append("low_aesthetic")
            if r.get("is_duplicate"):
                reasons.append("duplicate")

            r["reject_reasons"] = ";".join(reasons)
            writer.writerow({
                **{k: r.get(k, "") for k in headers},
                "vip_matches": "|".join(f"{n}:{c:.2f}" for n, c in r["vip_matches"]),
            })

            if not reasons:
                src, dst = r["path"], os.path.join(out_dir, r["filename"])
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                # XMP generation removed as per user request

                for vip, _ in r["vip_matches"]:
                    vip_dir = os.path.join(vip_root, vip)
                    os.makedirs(vip_dir, exist_ok=True)
                    try:
                        os.link(src, os.path.join(vip_dir, r["filename"]))
                    except FileExistsError:
                        pass

    logger.info("Completed in %.1f s → %s", time.time() - t0, out_dir)
    return out_dir

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wedding Image Culling Pipeline")
    parser.add_argument("input_folder")
    parser.add_argument("--output_base", default=config.OUTPUT_BASE)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_filtering(input_folder=args.input_folder,
                  output_base=args.output_base,
                  max_workers=args.workers)
