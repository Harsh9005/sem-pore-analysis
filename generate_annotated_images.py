#!/usr/bin/env python3
"""
=============================================================================
SEM Pore Annotator — Generate annotated images and binary masks
=============================================================================
Author:  Harshvardhan Modh
License: MIT

Reads SEM micrographs, detects pore contours using adaptive Gaussian
thresholding, and produces:
  1. Annotated images with green pore contours and red diameter labels
  2. Binary masks (white = pore, black = background)

Designed as a companion to sem_pore_analysis.py for visualisation of
detected pore regions.
"""

import argparse
import os
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_BLOCK_SIZE = 899
DEFAULT_C_VALUE = 10
DEFAULT_IMAGE_WIDTH_PX = 1536
DEFAULT_HFW_UM = 207.0
DEFAULT_MIN_AREA_UM2 = 0.5
DEFAULT_MAX_DIAMETER_UM = 50.0
DEFAULT_BLUR_KERNEL = 5
DEFAULT_CROP_HEIGHT = 1000
DEFAULT_MORPH_KERNEL = 3
DEFAULT_MAX_LABELS = 40

# Visual settings
CONTOUR_COLOR = (0, 255, 0)   # Green (BGR)
LABEL_COLOR = (0, 0, 255)     # Red (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
THICKNESS = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="generate_annotated_images",
        description=textwrap.dedent("""\
            Generate annotated SEM images with pore contour overlays and
            binary masks. Uses the same adaptive thresholding approach as
            sem_pore_analysis.py.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True,
                   help="Directory containing .tif SEM images (searched recursively)")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: <input>/annotated_output)")
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                   help=f"Adaptive threshold block size (default: {DEFAULT_BLOCK_SIZE})")
    p.add_argument("--c-value", type=int, default=DEFAULT_C_VALUE,
                   help=f"Adaptive threshold constant C (default: {DEFAULT_C_VALUE})")
    p.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH_PX,
                   help=f"Image width in pixels (default: {DEFAULT_IMAGE_WIDTH_PX})")
    p.add_argument("--hfw", type=float, default=DEFAULT_HFW_UM,
                   help=f"Horizontal field width in µm (default: {DEFAULT_HFW_UM})")
    p.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA_UM2,
                   help=f"Minimum pore area in µm² (default: {DEFAULT_MIN_AREA_UM2})")
    p.add_argument("--max-diameter", type=float, default=DEFAULT_MAX_DIAMETER_UM,
                   help=f"Maximum pore diameter in µm (default: {DEFAULT_MAX_DIAMETER_UM})")
    p.add_argument("--blur-kernel", type=int, default=DEFAULT_BLUR_KERNEL,
                   help=f"Gaussian blur kernel size (default: {DEFAULT_BLUR_KERNEL})")
    p.add_argument("--crop-height", type=int, default=DEFAULT_CROP_HEIGHT,
                   help=f"Crop height to exclude metadata bar (default: {DEFAULT_CROP_HEIGHT})")
    p.add_argument("--morph-kernel", type=int, default=DEFAULT_MORPH_KERNEL,
                   help=f"Morphological opening kernel size (default: {DEFAULT_MORPH_KERNEL})")
    p.add_argument("--max-labels", type=int, default=DEFAULT_MAX_LABELS,
                   help=f"Max diameter labels per image (default: {DEFAULT_MAX_LABELS})")
    p.add_argument("--magnification-filter", type=str, default=None,
                   help="Only process files containing this string (e.g. '1000x')")
    return p.parse_args(argv)


def annotate_image(image_path, args, px_per_um, min_area_px):
    """Process a single image: detect pores, draw annotations, create mask."""
    img_color = cv2.imread(str(image_path))
    if img_color is None:
        return False

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Crop for processing (detection only)
    if gray.shape[0] > args.crop_height:
        roi = gray[:args.crop_height, :]
    else:
        roi = gray

    # Pre-processing
    blurred = cv2.GaussianBlur(roi, (args.blur_kernel, args.blur_kernel), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        args.block_size, args.c_value,
    )

    # Morphological cleanup
    kernel = np.ones((args.morph_kernel, args.morph_kernel), np.uint8)
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contour detection and filtering
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    um_per_px = 1.0 / px_per_um
    valid_contours = []
    annotations = []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue
        eq_d_px = np.sqrt(4 * area_px / np.pi)
        eq_d_um = eq_d_px * um_per_px
        if eq_d_um > args.max_diameter:
            continue

        valid_contours.append(cnt)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            annotations.append((cx, cy, eq_d_um))

    # Draw annotated image (on full-size original including metadata bar)
    annotated = img_color.copy()
    cv2.drawContours(annotated, valid_contours, -1, CONTOUR_COLOR, THICKNESS)

    # Add diameter labels to a random subset to avoid clutter
    random.shuffle(annotations)
    for i, (x, y, d) in enumerate(annotations):
        if i >= args.max_labels:
            break
        cv2.putText(annotated, f"{d:.1f}", (x, y), FONT, FONT_SCALE,
                     LABEL_COLOR, THICKNESS)

    # Create full-size binary mask
    full_mask = np.zeros_like(gray)
    full_mask[:clean_mask.shape[0], :clean_mask.shape[1]] = clean_mask

    return annotated, full_mask, len(valid_contours)


def main():
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output) if args.output else input_dir / "annotated_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    px_per_um = args.image_width / args.hfw
    min_area_px = args.min_area * (px_per_um ** 2)

    print(f"SEM Pore Annotator")
    print(f"{'=' * 50}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 50}\n")

    count = 0
    for root, _dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".tif"):
                continue
            if args.magnification_filter and args.magnification_filter.lower() not in fname.lower():
                continue

            img_path = Path(root) / fname
            result = annotate_image(img_path, args, px_per_um, min_area_px)

            if result is False:
                print(f"  [SKIP] {fname}")
                continue

            annotated, mask, n_pores = result
            stem = img_path.stem

            ann_path = output_dir / f"Annotated_{stem}.png"
            mask_path = output_dir / f"Mask_{stem}.png"

            cv2.imwrite(str(ann_path), annotated)
            cv2.imwrite(str(mask_path), mask)

            print(f"  {fname}: {n_pores} pores detected")
            count += 1

    print(f"\nDone — {count} image(s) annotated.")


if __name__ == "__main__":
    main()
