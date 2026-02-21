#!/usr/bin/env python3
"""
=============================================================================
SEM Pore Analysis — Automated pore size quantification from SEM images
=============================================================================
Author:  Harshvardhan Modh
License: MIT

Segments dark pore regions in grayscale SEM micrographs using adaptive
Gaussian thresholding and quantifies pore equivalent diameters via contour
analysis. Designed for batch processing of porous material cross-sections
imaged at known magnification.

Algorithm
---------
1. Load grayscale image and crop metadata bar (bottom region)
2. Gaussian blur for noise reduction
3. Adaptive Gaussian thresholding (inverted: dark pores → white)
4. Morphological opening to remove salt noise
5. Contour detection with area and diameter filtering
6. Equivalent diameter: D = 2 * sqrt(Area / pi)

Outputs: CSV statistics, composite segmentation figure, pore diameter
bar chart, binary masks, and analysis parameter log.
"""

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DEFAULT_DPI = 300
DEFAULT_FORMAT = "png"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="sem_pore_analysis",
        description=textwrap.dedent("""\
            Quantify pore size from SEM micrographs using adaptive thresholding.
            Walks the input directory for .tif images, segments dark pore regions,
            and outputs per-image statistics, figures, and binary masks.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True,
                   help="Directory containing .tif SEM images (searched recursively)")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: <input>/analysis_output)")
    p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                   help=f"Adaptive threshold block size in pixels (default: {DEFAULT_BLOCK_SIZE})")
    p.add_argument("--c-value", type=int, default=DEFAULT_C_VALUE,
                   help=f"Adaptive threshold constant C (default: {DEFAULT_C_VALUE})")
    p.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH_PX,
                   help=f"Image width in pixels for calibration (default: {DEFAULT_IMAGE_WIDTH_PX})")
    p.add_argument("--hfw", type=float, default=DEFAULT_HFW_UM,
                   help=f"Horizontal field width in micrometres (default: {DEFAULT_HFW_UM})")
    p.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA_UM2,
                   help=f"Minimum pore area in um^2 (default: {DEFAULT_MIN_AREA_UM2})")
    p.add_argument("--max-diameter", type=float, default=DEFAULT_MAX_DIAMETER_UM,
                   help=f"Maximum pore equivalent diameter in um (default: {DEFAULT_MAX_DIAMETER_UM})")
    p.add_argument("--blur-kernel", type=int, default=DEFAULT_BLUR_KERNEL,
                   help=f"Gaussian blur kernel size (default: {DEFAULT_BLUR_KERNEL})")
    p.add_argument("--crop-height", type=int, default=DEFAULT_CROP_HEIGHT,
                   help=f"Crop image to this height to exclude metadata bar (default: {DEFAULT_CROP_HEIGHT})")
    p.add_argument("--morph-kernel", type=int, default=DEFAULT_MORPH_KERNEL,
                   help=f"Morphological opening kernel size (default: {DEFAULT_MORPH_KERNEL})")
    p.add_argument("--magnification-filter", type=str, default=None,
                   help="Only process files whose name contains this string (e.g. '1000x')")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                   help=f"Figure resolution (default: {DEFAULT_DPI})")
    p.add_argument("--format", choices=["png", "tiff", "pdf"], default=DEFAULT_FORMAT,
                   help=f"Output figure format (default: {DEFAULT_FORMAT})")
    return p.parse_args(argv)


# ── Core analysis ───────────────────────────────────────────────────────────

def analyze_image(image_path, block_size, c_value, blur_kernel, crop_height,
                  morph_kernel, px_per_um, min_area_px, max_diameter_um):
    """Analyse a single SEM image and return pore diameter list + mask."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None

    # Crop metadata bar
    if img.shape[0] > crop_height:
        roi = img[:crop_height, :]
    else:
        roi = img

    # Pre-processing
    blurred = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)

    # Adaptive threshold (dark pores → white after inversion)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c_value,
    )

    # Morphological opening to remove noise
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contour detection
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    um_per_px = 1.0 / px_per_um
    diameters = []
    valid_contours = []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue
        eq_d_px = np.sqrt(4 * area_px / np.pi)
        eq_d_um = eq_d_px * um_per_px
        if eq_d_um > max_diameter_um:
            continue
        diameters.append(eq_d_um)
        valid_contours.append(cnt)

    return diameters, clean_mask, valid_contours


def collect_images(input_dir, magnification_filter=None):
    """Recursively collect .tif images from input_dir."""
    images = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if not f.lower().endswith(".tif"):
                continue
            if magnification_filter and magnification_filter.lower() not in f.lower():
                continue
            images.append(Path(root) / f)
    return images


# ── Figures ─────────────────────────────────────────────────────────────────

def make_composite_figure(image_paths, masks_dict, output_path, dpi):
    """Create a composite figure: top row originals, bottom row mask overlays."""
    n = len(image_paths)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        name = img_path.stem

        # Top row: original
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(name, fontsize=10)
        axes[0, i].axis("off")

        # Bottom row: overlay
        mask = masks_dict.get(name)
        if mask is not None:
            overlay = cv2.cvtColor(img[:mask.shape[0], :], cv2.COLOR_GRAY2RGB)
            overlay[mask > 0] = [255, 0, 100]  # magenta for pores
            axes[1, i].imshow(overlay)
        else:
            axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"{name} — pores", fontsize=10)
        axes[1, i].axis("off")

    fig.suptitle("SEM Pore Segmentation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_bar_chart(results, output_path, dpi):
    """Bar chart of mean pore diameter per image."""
    if not results:
        return

    names = [r["image"] for r in results]
    means = [r["mean_diameter_um"] for r in results]
    stds = [r["std_um"] for r in results]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    x = range(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color="#2196F3", edgecolor="white", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean pore diameter (µm)", fontsize=11)
    ax.set_title("Pore Equivalent Diameter per Image", fontsize=13, fontweight="bold")

    # Value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output) if args.output else input_dir / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calibration
    px_per_um = args.image_width / args.hfw
    min_area_px = args.min_area * (px_per_um ** 2)

    print(f"SEM Pore Analysis")
    print(f"{'=' * 50}")
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Calibration: {px_per_um:.2f} px/µm  ({args.image_width} px / {args.hfw} µm)")
    print(f"Threshold:   Block {args.block_size}, C {args.c_value}")
    print(f"Filters:     min area {args.min_area} µm², max diameter {args.max_diameter} µm")
    print(f"{'=' * 50}\n")

    # Collect images
    images = collect_images(input_dir, args.magnification_filter)
    if not images:
        print("No .tif images found in the input directory.")
        sys.exit(1)

    print(f"Found {len(images)} image(s)\n")

    results = []
    masks_dict = {}

    for img_path in images:
        diameters, mask, contours = analyze_image(
            img_path, args.block_size, args.c_value, args.blur_kernel,
            args.crop_height, args.morph_kernel, px_per_um, min_area_px,
            args.max_diameter,
        )

        if diameters is None:
            print(f"  [SKIP] {img_path.name} — could not read")
            continue

        name = img_path.stem
        n_pores = len(diameters)

        if n_pores > 0:
            arr = np.array(diameters)
            mean_d = float(np.mean(arr))
            std_d = float(np.std(arr))
            median_d = float(np.median(arr))
            min_d = float(np.min(arr))
            max_d = float(np.max(arr))
        else:
            mean_d = std_d = median_d = min_d = max_d = 0.0

        results.append({
            "image": name,
            "n_pores": n_pores,
            "mean_diameter_um": round(mean_d, 2),
            "std_um": round(std_d, 2),
            "median_um": round(median_d, 2),
            "min_um": round(min_d, 2),
            "max_um": round(max_d, 2),
        })

        masks_dict[name] = mask

        # Save binary mask
        if mask is not None:
            mask_path = output_dir / f"{name}_mask.{args.format}"
            cv2.imwrite(str(mask_path), mask)

        print(f"  {name}: {n_pores} pores, mean {mean_d:.2f} ± {std_d:.2f} µm")

    # ── Write CSV ───────────────────────────────────────────────────────────
    csv_path = output_dir / "pore_analysis_results.csv"
    fieldnames = ["image", "n_pores", "mean_diameter_um", "std_um",
                  "median_um", "min_um", "max_um"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # ── Composite figure ────────────────────────────────────────────────────
    composite_path = output_dir / f"composite_segmentation.{args.format}"
    make_composite_figure(images, masks_dict, composite_path, args.dpi)
    print(f"Composite figure saved: {composite_path}")

    # ── Bar chart ───────────────────────────────────────────────────────────
    chart_path = output_dir / f"pore_diameter_chart.{args.format}"
    make_bar_chart(results, chart_path, args.dpi)
    print(f"Bar chart saved: {chart_path}")

    # ── Parameters log ──────────────────────────────────────────────────────
    params_path = output_dir / "analysis_parameters.txt"
    with open(params_path, "w") as f:
        f.write("SEM Pore Analysis — Parameters\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input directory:    {input_dir}\n")
        f.write(f"Images analysed:    {len(results)}\n")
        f.write(f"Block size:         {args.block_size}\n")
        f.write(f"C value:            {args.c_value}\n")
        f.write(f"Image width (px):   {args.image_width}\n")
        f.write(f"HFW (µm):           {args.hfw}\n")
        f.write(f"Pixels per µm:      {px_per_um:.2f}\n")
        f.write(f"Min pore area:      {args.min_area} µm²\n")
        f.write(f"Max pore diameter:  {args.max_diameter} µm\n")
        f.write(f"Blur kernel:        {args.blur_kernel}×{args.blur_kernel}\n")
        f.write(f"Crop height:        {args.crop_height} px\n")
        f.write(f"Morph kernel:       {args.morph_kernel}×{args.morph_kernel}\n")
        f.write(f"Magnification filter: {args.magnification_filter or 'none'}\n")
        f.write(f"Output DPI:         {args.dpi}\n")
        f.write(f"Output format:      {args.format}\n")
    print(f"Parameters saved: {params_path}")

    print(f"\nDone — {len(results)} image(s) processed.")


if __name__ == "__main__":
    main()
