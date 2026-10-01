"""Create repeatable Task 2 inputs and run Mean/Gaussian low-pass filters."""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
RESULTS = ROOT / "results"
RNG = np.random.default_rng(261)


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def make_inputs() -> dict[str, np.ndarray]:
    """Create three deterministic images with different frequency content."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    h, w = 360, 480

    detailed = np.full((h, w), 210, np.uint8)
    for x in range(0, w, 16):
        cv2.line(detailed, (x, 0), (x, h), 80, 1)
    for y in range(0, h, 16):
        cv2.line(detailed, (0, y), (w, y), 80, 1)
    cv2.putText(detailed, "LOW PASS", (75, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 15, 4, cv2.LINE_AA)
    cv2.circle(detailed, (370, 250), 58, 30, 3)
    detailed = np.clip(detailed.astype(np.int16) + RNG.normal(0, 22, detailed.shape), 0, 255).astype(np.uint8)

    x = np.linspace(30, 220, w, dtype=np.float32)
    smooth = np.tile(x, (h, 1))
    smooth += 15 * np.sin(np.linspace(0, 2 * np.pi, h, dtype=np.float32))[:, None]
    smooth = np.clip(smooth, 0, 255).astype(np.uint8)

    contrast = np.full((h, w), 25, np.uint8)
    cv2.rectangle(contrast, (55, 55), (225, 290), 230, -1)
    cv2.circle(contrast, (350, 175), 95, 180, -1)
    cv2.line(contrast, (0, 330), (w, 35), 255, 5)

    images = {"detailed_noisy": detailed, "smooth_gradient": smooth, "high_contrast": contrast}
    for name, image in images.items():
        save_gray(IMAGES / f"{name}.png", image)
    return images


def run_filtering(images: dict[str, np.ndarray]) -> None:
    """Run the two assigned filters and save an individual comparison per image."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_images, all_titles = [], []
    mean_kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    for name, image in images.items():
        mean = cv2.filter2D(image, -1, mean_kernel, borderType=cv2.BORDER_REFLECT)
        gaussian = cv2.GaussianBlur(image, (5, 5), sigmaX=1.2, sigmaY=1.2,
                                    borderType=cv2.BORDER_REFLECT)
        save_gray(RESULTS / f"{name}_mean_k5.png", mean)
        save_gray(RESULTS / f"{name}_gaussian_k5_sigma1.2.png", gaussian)
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, display, title in zip(axes, [image, mean, gaussian],
                                      ["Original", "Mean 5x5", "Gaussian 5x5, sigma=1.2"]):
            ax.imshow(display, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title)
            ax.axis("off")
        figure.suptitle(name.replace("_", " ").title())
        figure.tight_layout()
        figure.savefig(RESULTS / f"{name}_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        all_images.extend([image, mean, gaussian])
        all_titles.extend([f"{name}\nOriginal", "Mean 5x5", "Gaussian 5x5, sigma=1.2"])

    figure, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, display, title in zip(axes.ravel(), all_images, all_titles):
        ax.imshow(display, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    figure.tight_layout()
    figure.savefig(RESULTS / "task2_all_comparisons.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    run_filtering(make_inputs())
    print("Task 2 complete: 3 inputs and 10 result images saved.")
