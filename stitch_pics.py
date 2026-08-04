#!/usr/bin/env python3
"""
截图拼接工具：自动检测重叠并拼接成一张完整大图。

用法:
  python stitch_pics.py                  # 拼接当前目录下全部图片
  python stitch_pics.py -i ./tiles       # 指定输入目录
  python stitch_pics.py -o out.png       # 指定输出文件
  python stitch_pics.py 1.png 2.png 3.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_stdio() -> None:
    """避免 Windows 控制台中文乱码 / 编码异常，并关闭输出缓冲。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass


def log(msg: str) -> None:
    print(msg, flush=True)


def log_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


_setup_stdio()
log("stitch_pics: 启动中...")

try:
    import cv2
    import numpy as np
except ImportError as e:
    log_err(f"缺少依赖: {e}")
    log_err("请先安装: pip install opencv-python numpy")
    raise SystemExit(1) from e

log(f"OpenCV {cv2.__version__}, 工作目录: {Path.cwd()}")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_images(paths: list[Path]) -> list[tuple[Path, np.ndarray]]:
    images = []
    for p in paths:
        # Windows 下中文路径用 imdecode 更稳妥
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            log_err(f"警告: 无法读取 {p}，已跳过")
            continue
        log(f"  已加载 {p.name} ({img.shape[1]}x{img.shape[0]})")
        images.append((p, img))
    if len(images) < 2:
        raise SystemExit(f"至少需要 2 张有效图片，当前只有 {len(images)} 张")
    return images


def collect_paths(args: argparse.Namespace) -> list[Path]:
    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        root = Path(args.input).resolve()
        if not root.is_dir():
            raise SystemExit(f"输入目录不存在: {root}")
        paths = sorted(
            p for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and p.name != Path(args.output).name
        )
        log(f"扫描目录: {root}")
    if not paths:
        raise SystemExit(
            "未找到可拼接的图片。请确认：\n"
            "  1) 当前工作目录是否正确（cd 到放图片的目录）\n"
            "  2) 图片扩展名为 png/jpg/webp/bmp/tif\n"
            "  3) 或用: python stitch_pics.py 1.png 2.png"
        )
    return paths


def match_pair(img_a: np.ndarray, img_b: np.ndarray, max_features: int = 5000):
    """估计 img_b 相对 img_a 的仿射变换。返回 (M, inlier_count) 或 None。"""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(max_features)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    if not knn or len(knn[0]) < 2:
        return None
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(
        dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None or mask is None:
        return None
    return M, int(mask.sum())


def edge_weight(h: int, w: int) -> np.ndarray:
    wy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float64)
    wx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float64)
    wy = np.maximum(wy, 1.0)
    wx = np.maximum(wx, 1.0)
    return np.minimum(wy[:, None], wx[None, :])


def sample_background(img: np.ndarray) -> np.ndarray:
    patches = [
        img[10:40, 10:40],
        img[10:40, -40:-10],
        img[-40:-10, 10:40],
        img[-40:-10, -40:-10],
    ]
    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    return np.median(samples, axis=0).astype(np.uint8)


def compose_affine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def estimate_transforms(
    images: list[np.ndarray],
    max_features: int = 5000,
) -> list[np.ndarray]:
    n = len(images)
    transforms = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)]
    for i in range(1, n):
        best = None
        for j in range(i):
            log(f"  匹配 图[{i}] <-> 图[{j}] ...")
            result = match_pair(images[j], images[i], max_features=max_features)
            if result is None:
                continue
            M_ji, inliers = result
            M_ref = compose_affine(transforms[j], M_ji)
            if best is None or inliers > best[0]:
                best = (inliers, M_ref, j)
        if best is None:
            raise SystemExit(f"无法将第 {i} 张图与已有图片对齐")
        inliers, M_ref, j = best
        dx, dy = M_ref[0, 2], M_ref[1, 2]
        log(f"  图[{i}] <- 图[{j}]: 内点={inliers}, 平移=({dx:.1f}, {dy:.1f})")
        transforms.append(M_ref)
    return transforms


def stitch(images: list[np.ndarray], transforms: list[np.ndarray]) -> np.ndarray:
    corners = []
    for img, M in zip(images, transforms):
        h, w = img.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.transform(pts, M)
        corners.append(warped.reshape(-1, 2))

    all_pts = np.vstack(corners)
    min_x = int(np.floor(all_pts[:, 0].min()))
    min_y = int(np.floor(all_pts[:, 1].min()))
    max_x = int(np.ceil(all_pts[:, 0].max()))
    max_y = int(np.ceil(all_pts[:, 1].max()))

    shift = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y]], dtype=np.float64)
    cw, ch = max_x - min_x, max_y - min_y
    log(f"画布尺寸: {cw} x {ch}")

    canvas = np.zeros((ch, cw, 3), dtype=np.float64)
    weight = np.zeros((ch, cw), dtype=np.float64)

    for idx, (img, M) in enumerate(zip(images, transforms)):
        log(f"  融合第 {idx} 张...")
        M_shifted = compose_affine(shift, M)
        warped = cv2.warpAffine(
            img, M_shifted, (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        h, w = img.shape[:2]
        wmap = edge_weight(h, w).astype(np.float32)
        wmap_u8 = np.clip(wmap / wmap.max() * 255, 0, 255).astype(np.uint8)
        warped_w = cv2.warpAffine(
            wmap_u8, M_shifted, (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float64)

        mask = warped_w > 0
        canvas[mask] += warped[mask].astype(np.float64) * warped_w[mask, None]
        weight[mask] += warped_w[mask]

    empty = weight < 1e-6
    weight_safe = np.maximum(weight, 1e-6)
    result = (canvas / weight_safe[:, :, None]).clip(0, 255).astype(np.uint8)
    result[empty] = sample_background(images[0])
    return result


def save_image(path: Path, img: np.ndarray) -> None:
    """支持中文路径的保存。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise SystemExit(f"编码失败: {path}")
    buf.tofile(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动检测重叠区域并将多张截图拼接成一张大图"
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="待拼接图片路径（不填则扫描输入目录）",
    )
    parser.add_argument(
        "-i", "--input",
        default=".",
        help="输入目录（默认当前目录）",
    )
    parser.add_argument(
        "-o", "--output",
        default="stitched.png",
        help="输出文件路径（默认 stitched.png）",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="SIFT 最大特征点数（默认 5000）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_paths(args)
    log(f"找到 {len(paths)} 张图片:")
    for p in paths:
        log(f"  - {p}")

    loaded = load_images(paths)
    imgs = [img for _, img in loaded]

    log("正在估计相对位置（SIFT，可能较慢）...")
    transforms = estimate_transforms(imgs, max_features=args.max_features)

    log("正在拼接...")
    result = stitch(imgs, transforms)

    out = Path(args.output)
    save_image(out, result)
    h, w = result.shape[:2]
    log(f"已保存: {out.resolve()} ({w}x{h})")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log_err(f"运行失败: {type(e).__name__}: {e}")
        raise
