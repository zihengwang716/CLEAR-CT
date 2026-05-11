"""
Batch add Poisson noise to Mode 3 raw sinograms (before flat-field correction).
模拟 beam hardening + low-dose 的复合退化数据。

思路：不反缩放，直接输出低光子数的 sinogram，同时按同比例缩放 flat。
     dark 不缩放（暗场是探测器电子学噪声，与 X 射线剂量无关）。

用法：
  python add_poisson_to_mode3.py --dose_ratio 0.05 --seed 42
  python add_poisson_to_mode3.py --dose_ratio 0.05 --start 1 --end 100  # 只处理部分 slice

文件结构：
  输入: /ibex/user/huh0a/cs300/project/2DeteCT_slicesAll/slice{:05d}/mode3/
        ├── sinogram.tif
        ├── dark.tif
        ├── flat1.tif
        └── flat2.tif

  输出: /ibex/user/huh0a/cs300/project/2DeteCT_slicesAllmode3/slice{:05d}/mode3/
        ├── sinogram_with_poisson.tif   (加噪后的 sinogram)
        ├── dark.tif                    (原始 dark，直接复制，不缩放)
        ├── flat1_scaled.tif            (按比例缩放的 flat1)
        └── flat2_scaled.tif            (按比例缩放的 flat2)
"""

import os
import numpy as np
import argparse

try:
    import imageio
except ImportError:
    import imageio.v2 as imageio


# ============ 路径配置 ============
INPUT_BASE = "/ibex/user/huh0a/cs300/project/2DeteCT_slicesAll"
OUTPUT_BASE = "/ibex/user/huh0a/cs300/project/2DeteCT_slicesAllmode3"


def add_poisson_noise_no_rescale(sinogram, dark, dose_ratio, rng):
    """
    在 raw sinogram 上模拟低剂量泊松噪声，不反缩放。

    输出的 sinogram 信号量级 ≈ 原始的 dose_ratio 倍，
    噪声效果明显（和真实低剂量一致）。

    Parameters
    ----------
    sinogram : np.ndarray
        原始探测器读数 (n_angles, n_detectors)
    dark : np.ndarray
        暗场读数
    dose_ratio : float
        目标剂量 / 原始剂量
    rng : np.random.Generator
        随机数生成器

    Returns
    -------
    sinogram_noisy : np.ndarray
        加噪后的低剂量 sinogram（信号量级降低，噪声明显）
    """
    original_dtype = sinogram.dtype

    # 提取 dark frame
    dark_frame = dark.mean(axis=0) if dark.ndim > 1 and dark.shape[0] > 1 else dark[0] if dark.ndim > 1 else dark

    # 减去 dark 得到净光子信号
    signal = sinogram.astype(np.float64) - dark_frame.astype(np.float64)
    signal = np.clip(signal, 0, None)

    # 缩放到低剂量水平
    scaled_signal = signal * dose_ratio

    # 泊松采样（不反缩放！）
    noisy_signal = rng.poisson(scaled_signal).astype(np.float64)

    # 加回原始 dark（dark 是电子学噪声，与剂量无关，不缩放）
    noisy_sinogram = noisy_signal + dark_frame.astype(np.float64)

    # 保持原始数据类型
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
        noisy_sinogram = np.clip(noisy_sinogram, 0, max_val)
    else:
        noisy_sinogram = np.clip(noisy_sinogram, 0, None)

    return noisy_sinogram.astype(original_dtype)


def scale_flat_frame(flat, dark, dose_ratio):
    """
    按 dose_ratio 缩放 flat 的净信号部分，保留原始 dark 基底。

    物理上：flat = (X 射线信号) + (暗电流)
    低剂量时只有 X 射线信号按比例降低，暗电流不变。
    所以：flat_scaled = (flat - dark) * dose_ratio + dark
    """
    original_dtype = flat.dtype

    # 提取 dark frame
    dark_frame = dark.mean(axis=0) if dark.ndim > 1 and dark.shape[0] > 1 else dark[0] if dark.ndim > 1 else dark

    # 只缩放净信号部分
    flat_signal = flat.astype(np.float64) - dark_frame.astype(np.float64)
    flat_signal = np.clip(flat_signal, 0, None)
    scaled = flat_signal * dose_ratio + dark_frame.astype(np.float64)

    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
        scaled = np.clip(scaled, 0, max_val)
    else:
        scaled = np.clip(scaled, 0, None)

    return scaled.astype(original_dtype)


def process_slice(slice_idx, dose_ratio, rng, dry_run=False):
    """处理单个 slice：加噪 sinogram + 缩放 flat（dark 不缩放）。"""
    slice_name = f"slice{slice_idx:05d}"
    input_dir = os.path.join(INPUT_BASE, slice_name, "mode3")
    output_dir = os.path.join(OUTPUT_BASE, slice_name, "mode3")

    # 检查输入文件
    required_files = ["sinogram.tif", "dark.tif", "flat1.tif", "flat2.tif"]
    for f in required_files:
        if not os.path.exists(os.path.join(input_dir, f)):
            print(f"  [SKIP] {slice_name}: {f} not found")
            return False

    if dry_run:
        print(f"  [DRY RUN] {slice_name}: would process")
        return True

    # 读取所有文件
    sinogram = imageio.imread(os.path.join(input_dir, "sinogram.tif"))
    dark = imageio.imread(os.path.join(input_dir, "dark.tif"))
    flat1 = imageio.imread(os.path.join(input_dir, "flat1.tif"))
    flat2 = imageio.imread(os.path.join(input_dir, "flat2.tif"))

    # 加泊松噪声（不反缩放）
    noisy_sinogram = add_poisson_noise_no_rescale(sinogram, dark, dose_ratio, rng)

    # flat 只缩放净信号部分（减去 dark 后缩放，再加回 dark）
    # dark 不缩放（电子学噪声，与剂量无关）
    flat1_scaled = scale_flat_frame(flat1, dark, dose_ratio)
    flat2_scaled = scale_flat_frame(flat2, dark, dose_ratio)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    imageio.imwrite(os.path.join(output_dir, "sinogram_with_poisson.tif"), noisy_sinogram)
    imageio.imwrite(os.path.join(output_dir, "dark.tif"), dark)  # 原始 dark，直接复制
    imageio.imwrite(os.path.join(output_dir, "flat1_scaled.tif"), flat1_scaled)
    imageio.imwrite(os.path.join(output_dir, "flat2_scaled.tif"), flat2_scaled)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add Poisson noise to Mode 3 sinograms (no rescale, with scaled dark/flat)"
    )
    parser.add_argument("--dose_ratio", type=float, default=0.05,
                        help="Target dose / original dose ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--start", type=int, default=1,
                        help="Start slice index (default: 1)")
    parser.add_argument("--end", type=int, default=5000,
                        help="End slice index inclusive (default: 5000)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only check which files exist, don't process")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"=== Poisson Noise Addition for Mode 3 (No Rescale) ===")
    print(f"  Input:      {INPUT_BASE}/slice*/mode3/")
    print(f"  Output:     {OUTPUT_BASE}/slice*/mode3/")
    print(f"  dose_ratio: {args.dose_ratio}")
    print(f"  seed:       {args.seed}")
    print(f"  range:      slice{args.start:05d} ~ slice{args.end:05d}")
    print(f"  Output files per slice:")
    print(f"    - sinogram_with_poisson.tif  (noisy sinogram)")
    print(f"    - dark.tif                   (original dark, not scaled)")
    print(f"    - flat1_scaled.tif           (scaled flat1)")
    print(f"    - flat2_scaled.tif           (scaled flat2)")
    print()

    success_count = 0
    skip_count = 0

    for idx in range(args.start, args.end + 1):
        result = process_slice(idx, args.dose_ratio, rng, dry_run=args.dry_run)
        if result:
            success_count += 1
        else:
            skip_count += 1

        # 每 500 个打印一次进度
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{args.end} "
                  f"(processed: {success_count}, skipped: {skip_count})")

    print(f"\nDone! Processed: {success_count}, Skipped: {skip_count}")


if __name__ == "__main__":
    main()
