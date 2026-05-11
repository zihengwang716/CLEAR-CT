import argparse
import os
import numpy as np
import imageio.v2 as imageio

def modify_image(img_path, out_path=None):
    if not os.path.exists(img_path):
        print(f"错误: 找不到文件 {img_path}")
        return

    # 1. 读取图像并记录原始数据类型
    img = imageio.imread(img_path)
    original_dtype = img.dtype
    
    # 转换为 float64 进行计算，防止溢出
    img_float = img.astype(np.float64)

    # 兼容多通道，但通常CT是2D单通道
    h, w = img_float.shape[:2]
    
    # 2. 修改左上角第一个像素 (改成 += 1，防止背景是0时向下溢出被clip回0)
    img_float[0, 0] += 1
    
    # 3. 修改最中心的像素 (改成 += 1，安全起见保持操作一致)
    center_y, center_x = h // 2, w // 2
    img_float[center_y, center_x] += 1

    # 4. 范围保护与类型还原
    if np.issubdtype(original_dtype, np.integer):
        # 如果是整数类型，限制上下界防止溢出
        info = np.iinfo(original_dtype)
        img_modified = np.clip(img_float, info.min, info.max).astype(original_dtype)
    else:
        # 浮点型直接转换回去即可
        img_modified = img_float.astype(original_dtype)

    # 5. 生成并处理保存路径
    if out_path is None:
        # 如果什么都没传，默认在原文件名后加 _modified，存在原目录下
        base, ext = os.path.splitext(img_path)
        final_out_path = f"{base}_modified{ext}"
    elif os.path.isdir(out_path):
        # 如果传入的是一个已存在的文件夹，则把修改后的文件保存在该文件夹内
        filename = os.path.basename(img_path)
        base, ext = os.path.splitext(filename)
        final_out_path = os.path.join(out_path, f"{base}_modified{ext}")
    else:
        # 如果传入的是一个具体的文件路径 (或者是一个还不存在的路径)，则直接使用，但要确保父目录存在
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        final_out_path = out_path

    # 保存图像
    imageio.imwrite(final_out_path, img_modified)
    
    print("\n修改完成！")
    print(f"  - 修改操作: 左上角 (0, 0) 和 中心 ({center_x}, {center_y}) 像素值分别 +1")
    print(f"  - 图像类型: {original_dtype}")
    print(f"  - 保存路径: {final_out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微调GT图的2个特定像素以供Debug")
    parser.add_argument("--img_path", type=str, default="/ibex/user/liuj0s/CS_300/cache/data/GT_TEST/slice00001.tif")
    parser.add_argument("--out_path", type=str, default="/ibex/user/liuj0s/CS_300/cache/data/GT_TEST")
    args = parser.parse_args()

    modify_image(args.img_path, args.out_path)