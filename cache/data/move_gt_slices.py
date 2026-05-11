import argparse
import shutil
from pathlib import Path

def move_and_rename_files(src_base_dir, dst_base_dir):
    """
    遍历指定文件夹，寻找特定 slice 范围内的 tif 文件，重命名并移动到目标文件夹。
    """
    src_path = Path(src_base_dir)
    dst_path = Path(dst_base_dir)

    # 确保目标文件夹存在，如果不存在则自动创建
    dst_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    missing_count = 0

    # 遍历从 1 到 4000 的 slice
    print(f"开始处理，源目录: {src_path}")
    print(f"目标目录: {dst_path}")
    
    for i in range(1, 4001):
        # 格式化字符串，生成 slice00001, slice00002 等格式
        slice_folder_name = f"slice{i:05d}" 
        
        # 拼接源文件的完整路径
        # 例如: .../slice00001/mode2/reconstruction.tif
        source_file = src_path / slice_folder_name / "mode2" / "reconstruction.tif"
        
        # 拼接目标文件的完整路径
        # 例如: .../slice00001.tif
        target_file = dst_path / f"{slice_folder_name}.tif"

        if source_file.exists():
            # 移动文件并重命名
            shutil.copy2(str(source_file), str(target_file))
            success_count += 1
            # print(f"已移动: {slice_folder_name}.tif") # 如果嫌输出太多可以注释掉这行
        else:
            missing_count += 1
            # print(f"未找到文件，跳过: {source_file}")

    print("-" * 30)
    print("处理完成！")
    print(f"成功移动文件数: {success_count}")
    print(f"未找到的文件数: {missing_count}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="提取、重命名并移动特定的 tif 文件。")
    parser.add_argument("--src", type=str, required=True, help="源文件夹的基础路径")
    parser.add_argument("--dst", type=str, required=True, help="目标文件夹的路径")
    
    args = parser.parse_args()
    
    move_and_rename_files(args.src, args.dst)



# python /ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/data/move_gt_slices.py \
#   --src "/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_RecSeg/2DeteCT_slices_RecSeg_All" \
#   --dst "/ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/data/mode1_agd_ds6__20260502_044401/comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner/Train/gt"