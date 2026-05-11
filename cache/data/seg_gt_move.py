import os
import shutil
from pathlib import Path

def extract_segmentation_files(source_dir, target_dir, start_idx=4500, end_idx=5000):
    """
    遍历指定切片范围，提取 segmentation.tif 文件并复制到目标文件夹。
    """
    src_path = Path(source_dir)
    tgt_path = Path(target_dir)
    
    # 确保目标文件夹存在，如果不存在则创建
    tgt_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    missing_count = 0

    # 遍历 4500 到 5000（包含 5000）
    for i in range(start_idx, end_idx + 1):
        # 格式化文件夹名称，保持 5 位数字的零填充 (例如：slice04500)
        slice_name = f"slice{i:05d}"
        
        # 尝试使用小写的 mode2 (根据你的路径示例)
        file_path = src_path / slice_name / "mode2" / "segmentation.tif"
        
        # 如果小写没找到，尝试大写的 Mode2 (根据你的文字描述)
        if not file_path.exists():
            file_path = src_path / slice_name / "Mode2" / "segmentation.tif"

        if file_path.exists():
            # 为防止同名覆盖，在目标文件夹中重命名文件
            new_file_name = f"{slice_name}_segmentation.tif"
            destination = tgt_path / new_file_name
            
            # 复制文件并保留元数据
            shutil.copy2(file_path, destination)
            print(f"成功提取: {slice_name} -> {new_file_name}")
            success_count += 1
        else:
            print(f"未找到文件，已跳过: {src_path / slice_name}")
            missing_count += 1

    print("-" * 30)
    print(f"提取完成！成功复制 {success_count} 个文件，未找到 {missing_count} 个文件。")

if __name__ == "__main__":
    # 你的源文件夹路径 (无需包含 sliceXXXXX)
    SOURCE_DIRECTORY = "/ibex/user/wangz0r/CS_300_Final_Project/2DeteCT/2DeteCT_RecSeg/2DeteCT_slices_RecSeg_All"
    
    # 你想存放提取文件的目标文件夹路径 (请修改为你实际想要的路径)
    TARGET_DIRECTORY = "/ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/data/00_SEG_GT"
    
    extract_segmentation_files(SOURCE_DIRECTORY, TARGET_DIRECTORY)