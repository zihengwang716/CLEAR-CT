import shutil
from pathlib import Path

def move_slice_files(src_dir, dest_dir, start_idx, end_idx):
    """
    遍历指定文件夹及其子文件夹，移动指定范围的 sliceXXXXX.npy 文件
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    # 检查源文件夹是否存在
    if not src_path.exists():
        print(f"错误: 源文件夹不存在 -> {src_path}")
        return

    # 确保目标文件夹存在，如果不存在则自动创建
    dest_path.mkdir(parents=True, exist_ok=True)

    # 生成需要寻找的目标文件名集合 (利用集合 O(1) 的查找速度)
    # 例如：{'slice00001.npy', 'slice00002.npy', ..., 'slice04000.npy'}
    target_filenames = {f"slice{i:05d}.npy" for i in range(start_idx, end_idx + 1)}

    print(f"正在 {src_dir} 中查找文件...")
    print(f"目标文件范围: slice{start_idx:05d}.npy 到 slice{end_idx:05d}.npy")
    
    moved_count = 0

    # rglob("slice*.npy") 会递归遍历所有子文件夹下的 slice 开头的 npy 文件
    for file_path in src_path.rglob("slice*.npy"):
        if file_path.name in target_filenames:
            # 构建目标文件的完整路径
            target_file_path = dest_path / file_path.name
            
            # 移动文件
            try:
                shutil.move(str(file_path), str(target_file_path))
                print(f"成功移动: {file_path.relative_to(src_path)} -> {dest_path.name}/{file_path.name}")
                moved_count += 1
            except Exception as e:
                print(f"移动文件 {file_path.name} 时出错: {e}")

    print("-" * 40)
    print(f"操作完成！共成功移动了 {moved_count} 个文件。")

if __name__ == "__main__":
    # ================= 配置参数 =================
    
    # 1. 源文件夹路径（你提供的路径）
    SOURCE_DIRECTORY = "/ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/data/mode1_agd_ds6__20260502_044401/comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner/Test/agd_recon"
    
    # 2. 目标文件夹路径（请替换为你实际想移动到的路径）
    TARGET_DIRECTORY = "/ibex/user/wangz0r/CS_300_Final_Project/CS_300/cache/result/ORIGIN_comp__noBH__noW__noTV__ds6__mode1_plain__iter100__rec2048__step0p8__pwr25__noWclip__noWstep__noTVinner" 
    
    # 3. 文件的起始和结束序号 (例如 1 到 4000)
    START_INDEX = 4501
    END_INDEX = 5000
    
    # ===========================================

    move_slice_files(SOURCE_DIRECTORY, TARGET_DIRECTORY, START_INDEX, END_INDEX)