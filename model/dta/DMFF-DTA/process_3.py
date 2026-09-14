import pandas as pd
import os

def merge_protein_info_to_datasets(protein_csv_path, dataset_dir, output_dir):
    """
    将protein.csv中的信息根据Protein_index合并到train/val/test.csv中
    
    参数：
    protein_csv_path: protein.csv文件的路径
    dataset_dir: 存放train/val/test.csv的目录（如benchmark/davis_dta）
    output_dir: 输出合并后文件的目录（会自动创建）
    """
    # 1. 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 读取protein.csv，将Protein_index设为索引，方便快速匹配
    try:
        df_protein = pd.read_csv(protein_csv_path, index_col='Protein_index')
        print(f"成功读取protein.csv，共{len(df_protein)}条蛋白数据")
    except FileNotFoundError:
        print(f"错误：找不到protein.csv文件 - {protein_csv_path}")
        return
    except Exception as e:
        print(f"读取protein.csv出错：{e}")
        return
    
    # 3. 定义需要从protein.csv中添加的列
    columns_to_add = [
        'Protein_id', 'target_sequence',
        'target_sequence_start', 'target_sequence_end'
    ]
    
    # 检查protein.csv是否包含所需列
    missing_cols = [col for col in columns_to_add if col not in df_protein.columns]
    if missing_cols:
        print(f"警告：protein.csv 缺少以下列 - {missing_cols}")
        return
    
    # 4. 定义需要处理的三个数据集文件
    dataset_files = ['train.csv', 'val.csv', 'test.csv']
    
    # 5. 批量处理每个数据集文件
    for file_name in dataset_files:
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(file_path):
            print(f"警告：找不到文件 {file_path}，跳过该文件")
            continue
        
        # 读取数据集文件
        df_data = pd.read_csv(file_path)
        
        # 检查是否包含Protein_index列
        if 'Protein_index' not in df_data.columns:
            print(f"警告：{file_name} 缺少Protein_index列，跳过该文件")
            continue
        
        # 核心：根据Protein_index匹配并添加列
        for col in columns_to_add:
            df_data[col] = df_data['Protein_index'].map(df_protein[col])
        
        # 处理匹配不到的数据（填充为空字符串，也可保留NaN）
        df_data = df_data.fillna('')
        
        # 保存合并后的文件
        output_path = os.path.join(output_dir, file_name)
        df_data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"已完成 {file_name} 的合并，结果保存到：{output_path}")
        
        # 可选：打印前2行预览
        print(f"{file_name} 合并后前2行预览：")
        print(df_data.head(2))
        print("-" * 50)

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 请根据你的实际路径修改以下参数
    PROTEIN_CSV_PATH = "kiba_protein.csv"  # protein.csv的路径
    DATASET_DIR = "benchmark/kiba_dta"  # 存放train/val/test.csv的目录
    OUTPUT_DIR = "benchmark/kiba"  # 输出合并后文件的目录
    
    # 执行合并
    merge_protein_info_to_datasets(PROTEIN_CSV_PATH, DATASET_DIR, OUTPUT_DIR)
