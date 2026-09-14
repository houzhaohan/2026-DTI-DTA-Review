import pandas as pd

def merge_protein_data(a_csv_path, protein_csv_path, output_csv_path):
    """
    合并两个CSV文件，根据Protein_id（对应target_key）补充列数据
    
    参数：
    a_csv_path: 第一个CSV文件（a.csv）的路径
    protein_csv_path: 第二个CSV文件（protein.csv）的路径
    output_csv_path: 输出合并后文件的路径
    """
    # 1. 读取两个CSV文件
    try:
        # 读取a.csv，将target_key设为索引，方便后续匹配
        df_a = pd.read_csv(a_csv_path, index_col='target_key')
        # 读取protein.csv
        df_protein = pd.read_csv(protein_csv_path)
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        return
    except Exception as e:
        print(f"读取文件时出错 - {e}")
        return


    # 2. 选择需要添加的列
    columns_to_add = [
        'target_sequence', 
        'target_sequence_start', 
        'target_sequence_end', 
    ]
    
    # 3. 检查a.csv是否包含所需列
    missing_cols = [col for col in columns_to_add if col not in df_a.columns]
    if missing_cols:
        print(f"警告：a.csv 缺少以下列 - {missing_cols}")
        return
    
    # 4. 根据Protein_id匹配并添加列
    # 遍历需要添加的列，逐个匹配
    for col in columns_to_add:
        # map方法：用Protein_id匹配df_a的索引（target_key），获取对应列的值
        df_protein[col] = df_protein['Protein_id'].map(df_a[col])
    
    # 5. 处理匹配不到的情况（填充为NaN，也可以改为空字符串''）
    df_protein = df_protein.fillna('')  # 如需保留NaN，可注释这行
    
    # 6. 保存结果到新CSV文件
    try:
        df_protein.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"合并完成！结果已保存到：{output_csv_path}")
    except Exception as e:
        print(f"保存文件时出错 - {e}")
        return
    
    return df_protein

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 请替换为你的实际文件路径
    a_csv = "b.csv"
    protein_csv = "benchmark/kiba_dta/b.csv"
    output_csv = "kiba_protein.csv"
    
    # 执行合并
    merged_df = merge_protein_data(a_csv, protein_csv, output_csv)
    
    # 可选：打印前5行查看结果
    if merged_df is not None:
        print("\n合并后的数据预览：")
        print(merged_df.head())
