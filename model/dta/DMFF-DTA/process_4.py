import pandas as pd
import os

# ===================== 第一步：配置文件路径 =====================
# 三张待处理表的路径（替换为你的实际路径）
train_path = "benchmark/kiba/train.csv"
val_path = "benchmark/kiba/val.csv"
test_path = "benchmark/kiba/test.csv"
# 输出目录（过滤后的文件保存路径，默认当前目录）
output_dir = "benchmark/kiba/"

# ===================== 第二步：定义删除指定uniprot行的函数 =====================
def remove_uniprot_row(input_csv, output_csv, target_uniprot="P78527"):
    """
    删除CSV中uniprot列为指定值的行
    :param input_csv: 输入CSV路径（train/val/test）
    :param output_csv: 输出CSV路径（过滤后）
    :param target_uniprot: 要删除的uniprot值（此处为P78527）
    """
    # 1. 读取CSV
    df = pd.read_csv(input_csv, encoding="utf-8")
    
    # 2. 检查uniprot列是否存在
    if "Protein_id" not in df.columns:
        print(f"错误：{os.path.basename(input_csv)} 中无uniprot列！")
        return
    
    # 3. 统计删除前的行数
    before_count = len(df)
    
    # 4. 核心：过滤掉uniprot=P78527的行（!= 保留不等于的行）
    df_filtered = df[df["Protein_id"] != target_uniprot]
    
    # 5. 统计删除后的行数
    after_count = len(df_filtered)
    deleted_count = before_count - after_count
    
    # 6. 保存过滤后的文件
    df_filtered.to_csv(output_csv, index=False, encoding="utf-8")
    
    # 7. 打印结果
    print(f"{os.path.basename(input_csv)} 处理完成：")
    print(f"  删除前行数：{before_count}")
    print(f"  删除后行数：{after_count}")
    print(f"  共删除 {deleted_count} 行（uniprot=P78527）\n")

# ===================== 第三步：批量处理train/val/test =====================
for csv_path in [train_path, val_path, test_path]:
    # 构建输出文件名（如train_with_uniprot.csv → train_filtered.csv）
    file_name = os.path.basename(csv_path)
    name, ext = os.path.splitext(file_name)
    output_path = os.path.join(output_dir, f"{name}_filtered{ext}")
    
    # 执行删除操作
    remove_uniprot_row(csv_path, output_path, target_uniprot="P78527")

print("所有表过滤完成！过滤后的文件已保存至：", output_dir)
