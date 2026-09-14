"""
一键设置脚本：下载 ChemBoost 缺失的数据文件并训练 lingo2vec 嵌入。
在云主机项目根目录运行： python setup_cloud_data.py
"""
import os
import json
import subprocess
import pandas as pd

# ==================== 1. 从 GitHub 下载可用的数据文件 ====================
GITHUB_RAW = "https://raw.githubusercontent.com/boun-tabi/chemboost/master"

downloads = [
    # sw_8mer / sw_x / sw_random 都需要的蛋白相似度向量
    ("data/kiba/sw_sim_vectors.json",
     f"{GITHUB_RAW}/data/kiba/sw_sim_vectors.json"),
    # 以下文件可选，其他模型可能需要
    ("data/kiba/sw_sim_matrix.csv",
     f"{GITHUB_RAW}/data/kiba/sw_sim_matrix.csv"),
    ("data/kiba/ligands.json",
     f"{GITHUB_RAW}/data/kiba/ligands.json"),
    ("data/kiba/proteins.json",
     f"{GITHUB_RAW}/data/kiba/proteins.json"),
    ("data/bpe/chembl23_20K.model",
     f"{GITHUB_RAW}/data/bpe/chembl23_20K.model"),
    ("data/bpe/chembl23_20K.vocab",
     f"{GITHUB_RAW}/data/bpe/chembl23_20K.vocab"),
    ("data/bpe/chembl23_20K.txt",
     f"{GITHUB_RAW}/data/bpe/chembl23_20K.txt"),
]

print("=" * 60)
print("Step 1: 从 GitHub 下载数据文件")
print("=" * 60)

for local_path, url in downloads:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        print(f"  [跳过] {local_path} 已存在")
        continue
    print(f"  [下载] {local_path} ...")
    result = subprocess.run(
        ["wget", "-q", "-O", local_path, url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # 尝试 curl
        result = subprocess.run(
            ["curl", "-sL", "-o", local_path, url],
            capture_output=True, text=True
        )
    if result.returncode == 0 and os.path.getsize(local_path) > 0:
        print(f"    完成 ({os.path.getsize(local_path)} bytes)")
    else:
        print(f"    ❌ 失败: {result.stderr.strip()}")
        if os.path.exists(local_path):
            os.remove(local_path)

# ==================== 2. 训练 lingo2vec 嵌入 ====================
print()
print("=" * 60)
print("Step 2: 训练 lingo2vec 词嵌入 (这是 sw_8mer 模型需要的)")
print("=" * 60)

from gensim.models import Word2Vec
from src.representation_models.utils import get_lingos

# 收集所有 SMILES
data_dir = './data/kiba/'
dfs = []
for fname in ['train.csv', 'val.csv', 'test.csv']:
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        dfs.append(pd.read_csv(fpath))

if not dfs:
    print("❌ 未找到 kiba CSV 文件！")
    exit(1)

all_smiles = pd.concat(dfs)['SMILES'].unique().tolist()
print(f"  收集到 {len(all_smiles)} 个唯一 SMILES")

# 切分为 lingos (8-mer)
print("  正在用 get_lingos(smiles, q=8) 切分 SMILES ...")
tokenized = [get_lingos(smiles, q=8) for smiles in all_smiles]

# 过滤空列表
tokenized = [t for t in tokenized if len(t) > 0]
print(f"  有效分子文档: {len(tokenized)}")

# 训练 Word2Vec (参数跟 word2vec_trainer.py 保持一致)
print("  训练 Word2Vec (dim=100, window=20, sg=1, iter=20) ...")
model = Word2Vec(
    tokenized, vector_size=100, window=20, workers=4,
    min_count=1, sample=1e-4, negative=5, epochs=20, sg=1, hs=0
)

# 保存
save_dir = './data/embeddings'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'chembl_lingo_embeddings')
model.save(f'{save_path}.model')
model.wv.save_word2vec_format(f'{save_path}.kv')
print(f"  ✅ 已保存到 {save_path}.kv")

# 额外: 如果以后需要 bpe2vec 嵌入，可以用同样方法训练
# 这里先跳过，等需要 bpe 模型时再说

# ==================== 3. 验证 ====================
print()
print("=" * 60)
print("Step 3: 验证关键文件")
print("=" * 60)

required = [
    ("data/kiba/sw_sim_vectors.json", "sw_8mer / sw_x / sw_random 都需要"),
    ("data/embeddings/chembl_lingo_embeddings.kv", "sw_8mer / random_8mer / x_8mer 等需要"),
]

all_ok = True
for fpath, desc in required:
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    status = f"✅ {size} bytes" if exists else "❌ 缺失"
    print(f"  {status} | {fpath} ({desc})")
    if not exists:
        all_ok = False

print()
if all_ok:
    print("🎉 所有关键文件就绪！现在可以运行:")
    print("   python -m src.runner --dataset kiba --model sw_8mer --savefile exp1")
else:
    print("⚠️  有文件缺失，请检查上面的错误信息。")
