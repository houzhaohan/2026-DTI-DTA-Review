"""
生成 kiba.npz - 蛋白质 ESM 预训练嵌入
用法: python generate_npz.py [dataset_name]
"""
import sys
import os
import numpy as np
import pandas as pd
import esm
import torch

def generate_protein_embeddings(dataset_name='kiba', data_dir='data/kiba', output_dir='.'):
    # 1. 读取所有 CSV，收集唯一蛋白质
    print(f'Loading {dataset_name} dataset from {data_dir}...')
    dfs = []
    for fname in ['train', 'val', 'test']:
        fpath = os.path.join(data_dir, f'{fname}.csv')
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))
            print(f'  Loaded {fname}.csv: {len(dfs[-1])} rows')
    
    all_data = pd.concat(dfs, ignore_index=True)
    
    # 去重，保留 (Protein_index, Protein) 对
    unique_proteins = all_data.drop_duplicates(subset=['Protein_index'])[
        ['Protein_index', 'Protein']
    ].sort_values('Protein_index').reset_index(drop=True)
    
    print(f'Total unique proteins: {len(unique_proteins)}')
    print(f'Protein_index range: {unique_proteins["Protein_index"].min()} - {unique_proteins["Protein_index"].max()}')
    
    # 2. 加载 ESM 模型
    print('\nLoading ESM-1b model...')
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    # 使用 GPU（如果可用）
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        print('Using GPU for ESM inference')
    else:
        print('Using CPU for ESM inference')
    
    # 3. 准备数据
    prots = unique_proteins['Protein'].tolist()
    pids = unique_proteins['Protein_index'].tolist()
    
    # ESM-1b 最大 token 数（含特殊 token）为 1024，所以蛋白序列最多 1022
    MAX_LEN = 1022
    prots_tuple = [(str(i), prots[i][:MAX_LEN]) for i in range(len(prots))]
    
    # 4. 批量生成嵌入
    BATCH_SIZE = 4  # 根据 GPU 显存调整
    data_dict = {}
    
    print(f'\nGenerating ESM embeddings (batch_size={BATCH_SIZE})...')
    total_batches = (len(prots) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(prots))
        
        batch_data = prots_tuple[start:end]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        
        if use_cuda:
            batch_tokens = batch_tokens.cuda()
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        # results["representations"][33] shape: (batch_size, seq_len, 1280)
        token_representations = results["representations"][33].cpu().numpy()
        
        # 将每个蛋白质的嵌入存入字典，key 是 Protein_index
        for j in range(end - start):
            original_idx = start + j
            pid = pids[original_idx]
            # 提取有效长度的表示（去掉 padding）
            valid_len = len(prots[original_idx][:MAX_LEN]) + 2  # +2 for BOS/EOS
            emb = token_representations[j, :valid_len, :]
            data_dict[pid] = emb
        
        print(f'  Batch {batch_idx+1}/{total_batches}: processed {end}/{len(prots)} proteins')
    
    # 5. 保存为 npz（与原始 datahelper.py 格式一致）
    output_path = os.path.join(output_dir, f'{dataset_name}.npz')
    np.savez(output_path, dict=data_dict)
    print(f'\nSaved {dataset_name}.npz to {output_path}')
    print(f'data_dict keys: {len(data_dict)} proteins')
    
    # 验证加载
    z = np.load(output_path, allow_pickle=True)
    loaded_dict = z['dict'][()]
    print(f'Verified: loaded {len(loaded_dict)} protein embeddings')
    
    # 打印一个样例的形状
    sample_key = list(loaded_dict.keys())[0]
    print(f'Sample embedding shape (pid={sample_key}): {loaded_dict[sample_key].shape}')
    
    print('\nDone!')
    return output_path

if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'kiba'
    data_dir = sys.argv[2] if len(sys.argv) > 2 else f'data/{dataset}'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '.'
    
    generate_protein_embeddings(dataset, data_dir, output_dir)
