import os.path
import numpy as np
import pandas as pd
import torch
import esm
import argparse
import os

device='cuda:0'
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()


def get_pretrained_embedding(s):

    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", s)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens=batch_tokens.to(device)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1])

    return sequence_representations[0].cpu().numpy()


def generate_feature(args):

    data_path = args.root_data_path
    dataset = args.dataset
    output_data_path = data_path + '/' + dataset +"/protein"


    opts = ['train','val','test']

    for o in opts:

        if not os.path.exists(output_data_path+'/'+o):
            # 如果文件夹不存在，则创建
            os.makedirs(output_data_path +'/'+ o)
            print(f"{output_data_path} created")
        else:
            print(f"{output_data_path} exists")

        count = 0
        id_list = []
        raw_data = pd.read_csv(f'{data_path}/{dataset}/{o}.csv')
        sequence_values = raw_data['Protein'].values
        for i, s in enumerate(sequence_values):
            id = str(raw_data['Protein_index'][i])
            if id in id_list:
                continue
            if os.path.isfile(f'{output_data_path}/{o}/' + id + '.npy'):
                continue
            print(id)
            seq_emb = get_pretrained_embedding(s.upper())
            print(seq_emb.shape)
            np.save(f'{output_data_path}/{o}/' + id, seq_emb)
            print(raw_data['Protein_index'][i])
            id_list.append(id)
            count += 1
    print(count)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r','--root_data_path', type=str, default='benchmark', help='Raw Data Path')
    parser.add_argument('-d','--dataset', type=str, default='kiba_dta', help='Datasets')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params)
    generate_feature(params)
