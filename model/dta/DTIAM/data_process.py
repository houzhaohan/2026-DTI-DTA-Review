import os
import torch
import pickle
import dill as pickle
import json
import esm
import pandas as pd
from bermol.trainer import BerMolPreTrainer
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
max_length = 1000

def remove_forbidden_letters(sequence):
    forbidden_letters = {'B', 'J', 'O', 'U', 'X', 'Z'}
    filtered_sequence = [char for char in sequence if char not in forbidden_letters]
    return ''.join(filtered_sequence)

def cal_comp_feat(data: pd.DataFrame, model_path: str, device: str = "cuda") -> dict:
    """
    Calculate the compound features using the compound pre-trained model
    """
    with open(model_path, "rb") as f:
        comp_model = pickle.load(f)
        comp_model.model.to(device)
        comp_model.model.eval()

    def smi_to_vec(smi):
        output = comp_model.transform(smi, device)
        return output[1].cpu().detach().numpy().reshape(-1)

    comp_feat = {}
    comp_data = data[["cid", "smi"]].drop_duplicates(subset=["cid"])
    for _, row in tqdm(comp_data.iterrows()):
        cid, smi = row[0], row[1]
        comp_feat[cid] = smi_to_vec(smi)

    return comp_feat


def cal_prot_feat(data: pd.DataFrame) -> dict:
    """
    Calculate the protein features using the protein pre-trained model
    """
    # model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    repr_layer = model.num_layers

    def seq_to_vecs(pid, seq, max_length=1022):
        data = [
            (pid, seq[:max_length]),
        ]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])
        token_representations = results["representations"][repr_layer]
        sequence_representations = token_representations[0, 1:].mean(0)
        return sequence_representations.cpu().detach().numpy().reshape(-1)

    prot_feat = {}
    prot_data = data[["pid", "seq"]].drop_duplicates(subset=["pid"])
    for _, row in tqdm(prot_data.iterrows()):
        pid, seq = row[0], row[1]
        prot_feat[pid] = seq_to_vecs(pid, seq)

    return prot_feat


def extract_dti() -> None:
    for dataset in ["drugbank_case"]:
        data_path = "./dataset/" + dataset + "/"
        save_path = data_path + "features/"
        os.makedirs(save_path, exist_ok=True)

        drug_smi = pd.read_csv(data_path + "smile.csv")[['SMILES_index','SMILES']]
        tar_seq = pd.read_csv(data_path + "protein.csv")[['Protein_index','Protein']]

        for i in range(len(tar_seq)):

            seq = tar_seq.loc[i, "Protein"]

            seq = str(seq).upper()
            seq = remove_forbidden_letters(seq)

            if len(seq) > max_length:
                seq = seq[:max_length]

            tar_seq.loc[i, "Protein"] = seq

        drug_smi.columns = ["cid", "smi"]
        tar_seq.columns = ["pid", "seq"]


        print(f"Extracting compound features for {dataset} dataset ...")
        comp_feat = cal_comp_feat(drug_smi, bermol_model_path)
        with open(save_path + "compound_features.pkl", "wb") as f:
            pickle.dump(comp_feat, f)

        print(f"Extracting protein features for {dataset} dataset ...")
        prot_feat = cal_prot_feat(tar_seq)
        with open(save_path + "protein_features.pkl", "wb") as f:
            pickle.dump(prot_feat, f)

def extract():
    for dataset in ["BIOSNAP","BindingDB","davis"]:
        data_path = "./dataset/" + dataset + "/"
        save_path = f"{dataset}/feature/"
        os.makedirs(save_path, exist_ok=True)

        drug_smi = pd.read_csv(data_path + "smile.csv")
        drug_smi.columns = ["cid", "smi"]

        print(f"Extracting compound features for {dataset} dataset ...")
        comp_feat = cal_comp_feat(drug_smi, bermol_model_path)
        for cid, feat_vec in tqdm(comp_feat.items()):
            np.save(os.path.join(save_path, f"smile_{cid}.npy"), feat_vec)
    
    print(f"Features saved to {save_path}")


if __name__ == "__main__":

    bermol_model_path = "/data/lixinchi/li/DTIAM-main/models/BerMolModel_base.pkl"
    extract_dti()

