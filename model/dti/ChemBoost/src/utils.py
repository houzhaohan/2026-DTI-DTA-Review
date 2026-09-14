import argparse
import json
import itertools
import os
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from src.representation_models import SWX, X8mer, Random8mer, SWRandom
from src.representation_models import SW8mer, ProtVec8mer, ProtVecBPE
from src.representation_models import All8mer, SB8mer, SBBPE
from src.representation_models import SB8merDB, SWSB8mer, SWSB8merDB


BIOSNAP_SPLITS = ['biosnap_full', 'biosnap_unseen_drug', 'biosnap_unseen_protein']

def parse_terminal_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Choose dataset',
                        choices=['bdb', 'kiba'] + BIOSNAP_SPLITS)

    model_names = ['sw_x', 'x_8mer', 'sw_random', 'random_8mer',
                   'sw_8mer', 'pv_8mer', 'pv_bpe',
                   'all_8mer', 'sb_8mer', 'sb_bpe',
                   'sb_8mer_db', 'sw_sb_8mer', 'sw_sb_8mer_db']
    parser.add_argument('--model', help='Choose the model to train', choices=model_names)
    parser.add_argument('--savefile', help='Set name for saving')
    args = parser.parse_args()
    return args.dataset, args.model, args.savefile


def get_repr_model(dataset, model, configs, ligands=None, proteins=None):
    """Build a representation model.

    For standard datasets (bdb, kiba), ligands and proteins are loaded from
    configs paths. For biosnap, pass them in explicitly as dicts.
    """
    if ligands is None:
        with open(configs[f'{dataset}_ligands']) as f:
            ligands = json.load(f)
    if proteins is None:
        with open(configs[f'{dataset}_proteins']) as f:
            proteins = json.load(f)

    sb_threshold_key = f'{dataset}_sb_threshold'
    sb_threshold = configs.get(sb_threshold_key, configs.get('biosnap_sb_threshold', 0.5))

    # Models requiring protein similarity vectors — not available for biosnap
    no_prot_sim_models = {'sw_x', 'sw_random', 'sw_8mer', 'sw_sb_8mer', 'sw_sb_8mer_db'}
    is_biosnap = dataset.startswith('biosnap')
    if is_biosnap and model in no_prot_sim_models:
        raise ValueError(
            f"Model '{model}' requires pre-computed protein similarity vectors "
            f"which are not available for BIOSNAP. Choose one of: "
            f"x_8mer, random_8mer, pv_8mer, pv_bpe, all_8mer, sb_8mer, sb_bpe."
        )
    sb_bindingdb_models = {'sb_8mer_db'}
    if is_biosnap and model in sb_bindingdb_models:
        raise ValueError(
            f"Model '{model}' requires a binding-db file which is not available for BIOSNAP."
        )

    if model == 'sw_x':
        representation_model = SWX(configs[f'{dataset}_prot_sim'])
    elif model == 'x_8mer':
        representation_model = X8mer(ligands, configs['lingo2vec'])
    elif model == 'sw_random':
        representation_model = SWRandom(configs[f'{dataset}_prot_sim'], ligands)
    elif model == 'random_8mer':
        representation_model = Random8mer(proteins, ligands, configs['lingo2vec'])
    elif model == 'sw_8mer':
        representation_model = SW8mer(configs[f'{dataset}_prot_sim'], ligands, configs['lingo2vec'])
    elif model == 'pv_8mer':
        representation_model = ProtVec8mer(proteins, ligands, configs['prot2vec'], configs['lingo2vec'])
    elif model == 'pv_bpe':
        representation_model = ProtVecBPE(proteins, ligands, configs['prot2vec'], configs['bpe2vec'], configs['smiles_bpe'])
    elif model == 'all_8mer':
        representation_model = All8mer(ligands, configs['lingo2vec'])
    elif model == 'sb_8mer':
        representation_model = SB8mer(ligands, configs['lingo2vec'], sb_threshold)
    elif model == 'sb_bpe':
        representation_model = SBBPE(ligands, configs['bpe2vec'], configs['smiles_bpe'], sb_threshold)
    elif model == 'sb_8mer_db':
        representation_model = SB8merDB(ligands, configs['lingo2vec'], sb_threshold, configs['sb_bindingdb_path'])
    elif model == 'sw_sb_8mer':
        representation_model = SWSB8mer(configs[f'{dataset}_prot_sim'], ligands, configs['lingo2vec'], sb_threshold)
    elif model == 'sw_sb_8mer_db':
        representation_model = SWSB8merDB(configs[f'{dataset}_prot_sim'], ligands, configs['lingo2vec'], sb_threshold, configs['sb_bindingdb_path'])

    return representation_model


def load_biosnap_data(dataset, configs):
    """Load BIOSNAP dataset, returning folds, test, ligands, proteins.

    Args:
        dataset: e.g. 'biosnap_full', 'biosnap_unseen_drug', 'biosnap_unseen_protein'
        configs: loaded configs.json dict

    Returns:
        folds: list of 5 DataFrames (KFold splits from train+val)
        test: DataFrame with renamed columns
        ligands: dict {ligand_id: smiles}
        proteins: dict {prot_id: sequence}
        sb_threshold: float from configs
        raw_test_path: str, path to the original test.csv (for test_r.csv output)
    """
    split_map = {
        'biosnap_full': 'full_data',
        'biosnap_unseen_drug': 'unseen_drug',
        'biosnap_unseen_protein': 'unseen_protein',
    }
    split = split_map[dataset]
    biosnap_dir = configs['biosnap_dir']
    raw_test_path = os.path.join(biosnap_dir, split, 'test.csv')

    COL_RENAME = {
        'DrugBank ID': 'ligand_id',
        'Gene': 'prot_id',
        'Label': 'affinity_score',
        'SMILES': 'smiles',
        'Target Sequence': 'prot_sequence',
    }

    train_df = pd.read_csv(os.path.join(biosnap_dir, split, 'train.csv'))
    val_df = pd.read_csv(os.path.join(biosnap_dir, split, 'val.csv'))
    test_df = pd.read_csv(raw_test_path)

    # Rename + keep only needed columns
    KEEP_COLS = ['ligand_id', 'prot_id', 'affinity_score', 'smiles', 'prot_sequence']
    def _prep(df):
        df = df.rename(columns=COL_RENAME)
        # Handle 'Unnamed' columns
        drop_cols = [c for c in df.columns if c.startswith('Unnamed')]
        df = df.drop(columns=drop_cols, errors='ignore')
        # Keep only the cols we need (extra ones are dropped)
        # Use reindex to be safe — if any renamed col is missing, it'll be NaN
        df = df[KEEP_COLS]
        df['ligand_id'] = df['ligand_id'].astype(str)
        df['prot_id'] = df['prot_id'].astype(str)
        return df

    train_df = _prep(train_df)
    val_df = _prep(val_df)
    test_df = _prep(test_df)

    # Combine train + val, then KFold
    train_val = pd.concat([train_df, val_df], ignore_index=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for _, fold_idx in kf.split(train_val):
        # Each fold holds a subset of train_val — runner.py will concat the others
        # to form the training set, and use this one as val
        folds.append(train_val.iloc[fold_idx].reset_index(drop=True))

    # Build ligands & proteins dicts from ALL data (train+val+test), so that
    # unseen-drug / unseen-protein test samples have their SMILES/sequence covered
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ligands = dict(zip(all_df['ligand_id'], all_df['smiles']))
    proteins = dict(zip(all_df['prot_id'], all_df['prot_sequence']))

    sb_threshold = configs['biosnap_sb_threshold']
    return folds, test_df, ligands, proteins, sb_threshold, raw_test_path


def dict_cartesian_product(dct):
    return [dict(zip(dct.keys(), items)) for items in itertools.product(*dct.values())]
