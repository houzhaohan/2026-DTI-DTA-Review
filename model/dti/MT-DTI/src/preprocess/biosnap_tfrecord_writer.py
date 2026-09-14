"""
BioSNAP TFRecord writer
Reads CSV files (train.csv, val.csv, test.csv) from BioSNAP dataset directories
and converts them to TFRecord format for training.

CSV columns expected: DrugBank ID, Gene, Label, SMILES, Target Sequence
"""
import os
import sys
import random
import collections
import numpy as np
import tensorflow as tf
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.utils import DTITokenizer


tf.logging.set_verbosity(tf.logging.INFO)

# Configuration
MAX_MOLECULE_LENGTH = 100
MAX_PROTEIN_LENGTH = 1000
MAX_NUM_TOKENS_MOL = MAX_MOLECULE_LENGTH - 1  # reserve 1 for [CLS]
MAX_NUM_TOKENS_PRO = MAX_PROTEIN_LENGTH - 1
RANDOM_SEED = 12345

# Paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
VOCAB_SMILES = os.path.join(PROJECT_ROOT, 'config', 'vocab_smiles.txt')
VOCAB_FASTA = os.path.join(PROJECT_ROOT, 'config', 'vocab_fasta.txt')

# BioSNAP datasets to process
BIOSNAP_DATASETS = ['full_data', 'unseen_drug', 'unseen_protein']


def truncate_seq_pair(tokens, max_num_tokens, rng):
    """Truncate tokens to fit within max_num_tokens."""
    while len(tokens) > max_num_tokens:
        if rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


def tokenize_sequence(seq, tokenizer, max_num_tokens, rng):
    """Tokenize a single sequence and pad/truncate."""
    tokens = tokenizer.tokenize(seq)
    truncate_seq_pair(tokens, max_num_tokens, rng)
    tokens.insert(0, "[CLS]")

    input_mask = [1] * len(tokens)
    while len(tokens) < max_num_tokens + 1:
        tokens.append("[PAD]")
        input_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return input_ids, input_mask


def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def create_example(xd, xdm, xt, xtm, y):
    """Create a tf.train.Example from tokenized inputs."""
    features = collections.OrderedDict()
    features["xd"] = create_int_feature(xd)
    features["xdm"] = create_int_feature(xdm)
    features["xt"] = create_int_feature(xt)
    features["xtm"] = create_int_feature(xtm)
    features["y"] = create_float_feature([y])
    return tf.train.Example(features=tf.train.Features(feature=features))


def process_csv(csv_path, molecule_tokenizer, protein_tokenizer, rng):
    """
    Read a CSV file and return tokenized arrays.
    Returns: xd_list, xdm_list, xt_list, xtm_list, y_list
    """
    tf.logging.info("Reading: %s" % csv_path)
    df = pd.read_csv(csv_path)

    xd_list, xdm_list = [], []
    xt_list, xtm_list = [], []
    y_list = []

    for idx, row in df.iterrows():
        smiles = str(row['SMILES'])
        sequence = str(row['Target Sequence'])
        label = float(row['Label'])

        xd, xdm = tokenize_sequence(smiles, molecule_tokenizer, MAX_NUM_TOKENS_MOL, rng)
        xt, xtm = tokenize_sequence(sequence, protein_tokenizer, MAX_NUM_TOKENS_PRO, rng)

        xd_list.append(xd)
        xdm_list.append(xdm)
        xt_list.append(xt)
        xtm_list.append(xtm)
        y_list.append(label)

        if idx % 1000 == 0:
            tf.logging.info("  Processing... [%d/%d]" % (idx, len(df)))

    tf.logging.info("  Done. Total: %d samples" % len(df))
    return (np.array(xd_list), np.array(xdm_list),
            np.array(xt_list), np.array(xtm_list),
            np.array(y_list))


def write_tfrecord(xd, xdm, xt, xtm, y, output_path):
    """Write arrays to TFRecord file."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = tf.python_io.TFRecordWriter(output_path)
    for i in range(len(xd)):
        example = create_example(xd[i], xdm[i], xt[i], xtm[i], y[i])
        writer.write(example.SerializeToString())
    writer.close()
    tf.logging.info("Wrote %d examples to %s" % (len(xd), output_path))


def process_dataset(dataset_dir, molecule_tokenizer, protein_tokenizer):
    """Process all splits (train, val, test) for one BioSNAP dataset."""
    rng = random.Random(RANDOM_SEED)

    split_map = {
        'train': 'train.tfrecord',
        'val': 'dev.tfrecord',
        'test': 'test.tfrecord'
    }

    for csv_name, tfrecord_name in [('train.csv', 'trn'), ('val.csv', 'dev'), ('test.csv', 'tst')]:
        csv_path = os.path.join(dataset_dir, csv_name)
        if not os.path.isfile(csv_path):
            tf.logging.warning("Missing: %s" % csv_path)
            continue

        tf.logging.info("=" * 60)
        tf.logging.info("Processing %s" % csv_name)
        tf.logging.info("=" * 60)

        xd, xdm, xt, xtm, y = process_csv(csv_path, molecule_tokenizer, protein_tokenizer, rng)

        # Output path: dataset_dir/tfrecord/{trn,dev,tst}.tfrecord
        output_path = os.path.join(dataset_dir, 'tfrecord', '%s.tfrecord' % tfrecord_name)
        write_tfrecord(xd, xdm, xt, xtm, y, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BioSNAP TFRecord Converter')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory containing BioSNAP dataset folders')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all'] + BIOSNAP_DATASETS,
                        help='Which dataset to process')
    args = parser.parse_args()

    # Determine data root
    if args.data_root:
        data_root = args.data_root
    else:
        # Try both common locations
        candidate_paths = [
            os.path.join(PROJECT_ROOT, 'biosnap'),
            r'd:\2_temporary\biosnap_data',
        ]
        data_root = None
        for p in candidate_paths:
            if os.path.isdir(p):
                data_root = p
                break
        if data_root is None:
            raise FileNotFoundError(
                "Cannot find BioSNAP data directory. "
                "Searched: %s" % str(candidate_paths))

    tf.logging.info("Using data root: %s" % data_root)

    # Initialize tokenizers
    molecule_tokenizer = DTITokenizer(VOCAB_SMILES)
    protein_tokenizer = DTITokenizer(VOCAB_FASTA)

    # Determine which datasets to process
    datasets_to_process = BIOSNAP_DATASETS if args.dataset == 'all' else [args.dataset]

    for ds_name in datasets_to_process:
        ds_dir = os.path.join(data_root, ds_name)
        if not os.path.isdir(ds_dir):
            tf.logging.warning("Skipping %s: directory not found at %s" % (ds_name, ds_dir))
            continue
        tf.logging.info("\n" + "#" * 60)
        tf.logging.info("# Dataset: %s" % ds_name)
        tf.logging.info("#" * 60)
        process_dataset(ds_dir, molecule_tokenizer, protein_tokenizer)

    tf.logging.info("\nAll BioSNAP datasets processed!")
