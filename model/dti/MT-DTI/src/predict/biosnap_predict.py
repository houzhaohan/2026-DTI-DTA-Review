"""
BioSNAP Prediction Script
Loads a trained model and runs prediction on test/dev set.
"""
import tensorflow as tf
from src.finetune.dti_model import MbertPcnnModel
import argparse
import os
import re
import pandas as pd
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='BioSNAP Prediction Parser')
parser.add_argument('--gpu_num', default="0", type=str)
parser.add_argument('--model_version', default="11", choices=["1", "11", "14"], type=str)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset_name', type=str, default="full_data",
                    choices=["full_data", "unseen_drug", "unseen_protein"])
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--split', type=str, default="test", choices=["dev", "test"],
                    help='Which split to predict on')
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='Directory containing model checkpoint (e.g., best_auc dir)')
parser.add_argument('--bert_config_file', type=str, default=None)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--k1', type=int, default=12)
parser.add_argument('--k2', type=int, default=12)
parser.add_argument('--k3', type=int, default=12)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if args.bert_config_file is None:
    args.bert_config_file = os.path.join(PROJECT_ROOT, 'config', 'm_bert_base_config.json')

if args.data_root is None:
    candidate_paths = [
        os.path.join(PROJECT_ROOT, 'biosnap'),
        r'd:\2_temporary\biosnap_data',
    ]
    args.data_root = None
    for p in candidate_paths:
        if os.path.isdir(p):
            args.data_root = p
            break

dataset_dir = os.path.join(args.data_root, args.dataset_name)
tfrecord_dir = os.path.join(dataset_dir, 'tfrecord')

split_map = {'dev': 'dev', 'test': 'tst'}
tfrecord_name = split_map[args.split]
input_tfrecord = os.path.join(tfrecord_dir, '%s.tfrecord' % tfrecord_name)

# Find checkpoint
if args.checkpoint_dir is None:
    # Auto-find best_auc
    model_dir = os.path.join(dataset_dir, 'mbert_cnn_v%s_lr%.4f_k%d_k%d_k%d' %
                             (args.model_version, args.learning_rate, args.k1, args.k2, args.k3))
    args.checkpoint_dir = os.path.join(model_dir, 'best_auc')

checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint')
if not os.path.isfile(checkpoint_file):
    tf.logging.error("No checkpoint found in %s" % args.checkpoint_dir)
    exit(1)

with open(checkpoint_file, 'rt') as handle:
    line = handle.readline()
    best_model_prefix = re.findall(r'"(.*?)"', line)[0]
checkpoint_path = os.path.join(args.checkpoint_dir, best_model_prefix)

tf.logging.info("Dataset: %s, Split: %s" % (args.dataset_name, args.split))
tf.logging.info("Checkpoint: %s" % checkpoint_path)

# Build model
model = MbertPcnnModel(
    batch_size=args.batch_size,
    dev_batch_size=args.batch_size,
    max_molecule_length=100,
    max_protein_length=1000,
    bert_config_file=args.bert_config_file,
    init_checkpoint=None,  # We load directly from checkpoint_path
    learning_rate=args.learning_rate,
    num_train_steps=1000,
    num_warmup_steps=100,
    use_tpu=False,
    kernel_size1=args.k1,
    kernel_size2=args.k2,
    kernel_size3=args.k3,
    task_type='classification'
)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

run_config = tf.contrib.tpu.RunConfig(
    session_config=config,
    model_dir=args.checkpoint_dir,
    save_checkpoints_steps=1000000,  # no saving during predict
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=1000000,
        num_shards=1,
        per_host_input_for_training=True))

model_fn = eval("model.model_fn_v%s" % args.model_version)
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=args.batch_size,
    eval_batch_size=args.batch_size)

input_fn = model.input_fn_builder([input_tfrecord], is_training=False)

# Evaluate first
# With drop_remainder=False for eval, use ceil to cover all samples
num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(input_tfrecord))
eval_steps = (num_examples + args.batch_size - 1) // args.batch_size  # ceil division
eval_results = estimator.evaluate(input_fn=input_fn, checkpoint_path=checkpoint_path, steps=eval_steps)
tf.logging.info("Evaluation results: %s" % eval_results)

# Predict
results = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)

# Read original CSV for drug/protein IDs
csv_name = 'val.csv' if args.split == 'dev' else 'test.csv'
csv_path = os.path.join(dataset_dir, csv_name)
df = pd.read_csv(csv_path)

# Output: test_r.csv (or val_r.csv) — adds a single "Prediction" column
out_name = 'val_r.csv' if args.split == 'dev' else 'test_r.csv'
output_csv = os.path.join(dataset_dir, out_name)
predictions_list = []

for idx, result in enumerate(results):
    y_hat = result['predictions'][0]
    predictions_list.append(y_hat)
    if idx % 1000 == 0:
        tf.logging.info("Predicted %d / %d" % (idx, num_examples))

df['Prediction'] = predictions_list
df.to_csv(output_csv, index=False)
tf.logging.info("Predictions saved to: %s" % output_csv)

# Also compute metrics against ground truth for reference
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

y_true = df['Label'].values.astype(float)
y_pred_proba = np.array(predictions_list)
y_pred_binary = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_true, y_pred_proba)
acc = accuracy_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
prec = precision_score(y_true, y_pred_binary)
rec = recall_score(y_true, y_pred_binary)

tf.logging.info("=" * 50)
tf.logging.info("Results for %s (%s set):" % (args.dataset_name, args.split))
tf.logging.info("  AUC:       %.4f" % auc)
tf.logging.info("  Accuracy:  %.4f" % acc)
tf.logging.info("  F1:        %.4f" % f1)
tf.logging.info("  Precision: %.4f" % prec)
tf.logging.info("  Recall:    %.4f" % rec)
tf.logging.info("=" * 50)

# Save metrics
metrics_path = os.path.join(dataset_dir, 'metrics_%s.txt' % args.split)
with open(metrics_path, 'w') as f:
    f.write("Dataset: %s\n" % args.dataset_name)
    f.write("Split: %s\n" % args.split)
    f.write("AUC: %.4f\n" % auc)
    f.write("Accuracy: %.4f\n" % acc)
    f.write("F1: %.4f\n" % f1)
    f.write("Precision: %.4f\n" % prec)
    f.write("Recall: %.4f\n" % rec)
