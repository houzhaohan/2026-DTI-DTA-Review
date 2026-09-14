"""
BioSNAP Fine-tuning Script
Supports three BioSNAP datasets: full_data, unseen_drug, unseen_protein
Binary classification task (Label: 0/1)
"""
import tensorflow as tf
from src.finetune.dti_model import MbertPcnnModel
from src.finetune.dti_model import load_global_step_from_checkpoint_dir
import argparse
import os
import time
import shutil
import glob
import re
import pandas as pd


tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='BioSNAP Fine-tuning Parser')
parser.add_argument('--gpu_num', default="0", type=str)
parser.add_argument('--model_version', default="11", choices=["1", "11", "14"], type=str,
                    help='Model version (1, 11, or 14)')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset_name', type=str, default="full_data",
                    choices=["full_data", "unseen_drug", "unseen_protein"],
                    help='BioSNAP dataset to use')
parser.add_argument('--data_root', type=str, default=None,
                    help='Root directory containing BioSNAP dataset folders')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--bert_config_file', type=str, default=None)
parser.add_argument('--init_checkpoint', type=str, default=None,
                    help='Pre-trained checkpoint path (optional)')
parser.add_argument('--k1', type=int, default=12, help='Protein CNN kernel_size1')
parser.add_argument('--k2', type=int, default=12, help='Protein CNN kernel_size2')
parser.add_argument('--k3', type=int, default=12, help='Protein CNN kernel_size3')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--dev_eval_steps', type=int, default=50,
                    help='Evaluate on dev set every N checkpoint steps')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

# ---- Path Configuration ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default paths
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
    if args.data_root is None:
        raise FileNotFoundError("Cannot find BioSNAP data directory")

# Dataset-specific paths
dataset_dir = os.path.join(args.data_root, args.dataset_name)
tfrecord_dir = os.path.join(dataset_dir, 'tfrecord')

i_trn = os.path.join(tfrecord_dir, 'trn.tfrecord')
i_dev = os.path.join(tfrecord_dir, 'dev.tfrecord')
i_tst = os.path.join(tfrecord_dir, 'tst.tfrecord')

output_dir = os.path.join(dataset_dir, 'mbert_cnn_v%s_lr%.4f_k%d_k%d_k%d' %
                          (args.model_version, args.learning_rate, args.k1, args.k2, args.k3))
best_model_dir_auc = os.path.join(output_dir, 'best_auc')

os.makedirs(output_dir, exist_ok=True)

# ---- Compute dataset sizes from CSV ----
def count_csv_rows(csv_path):
    if not os.path.isfile(csv_path):
        return 0
    df = pd.read_csv(csv_path)
    return len(df)

train_csv = os.path.join(dataset_dir, 'train.csv')
dev_csv = os.path.join(dataset_dir, 'val.csv')
test_csv = os.path.join(dataset_dir, 'test.csv')

num_trn_example = count_csv_rows(train_csv)
num_dev_example = count_csv_rows(dev_csv)
num_tst_example = count_csv_rows(test_csv)

tf.logging.info("Dataset: %s" % args.dataset_name)
tf.logging.info("  Train: %d examples" % num_trn_example)
tf.logging.info("  Val:   %d examples" % num_dev_example)
tf.logging.info("  Test:  %d examples" % num_tst_example)

# ---- Training Hyperparameters ----
batch_size = args.batch_size
num_train_steps = int(num_trn_example * 1.0 / batch_size * args.num_epochs)
num_warmup_steps = num_train_steps // 10
dev_batch_size = batch_size
save_checkpoints_steps = max(10, num_trn_example // batch_size)  # save roughly every epoch
# With drop_remainder=False for eval, we need ceil to cover all samples
dev_steps = (num_dev_example + dev_batch_size - 1) // dev_batch_size  # ceil division
tst_steps = (num_tst_example + dev_batch_size - 1) // dev_batch_size

tf.logging.info("Training config:")
tf.logging.info("  num_train_steps: %d" % num_train_steps)
tf.logging.info("  num_warmup_steps: %d" % num_warmup_steps)
tf.logging.info("  save_checkpoints_steps: %d" % save_checkpoints_steps)
tf.logging.info("  dev_steps: %d" % dev_steps)
tf.logging.info("  num_epochs: %d" % args.num_epochs)
tf.logging.info("  approx epochs per save: %.1f" % (save_checkpoints_steps * batch_size / num_trn_example))


def log_scores(current_step, best_step, best_dev_auc, best_dev_loss,
               test_auc, test_loss, prefix=''):
    line1 = '==== [%s-V%s-lr(%.4f)-k(%d,%d,%d)] step(%d/%d) ====' % \
            (args.dataset_name, args.model_version, args.learning_rate,
             args.k1, args.k2, args.k3, current_step, num_train_steps)
    line2 = '  [dev] best_auc@%d: auc=%.4f loss=%.4f' % (best_step, best_dev_auc, best_dev_loss)
    line3 = '  [tst]  auc=%.4f loss=%.4f' % (test_auc, test_loss)
    tf.logging.info(line1)
    tf.logging.info(line2)
    tf.logging.info(line3)

    with open(os.path.join(output_dir, '%s_status.txt' % prefix), 'a') as handle:
        handle.write(line1 + '\n' + line2 + '\n' + line3 + '\n\n')


def restore_best(best_model_dir_auc, estimator, input_fn_dev, input_fn_tst):
    """Restore the best saved model and re-evaluate."""
    best_step = 0
    best_dev_auc = 0.0
    best_dev_loss = 10.0
    best_test_auc = 0.0
    best_test_loss = 10.0

    checkpoint_file = os.path.join(best_model_dir_auc, 'checkpoint')
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'rt') as handle:
            line = handle.readline()
            best_model_prefix = re.findall(r'"(.*?)"', line)[0]

        checkpoint_path = os.path.join(best_model_dir_auc, best_model_prefix)
        dev_results = estimator.evaluate(input_fn=input_fn_dev,
                                         checkpoint_path=checkpoint_path,
                                         steps=dev_steps)
        tst_results = estimator.evaluate(input_fn=input_fn_tst,
                                         checkpoint_path=checkpoint_path,
                                         steps=tst_steps)

        best_dev_auc = dev_results.get('auc', 0.0)
        best_dev_loss = dev_results.get('loss', dev_results.get('mse', 0.0))
        best_step = dev_results.get('global_step', 0)
        best_test_auc = tst_results.get('auc', 0.0)
        best_test_loss = tst_results.get('loss', tst_results.get('mse', 0.0))

        log_scores(best_step, best_step, best_dev_auc, best_dev_loss,
                   best_test_auc, best_test_loss, prefix='Restored')

    return best_step, best_dev_auc, best_dev_loss, best_test_auc, best_test_loss


def check_and_save_best(current_step, best_dev_auc, best_dev_loss,
                        best_test_auc, best_test_loss, eval_results,
                        estimator, input_fn_tst, tst_steps):
    """Check if current model improves AUC on dev set; if so, save it."""
    current_auc = eval_results.get('auc', 0.0)
    current_loss = eval_results.get('loss', eval_results.get('mse', 0.0))

    if best_dev_auc < current_auc:
        tf.logging.info('AUC improved! from %.4f to %.4f' % (best_dev_auc, current_auc))
        best_dev_auc = current_auc
        best_dev_loss = current_loss

        # Evaluate on test set with current model
        tst_results = estimator.evaluate(input_fn=input_fn_tst, steps=tst_steps)
        best_test_auc = tst_results.get('auc', 0.0)
        best_test_loss = tst_results.get('loss', tst_results.get('mse', 0.0))

        # Save checkpoint
        if not os.path.exists(best_model_dir_auc):
            os.makedirs(best_model_dir_auc)

        for file in glob.glob(r'%s/model.ckpt-%d.*' % (output_dir, current_step)):
            shutil.copy(file, best_model_dir_auc)

        step_list = []
        for file in glob.glob(r'%s/model.ckpt-*.meta' % best_model_dir_auc):
            step_list.append(int(re.findall(r'\d+', file)[-1]))

        n_keep = 5
        if len(step_list) > n_keep:
            for del_step in sorted(step_list)[:-n_keep]:
                for f in glob.glob(r'%s/model.ckpt-%d.*' % (best_model_dir_auc, del_step)):
                    os.remove(f)

        step_list_sorted = sorted(step_list)
        with open(os.path.join(best_model_dir_auc, 'checkpoint'), 'wt') as handle:
            handle.write('model_checkpoint_path: "model.ckpt-%d"\n' % step_list_sorted[-1])
            for s in step_list_sorted[-n_keep:]:
                handle.write('all_model_checkpoint_paths: "model.ckpt-%d"\n' % s)

    return best_dev_auc, best_dev_loss, best_test_auc, best_test_loss


def main(argv):
    del argv

    # Verify TFRecord files exist
    for label, path in [('train', i_trn), ('dev', i_dev), ('test', i_tst)]:
        if not os.path.isfile(path):
            tf.logging.error("TFRecord not found: %s" % path)
            tf.logging.error("Run biosnap_tfrecord_writer.py first!")
            return

    # Initialize model (classification task)
    model = MbertPcnnModel(
        batch_size=batch_size,
        dev_batch_size=dev_batch_size,
        max_molecule_length=100,
        max_protein_length=1000,
        bert_config_file=args.bert_config_file,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
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
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=save_checkpoints_steps,
            num_shards=1,
            per_host_input_for_training=True))

    model_fn = eval("model.model_fn_v%s" % args.model_version)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=dev_batch_size)

    input_fn_trn = model.input_fn_builder([i_trn], is_training=True)
    input_fn_dev = model.input_fn_builder([i_dev], is_training=False)
    input_fn_tst = model.input_fn_builder([i_tst], is_training=False)

    current_step = load_global_step_from_checkpoint_dir(output_dir)
    tf.logging.info('Resuming from step %d / %d' % (current_step, num_train_steps))

    start_timestamp = time.time()

    # Restore best if exists
    best_step, best_dev_auc, best_dev_loss, best_test_auc, best_test_loss = \
        restore_best(best_model_dir_auc, estimator, input_fn_dev, input_fn_tst)

    while current_step < num_train_steps:
        next_checkpoint = min(current_step + save_checkpoints_steps, num_train_steps)
        estimator.train(input_fn=input_fn_trn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        # Evaluate on dev set
        tf.logging.info('Evaluating at step %d' % current_step)
        eval_results = estimator.evaluate(input_fn=input_fn_dev, steps=dev_steps)

        best_dev_auc, best_dev_loss, best_test_auc, best_test_loss = \
            check_and_save_best(current_step, best_dev_auc, best_dev_loss,
                                best_test_auc, best_test_loss, eval_results,
                                estimator, input_fn_tst, tst_steps)

        log_scores(current_step, best_step if best_dev_auc == 0 else current_step,
                   best_dev_auc, best_dev_loss, best_test_auc, best_test_loss,
                   prefix='Current')

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Training finished in %d seconds' % elapsed_time)

    # Final evaluation
    best_step, best_dev_auc, best_dev_loss, best_test_auc, best_test_loss = \
        restore_best(best_model_dir_auc, estimator, input_fn_dev, input_fn_tst)

    log_scores(current_step, best_step, best_dev_auc, best_dev_loss,
               best_test_auc, best_test_loss, prefix='Final')


if __name__ == '__main__':
    tf.app.run(main)
