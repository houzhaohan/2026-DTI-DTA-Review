#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch run single_docking.py using protein-ligand pairs.

Input CSV example:

protein_index,smile_index
378,35
1282,2387
118,96
...

Expected directories:

receptor_pdbqt/
    0.pdbqt
    1.pdbqt
    ...

ligands_pdbqt/
    0.pdbqt
    1.pdbqt
    ...

pocket/
    0/
        0.pdb_predictions.csv
    1/
        1.pdb_predictions.csv
    ...

Output:

batch_docking_results/

    pairs/
        protein_378_ligand_35/
            protein_378_ligand_35.pdbqt
            summary.csv
            docking.log

        protein_1282_ligand_2387/
            ...

    docking_results.csv
    failed_pairs.csv

Usage:

python batch_docking.py \
    --pairs biosnap_index.csv \
    --single-script single_docking.py

"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Read pair file
# ============================================================

def read_pairs(csv_file):

    pairs = []

    with open(
        csv_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError("Pair CSV is empty.")

        # 去除可能存在的空格
        fieldnames = [
            x.strip()
            for x in reader.fieldnames
        ]

        if "protein_index" not in fieldnames:
            raise RuntimeError(
                f"Missing column: protein_index\n"
                f"Available columns: {fieldnames}"
            )

        if "smile_index" not in fieldnames:
            raise RuntimeError(
                f"Missing column: smile_index\n"
                f"Available columns: {fieldnames}"
            )

        for row_number, raw_row in enumerate(
            reader,
            start=2
        ):

            row = {
                str(k).strip(): str(v).strip()
                for k, v in raw_row.items()
                if k is not None
            }

            protein_id = row.get("protein_index", "")
            ligand_id = row.get("smile_index", "")

            if protein_id == "" or ligand_id == "":
                print(
                    f"[WARNING] Skip invalid row {row_number}: "
                    f"{raw_row}"
                )
                continue

            pairs.append(
                (protein_id, ligand_id)
            )

    return pairs


# ============================================================
# Read result produced by single_docking.py
# ============================================================

def read_single_summary(summary_file):

    summary_file = Path(summary_file)

    if not summary_file.exists():
        return None

    with open(
        summary_file,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    if not rows:
        return None

    # 每个pair目录只运行一对，所以取最后一行
    return rows[-1]


# ============================================================
# Run one pair
# ============================================================

def run_one_pair(
    protein_id,
    ligand_id,
    single_script,
    python_exe,
    protein_dir,
    ligand_dir,
    pocket_dir,
    pair_root,
    pocket_rank,
    exhaustiveness,
    skip_existing=True,
):

    pair_name = (
        f"protein_{protein_id}_ligand_{ligand_id}"
    )

    pair_dir = (
        Path(pair_root)
        / pair_name
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_file = (
        pair_dir
        / "summary.csv"
    )

    log_file = (
        pair_dir
        / "docking.log"
    )

    # ========================================================
    # Resume support
    # ========================================================

    if skip_existing and summary_file.exists():

        old_result = read_single_summary(
            summary_file
        )

        if (
            old_result is not None
            and old_result.get("score", "") not in ("", "None", "NA")
        ):

            return {
                "protein_index": protein_id,
                "smile_index": ligand_id,
                "status": "SKIPPED_EXISTING",
                "score": old_result.get("score", ""),
                "pose": old_result.get("pose", ""),
                "result_dir": str(pair_dir),
                "error": "",
            }

    # ========================================================
    # Build command
    # ========================================================

    cmd = [
        str(python_exe),
        str(single_script),

        "--protein_id",
        str(protein_id),

        "--ligand_id",
        str(ligand_id),

        "--protein_dir",
        str(protein_dir),

        "--ligand_dir",
        str(ligand_dir),

        "--pocket_dir",
        str(pocket_dir),

        "--pocket_rank",
        str(pocket_rank),

        "--exhaustiveness",
        str(exhaustiveness),

        "--outdir",
        str(pair_dir),
    ]

    # ========================================================
    # Run single docking
    # ========================================================

    try:

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # 保存完整日志
        with open(
            log_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("[COMMAND]\n")
            f.write(" ".join(cmd))
            f.write("\n\n")

            f.write("[OUTPUT]\n")
            f.write(result.stdout or "")

        # ----------------------------------------------------
        # Vina / single script failed
        # ----------------------------------------------------

        if result.returncode != 0:

            error_tail = (
                result.stdout[-3000:]
                if result.stdout
                else
                "single_docking.py returned non-zero status"
            )

            return {
                "protein_index": protein_id,
                "smile_index": ligand_id,
                "status": "FAILED",
                "score": "",
                "pose": "",
                "result_dir": str(pair_dir),
                "error": error_tail.replace("\n", " | "),
            }

        # ----------------------------------------------------
        # Read single docking result
        # ----------------------------------------------------

        single_result = read_single_summary(
            summary_file
        )

        if single_result is None:

            return {
                "protein_index": protein_id,
                "smile_index": ligand_id,
                "status": "FAILED",
                "score": "",
                "pose": "",
                "result_dir": str(pair_dir),
                "error": "Docking finished but summary.csv was not generated.",
            }

        score = single_result.get(
            "score",
            ""
        )

        pose = single_result.get(
            "pose",
            ""
        )

        if score in ("", "None", "NA"):

            return {
                "protein_index": protein_id,
                "smile_index": ligand_id,
                "status": "FAILED",
                "score": "",
                "pose": pose,
                "result_dir": str(pair_dir),
                "error": "Docking finished but Vina score could not be parsed.",
            }

        return {
            "protein_index": protein_id,
            "smile_index": ligand_id,
            "status": "SUCCESS",
            "score": score,
            "pose": pose,
            "result_dir": str(pair_dir),
            "error": "",
        }

    except Exception as e:

        return {
            "protein_index": protein_id,
            "smile_index": ligand_id,
            "status": "FAILED",
            "score": "",
            "pose": "",
            "result_dir": str(pair_dir),
            "error": str(e),
        }


# ============================================================
# Save global result
# ============================================================

def write_results(results, output_file):

    fields = [
        "protein_index",
        "smile_index",
        "status",
        "score",
        "pose",
        "result_dir",
        "error",
    ]

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for r in results:
            writer.writerow(r)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Batch protein-ligand docking using single_docking.py"
        )
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help=(
            "CSV containing protein_index and smile_index"
        )
    )

    parser.add_argument(
        "--single-script",
        default="single_docking.py",
        help="Path to single_docking.py"
    )

    parser.add_argument(
        "--protein_dir",
        default="receptor_pdbqt",
        help="Directory containing receptor PDBQT files"
    )

    parser.add_argument(
        "--ligand_dir",
        default="ligands_pdbqt",
        help="Directory containing ligand PDBQT files"
    )

    parser.add_argument(
        "--pocket_dir",
        default="pocket",
        help="P2Rank results directory"
    )

    parser.add_argument(
        "--outdir",
        default="batch_docking_results",
        help="Main output directory"
    )

    parser.add_argument(
        "--pocket_rank",
        type=int,
        default=1,
        help="P2Rank pocket rank used for docking"
    )

    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=16,
        help="Vina exhaustiveness"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of docking jobs executed simultaneously. "
            "Default=1. Be careful with CPU oversubscription."
        )
    )

    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Run again even if a successful result already exists"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start pair index, default 0"
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End pair index (exclusive)"
    )

    args = parser.parse_args()

    # ========================================================
    # Basic validation
    # ========================================================

    pair_file = Path(args.pairs)

    if not pair_file.exists():
        raise FileNotFoundError(
            f"Pair CSV not found: {pair_file}"
        )

    single_script = Path(
        args.single_script
    )

    if not single_script.exists():
        raise FileNotFoundError(
            f"single_docking.py not found: {single_script}"
        )

    protein_dir = Path(
        args.protein_dir
    )

    ligand_dir = Path(
        args.ligand_dir
    )

    pocket_dir = Path(
        args.pocket_dir
    )

    for path, description in [
        (protein_dir, "protein directory"),
        (ligand_dir, "ligand directory"),
        (pocket_dir, "pocket directory"),
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    pair_root = (
        outdir
        / "pairs"
    )

    pair_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Read pairs
    # ========================================================

    pairs = read_pairs(
        pair_file
    )

    total_original = len(pairs)

    pairs = pairs[
        args.start:args.end
    ]

    total = len(pairs)

    print("=" * 70)
    print(
        f"Pairs in CSV      : {total_original}"
    )
    print(
        f"Pairs to process  : {total}"
    )
    print(
        f"Workers           : {args.workers}"
    )
    print(
        f"Pocket rank       : {args.pocket_rank}"
    )
    print(
        f"Exhaustiveness    : {args.exhaustiveness}"
    )
    print("=" * 70)

    if total == 0:
        print("No pairs to process.")
        return

    # ========================================================
    # Run
    # ========================================================

    results = []

    completed = 0
    success = 0
    failed = 0
    skipped = 0

    python_exe = sys.executable

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {}

        for protein_id, ligand_id in pairs:

            future = executor.submit(
                run_one_pair,

                protein_id,
                ligand_id,

                single_script,
                python_exe,

                protein_dir,
                ligand_dir,
                pocket_dir,

                pair_root,

                args.pocket_rank,
                args.exhaustiveness,

                not args.no_skip,
            )

            futures[future] = (
                protein_id,
                ligand_id
            )

        for future in as_completed(
            futures
        ):

            protein_id, ligand_id = (
                futures[future]
            )

            try:
                r = future.result()

            except Exception as e:

                r = {
                    "protein_index": protein_id,
                    "smile_index": ligand_id,
                    "status": "FAILED",
                    "score": "",
                    "pose": "",
                    "result_dir": "",
                    "error": str(e),
                }

            results.append(r)

            completed += 1

            if r["status"] == "SUCCESS":
                success += 1

            elif r["status"] == "SKIPPED_EXISTING":
                skipped += 1

            else:
                failed += 1

            print(
                f"[{completed}/{total}] "
                f"P={protein_id} "
                f"L={ligand_id} "
                f"{r['status']} "
                f"score={r['score']}"
            )

            # ------------------------------------------------
            # 每完成一个就更新总表
            # 防止程序中途停止时丢失结果
            # ------------------------------------------------

            write_results(
                results,
                outdir / "docking_results.csv"
            )

    # ========================================================
    # Sort results back into input order
    # ========================================================

    pair_order = {
        (str(p), str(l)): i
        for i, (p, l) in enumerate(pairs)
    }

    results.sort(
        key=lambda x: pair_order.get(
            (
                str(x["protein_index"]),
                str(x["smile_index"])
            ),
            10**12
        )
    )

    final_file = (
        outdir
        / "docking_results.csv"
    )

    write_results(
        results,
        final_file
    )

    # ========================================================
    # Failed only
    # ========================================================

    failed_results = [
        r
        for r in results
        if r["status"] == "FAILED"
    ]

    if failed_results:

        write_results(
            failed_results,
            outdir / "failed_pairs.csv"
        )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"Total   : {total}"
    )

    print(
        f"Success : {success}"
    )

    print(
        f"Skipped : {skipped}"
    )

    print(
        f"Failed  : {failed}"
    )

    print(
        f"Results : {final_file}"
    )

    if failed_results:

        print(
            f"Failures: {outdir / 'failed_pairs.csv'}"
        )


if __name__ == "__main__":
    main()