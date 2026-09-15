#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Relaxed parameter sets
# ============================================================

PARAMETER_SETS = [
    {
        "level": 1,
        "point_threshold": 0.30,
        "min_cluster_size": 2,
    },
    {
        "level": 2,
        "point_threshold": 0.25,
        "min_cluster_size": 2,
    },
    {
        "level": 3,
        "point_threshold": 0.20,
        "min_cluster_size": 1,
    },
]


# ============================================================
# Read failed protein IDs
# ============================================================

def read_failed_proteins(report_file, statuses):

    report_file = Path(report_file)

    if not report_file.exists():
        raise FileNotFoundError(
            f"Report file not found: {report_file}"
        )

    proteins = []

    with open(
        report_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"Empty report file: {report_file}"
            )

        fieldnames = [
            x.strip()
            for x in reader.fieldnames
        ]

        if "protein_id" not in fieldnames:
            raise RuntimeError(
                f"Missing column 'protein_id'. "
                f"Available: {fieldnames}"
            )

        if "status" not in fieldnames:
            raise RuntimeError(
                f"Missing column 'status'. "
                f"Available: {fieldnames}"
            )

        for raw_row in reader:

            row = {
                str(k).strip(): str(v).strip()
                for k, v in raw_row.items()
                if k is not None
            }

            protein_id = row.get(
                "protein_id",
                ""
            )

            status = row.get(
                "status",
                ""
            ).upper()

            if (
                protein_id
                and status in statuses
            ):
                proteins.append(
                    protein_id
                )

    # 去重，同时保持原顺序
    proteins = list(
        dict.fromkeys(proteins)
    )

    return proteins


# ============================================================
# Sort protein IDs
# ============================================================

def protein_sort_key(protein_id):

    try:
        return (0, int(protein_id))
    except ValueError:
        return (1, protein_id)


# ============================================================
# Find input PDB
# ============================================================

def find_pdb(pdb_dir, protein_id):

    pdb_dir = Path(pdb_dir)

    # 最常见情况
    expected = (
        pdb_dir
        / f"{protein_id}.pdb"
    )

    if expected.exists():
        return expected

    # 兼容例如 0_model.pdb 等情况
    candidates = sorted(
        pdb_dir.glob(
            f"{protein_id}*.pdb"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple PDB files found for protein "
            f"{protein_id}: {candidates}"
        )

    raise FileNotFoundError(
        f"PDB not found for protein {protein_id}. "
        f"Expected: {expected}"
    )


# ============================================================
# Find prediction files
# ============================================================

def find_prediction_files(output_dir):

    output_dir = Path(
        output_dir
    )

    if not output_dir.exists():
        return []

    return sorted(
        output_dir.glob(
            "*predictions*.csv"
        )
    )


# ============================================================
# Check whether prediction file contains pocket
# ============================================================

def count_pockets(prediction_file):

    prediction_file = Path(
        prediction_file
    )

    if not prediction_file.exists():
        return 0

    try:

        with open(
            prediction_file,
            "r",
            encoding="utf-8-sig"
        ) as f:

            nonempty_lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # 只有header或者完全空
        if len(nonempty_lines) <= 1:
            return 0

        # header之外的数据行数
        return len(nonempty_lines) - 1

    except Exception:
        return 0


# ============================================================
# Check output directory for a valid prediction
# ============================================================

def find_valid_prediction(output_dir):

    prediction_files = (
        find_prediction_files(
            output_dir
        )
    )

    for prediction_file in prediction_files:

        n_pockets = count_pockets(
            prediction_file
        )

        if n_pockets > 0:

            return (
                prediction_file,
                n_pockets
            )

    return None, 0


# ============================================================
# Remove old prediction CSV
# ============================================================

def remove_old_prediction_files(output_dir):

    output_dir = Path(
        output_dir
    )

    if not output_dir.exists():
        return

    for file in output_dir.glob(
        "*predictions*.csv"
    ):

        try:
            file.unlink()
        except Exception as e:
            print(
                f"[WARNING] Cannot remove old prediction: "
                f"{file}: {e}"
            )


# ============================================================
# Run one P2Rank attempt
# ============================================================

def run_p2rank_once(
    prank,
    pdb_file,
    output_dir,
    parameter,
    config="alphafold"
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    level = parameter[
        "level"
    ]

    threshold = parameter[
        "point_threshold"
    ]

    cluster_size = parameter[
        "min_cluster_size"
    ]

    # --------------------------------------------------------
    # 删除旧 predictions，避免把上一次结果误认为新结果
    # --------------------------------------------------------

    remove_old_prediction_files(
        output_dir
    )

    # --------------------------------------------------------
    # Build command
    # --------------------------------------------------------

    cmd = [
        str(prank),
        "predict",

        "-f",
        str(pdb_file),
    ]

    if config:

        cmd.extend([
            "-c",
            str(config)
        ])

    cmd.extend([

        "-pred_point_threshold",
        str(threshold),

        "-pred_min_cluster_size",
        str(cluster_size),

        "-o",
        str(output_dir),
    ])

    log_file = (
        output_dir
        / f"retry_level_{level}.log"
    )

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

            f.write(
                "[COMMAND]\n"
            )

            f.write(
                " ".join(cmd)
            )

            f.write(
                "\n\n[OUTPUT]\n"
            )

            f.write(
                result.stdout or ""
            )

        prediction_file, pocket_count = (
            find_valid_prediction(
                output_dir
            )
        )

        return {
            "returncode": result.returncode,
            "prediction_file": prediction_file,
            "pocket_count": pocket_count,
            "log_file": log_file,
            "output": result.stdout or "",
        }

    except Exception as e:

        with open(
            log_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"ERROR:\n{e}\n"
            )

        return {
            "returncode": -1,
            "prediction_file": None,
            "pocket_count": 0,
            "log_file": log_file,
            "output": str(e),
        }


# ============================================================
# Process one protein
# ============================================================

def process_one_protein(
    protein_id,
    pdb_dir,
    pocket_dir,
    prank,
    config
):

    output_dir = (
        Path(pocket_dir)
        / str(protein_id)
    )

    try:

        pdb_file = find_pdb(
            pdb_dir,
            protein_id
        )

    except Exception as e:

        return {
            "protein_id": protein_id,
            "status": "PDB_NOT_FOUND",
            "level": "",
            "point_threshold": "",
            "min_cluster_size": "",
            "pocket_count": 0,
            "prediction_file": "",
            "pdb_file": "",
            "error": str(e),
        }

    last_error = ""

    # ========================================================
    # Adaptive retry
    # ========================================================

    for parameter in PARAMETER_SETS:

        level = parameter[
            "level"
        ]

        threshold = parameter[
            "point_threshold"
        ]

        cluster = parameter[
            "min_cluster_size"
        ]

        print(
            f"[Protein {protein_id}] "
            f"Trying level {level}: "
            f"threshold={threshold}, "
            f"cluster={cluster}"
        )

        result = run_p2rank_once(
            prank=prank,
            pdb_file=pdb_file,
            output_dir=output_dir,
            parameter=parameter,
            config=config
        )

        # ----------------------------------------------------
        # Valid pocket found
        # ----------------------------------------------------

        if result[
            "pocket_count"
        ] > 0:

            return {
                "protein_id": protein_id,
                "status": "SUCCESS",
                "level": level,
                "point_threshold": threshold,
                "min_cluster_size": cluster,
                "pocket_count": result["pocket_count"],
                "prediction_file": str(
                    result["prediction_file"]
                ),
                "pdb_file": str(pdb_file),
                "error": "",
            }

        # ----------------------------------------------------
        # Save error tail
        # ----------------------------------------------------

        if result[
            "returncode"
        ] != 0:

            output = (
                result["output"]
                or ""
            )

            last_error = (
                output[-2000:]
                .replace(
                    "\n",
                    " | "
                )
            )

        else:

            last_error = (
                "P2Rank finished successfully "
                "but no pocket was detected."
            )

    # ========================================================
    # All relaxed parameters failed
    # ========================================================

    return {
        "protein_id": protein_id,
        "status": "STILL_EMPTY",
        "level": PARAMETER_SETS[-1]["level"],
        "point_threshold":
            PARAMETER_SETS[-1]["point_threshold"],
        "min_cluster_size":
            PARAMETER_SETS[-1]["min_cluster_size"],
        "pocket_count": 0,
        "prediction_file": "",
        "pdb_file": str(pdb_file),
        "error": last_error,
    }


# ============================================================
# Write final report
# ============================================================

def write_report(
    results,
    output_file
):

    output_file = Path(
        output_file
    )

    fields = [
        "protein_id",
        "status",
        "level",
        "point_threshold",
        "min_cluster_size",
        "pocket_count",
        "prediction_file",
        "pdb_file",
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

        for result in results:
            writer.writerow(
                result
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Rerun P2Rank for proteins with "
            "EMPTY/MISSING pocket predictions."
        )
    )

    parser.add_argument(
        "--report",
        default="empty_pocket_report.csv",
        help=(
            "Report generated by check_empty_pockets.py"
        )
    )

    parser.add_argument(
        "--pdb_dir",
        default="esmfold",
        help=(
            "Directory containing ESMFold PDB files. "
            "Default: pdb"
        )
    )

    parser.add_argument(
        "--pocket_dir",
        default="pocket",
        help=(
            "P2Rank output root directory. "
            "Default: pocket"
        )
    )

    parser.add_argument(
        "--prank",
        default="./p2rank_2.5.1/prank",
        help=(
            "Path to P2Rank prank executable"
        )
    )

    parser.add_argument(
        "--config",
        default="alphafold",
        help=(
            "P2Rank configuration. "
            "Default: alphafold"
        )
    )

    parser.add_argument(
        "--statuses",
        default="EMPTY,MISSING",
        help=(
            "Statuses to rerun. "
            "Default: EMPTY,MISSING"
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of proteins processed simultaneously. "
            "Default: 1"
        )
    )

    parser.add_argument(
        "--output",
        default="rerun_p2rank_report.csv",
        help=(
            "Output report CSV"
        )
    )

    args = parser.parse_args()

    # ========================================================
    # Validate inputs
    # ========================================================

    report_file = Path(
        args.report
    )

    pdb_dir = Path(
        args.pdb_dir
    )

    pocket_dir = Path(
        args.pocket_dir
    )

    prank = Path(
        args.prank
    )

    if not report_file.exists():

        raise FileNotFoundError(
            f"Report not found: {report_file}"
        )

    if not pdb_dir.exists():

        raise FileNotFoundError(
            f"PDB directory not found: {pdb_dir}"
        )

    if not pocket_dir.exists():

        pocket_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    if not prank.exists():

        raise FileNotFoundError(
            f"P2Rank executable not found: {prank}\n"
            f"Note: --prank must point to the 'prank' executable, "
            f"not the P2Rank directory."
        )

    # ========================================================
    # Parse statuses
    # ========================================================

    statuses = {
        x.strip().upper()
        for x in args.statuses.split(",")
        if x.strip()
    }

    # ========================================================
    # Read proteins
    # ========================================================

    protein_ids = read_failed_proteins(
        report_file,
        statuses
    )

    protein_ids.sort(
        key=protein_sort_key
    )

    total = len(
        protein_ids
    )

    print("=" * 70)

    print(
        f"Failed proteins found : {total}"
    )

    print(
        f"Statuses              : "
        f"{', '.join(sorted(statuses))}"
    )

    print(
        f"PDB directory         : {pdb_dir}"
    )

    print(
        f"Pocket directory      : {pocket_dir}"
    )

    print(
        f"P2Rank                : {prank}"
    )

    print(
        f"Configuration         : {args.config}"
    )

    print(
        f"Workers               : {args.workers}"
    )

    print("=" * 70)

    if total == 0:

        print(
            "No failed proteins to rerun."
        )

        return

    # ========================================================
    # Run
    # ========================================================

    results = []

    success = 0
    still_empty = 0
    other_failed = 0
    completed = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_to_id = {}

        for protein_id in protein_ids:

            future = executor.submit(
                process_one_protein,
                protein_id,
                pdb_dir,
                pocket_dir,
                prank,
                args.config
            )

            future_to_id[
                future
            ] = protein_id

        for future in as_completed(
            future_to_id
        ):

            protein_id = (
                future_to_id[
                    future
                ]
            )

            try:

                result = (
                    future.result()
                )

            except Exception as e:

                result = {
                    "protein_id": protein_id,
                    "status": "ERROR",
                    "level": "",
                    "point_threshold": "",
                    "min_cluster_size": "",
                    "pocket_count": 0,
                    "prediction_file": "",
                    "pdb_file": "",
                    "error": str(e),
                }

            results.append(
                result
            )

            completed += 1

            if (
                result["status"]
                == "SUCCESS"
            ):

                success += 1

            elif (
                result["status"]
                == "STILL_EMPTY"
            ):

                still_empty += 1

            else:

                other_failed += 1

            print(
                f"[{completed}/{total}] "
                f"Protein={protein_id} "
                f"Status={result['status']} "
                f"Pockets={result['pocket_count']} "
                f"Level={result['level']}"
            )

            # 每完成一个就保存一次，避免中途退出丢数据
            write_report(
                results,
                args.output
            )

    # ========================================================
    # Final sorting
    # ========================================================

    results.sort(
        key=lambda x:
        protein_sort_key(
            str(
                x["protein_id"]
            )
        )
    )

    write_report(
        results,
        args.output
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "RERUN FINISHED"
    )

    print("=" * 70)

    print(
        f"Total processed : {total}"
    )

    print(
        f"Recovered       : {success}"
    )

    print(
        f"Still empty     : {still_empty}"
    )

    print(
        f"Other failures  : {other_failed}"
    )

    print(
        f"Report          : {args.output}"
    )


if __name__ == "__main__":
    main()