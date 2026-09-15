#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch run P2Rank pocket prediction

Input:
    pdb_dir/
        *.pdb

Output:
    pocket_dir/
        pdb_name/
            *_predictions.csv
            *_pockets.pdb
            ...

Usage:

python batch_prank.py \
    --pdb_dir ./pdb \
    --out_dir ./pocket \
    --prank /path/to/prank

"""


import argparse
import subprocess
from pathlib import Path
import multiprocessing
import os


def run_prank(
        pdb_file,
        out_dir,
        prank="prank"
):

    pdb_file = Path(pdb_file)

    name = pdb_file.stem

    result_dir = Path(out_dir) / name

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    cmd = [
        prank,
        "predict",
        "-f",
        str(pdb_file),
        "-o",
        str(result_dir)
    ]


    print(
        f"\nRunning P2Rank: {name}"
    )

    print(
        " ".join(cmd)
    )


    try:

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )


        log_file = result_dir / "prank.log"

        with open(
            log_file,
            "w"
        ) as f:
            f.write(result.stdout)


        if result.returncode == 0:

            print(
                f"[OK] {name}"
            )

        else:

            print(
                f"[FAILED] {name}"
            )

            print(
                result.stdout[-500:]
            )


    except Exception as e:

        print(
            f"[ERROR] {name}: {e}"
        )



def main():

    parser = argparse.ArgumentParser(
        description="Batch run P2Rank"
    )


    parser.add_argument(
        "--pdb_dir",
        required=True,
        help="Directory containing pdb files"
    )


    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output pocket directory"
    )


    parser.add_argument(
        "--prank",
        default="prank",
        help="Path of prank executable"
    )


    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Parallel jobs"
    )


    args = parser.parse_args()


    pdb_dir = Path(
        args.pdb_dir
    )


    out_dir = Path(
        args.out_dir
    )


    pdb_files = sorted(
        pdb_dir.glob("*.pdb")
    )


    if len(pdb_files)==0:

        raise RuntimeError(
            "No pdb files found"
        )


    print(
        f"Found {len(pdb_files)} pdb files"
    )


    tasks = []


    for pdb in pdb_files:

        tasks.append(
            (
                pdb,
                out_dir,
                args.prank
            )
        )


    # 多进程运行
    with multiprocessing.Pool(
        processes=args.threads
    ) as pool:

        pool.starmap(
            run_prank,
            tasks
        )


    print(
        "\nAll finished!"
    )



if __name__ == "__main__":

    main()