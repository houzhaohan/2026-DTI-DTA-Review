#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path


def has_pocket(prediction_file):
    """
    判断 P2Rank predictions.csv 是否包含至少一个 pocket。

    返回:
        True  -> 有 pocket
        False -> 文件为空，或者只有表头
    """

    try:
        with open(
            prediction_file,
            "r",
            encoding="utf-8-sig"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # 0行：完全空文件
        if len(lines) == 0:
            return False

        # 1行：只有表头
        if len(lines) == 1:
            return False

        # >=2 行：至少有一个 pocket
        return True

    except Exception:
        return False


def find_prediction_file(protein_dir):
    """
    自动寻找 P2Rank predictions 文件。

    可以匹配例如：
        0.pdb_predictions.csv
        0_predictions.csv
        xxx_predictions.csv
    """

    files = sorted(
        protein_dir.glob("*predictions*.csv")
    )

    if not files:
        return None

    return files[0]


def main():

    parser = argparse.ArgumentParser(
        description="Check empty P2Rank prediction files"
    )

    parser.add_argument(
        "--pocket_dir",
        default="pocket",
        help="P2Rank result root directory, default: pocket"
    )

    parser.add_argument(
        "--output",
        default="empty_pocket_report.csv",
        help="Output report CSV"
    )

    args = parser.parse_args()

    pocket_dir = Path(args.pocket_dir)

    if not pocket_dir.exists():
        raise FileNotFoundError(
            f"Pocket directory not found: {pocket_dir}"
        )

    # -------------------------------------------------------
    # 获取所有蛋白目录
    # 只考虑目录
    # -------------------------------------------------------

    protein_dirs = [
        p
        for p in pocket_dir.iterdir()
        if p.is_dir()
    ]

    # 如果目录名是数字，按照数字排序
    def sort_key(p):
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)

    protein_dirs.sort(
        key=sort_key
    )

    results = []

    n_total = 0
    n_valid = 0
    n_empty = 0
    n_missing = 0

    empty_ids = []
    missing_ids = []

    # -------------------------------------------------------
    # 检查每个蛋白
    # -------------------------------------------------------

    for protein_dir in protein_dirs:

        protein_id = protein_dir.name

        n_total += 1

        prediction_file = find_prediction_file(
            protein_dir
        )

        # 没找到 predictions 文件
        if prediction_file is None:

            status = "MISSING"

            n_missing += 1

            missing_ids.append(
                protein_id
            )

            results.append({
                "protein_id": protein_id,
                "status": status,
                "prediction_file": ""
            })

            continue

        # 找到了 predictions 文件
        if has_pocket(prediction_file):

            status = "VALID"
            n_valid += 1

        else:

            status = "EMPTY"
            n_empty += 1

            empty_ids.append(
                protein_id
            )

        results.append({
            "protein_id": protein_id,
            "status": status,
            "prediction_file": str(prediction_file)
        })

    # -------------------------------------------------------
    # 保存报告
    # -------------------------------------------------------

    output_file = Path(
        args.output
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "protein_id",
                "status",
                "prediction_file"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # -------------------------------------------------------
    # 打印统计
    # -------------------------------------------------------

    print("=" * 60)

    print(
        f"Total protein folders : {n_total}"
    )

    print(
        f"Valid pockets         : {n_valid}"
    )

    print(
        f"Empty predictions     : {n_empty}"
    )

    print(
        f"Missing predictions   : {n_missing}"
    )

    print("=" * 60)

    if n_total > 0:

        empty_percent = (
            n_empty
            / n_total
            * 100
        )

        print(
            f"Empty percentage      : "
            f"{empty_percent:.2f}%"
        )

    print(
        f"\nReport saved to: {output_file}"
    )

    # -------------------------------------------------------
    # 打印空 pocket 的蛋白 ID
    # -------------------------------------------------------

    if empty_ids:

        print(
            "\nProteins with EMPTY predictions:"
        )

        print(
            ", ".join(
                empty_ids
            )
        )

    if missing_ids:

        print(
            "\nProteins with MISSING predictions file:"
        )

        print(
            ", ".join(
                missing_ids
            )
        )


if __name__ == "__main__":
    main()