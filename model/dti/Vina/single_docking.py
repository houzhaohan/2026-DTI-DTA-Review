#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import subprocess
import csv
import sys



# =====================================================
# Find file
# =====================================================

def find_file(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return path



# =====================================================
# Find P2Rank csv automatically
# =====================================================

def find_pocket_csv(
        pocket_dir,
        protein_id
):

    folder = (
        Path(pocket_dir)
        /
        str(protein_id)
    )


    if not folder.exists():

        raise FileNotFoundError(
            f"Pocket folder not found: {folder}"
        )


    files = list(
        folder.glob("*predictions*.csv")
    )


    if len(files) == 0:

        raise FileNotFoundError(
            f"No prediction csv found in {folder}"
        )


    # 如果多个，默认第一个
    return files[0]



# =====================================================
# Read P2Rank center
# =====================================================

def get_pocket_center(
        pocket_csv,
        rank=1
):


    with open(
        pocket_csv,
        "r"
    ) as f:


        reader = csv.DictReader(f)


        if reader.fieldnames is None:

            raise RuntimeError(
                "Empty pocket csv"
            )


        # 去除列名空格
        reader.fieldnames = [
            x.strip()
            for x in reader.fieldnames
        ]


        rows=[]


        for row in reader:


            clean_row={}


            for k,v in row.items():

                clean_row[
                    k.strip()
                ] = v.strip()


            rows.append(
                clean_row
            )



    required = [

        "rank",
        "center_x",
        "center_y",
        "center_z"

    ]



    for col in required:

        if col not in reader.fieldnames:

            raise RuntimeError(
                "Missing column: "
                + col
                +
                "\nAvailable columns:"
                +
                str(reader.fieldnames)
            )



    for row in rows:


        try:

            r=int(row["rank"])

        except:

            continue



        if r == rank:


            return {

                "x": float(
                    row["center_x"]
                ),

                "y": float(
                    row["center_y"]
                ),

                "z": float(
                    row["center_z"]
                )

            }



    raise RuntimeError(
        f"Cannot find pocket rank {rank}"
    )



# =====================================================
# Run Vina
# =====================================================

def run_vina(
        receptor,
        ligand,
        center,
        output,
        exhaustiveness=16
):


    cmd=[

        "vina",

        "--receptor",
        str(receptor),

        "--ligand",
        str(ligand),


        "--center_x",
        str(center["x"]),

        "--center_y",
        str(center["y"]),

        "--center_z",
        str(center["z"]),


        "--size_x",
        "22",

        "--size_y",
        "22",

        "--size_z",
        "22",


        "--exhaustiveness",
        str(exhaustiveness),


        "--num_modes",
        "10",


        "--out",
        str(output)

    ]



    print("\nRunning Vina:")
    print(
        " ".join(cmd)
    )



    result=subprocess.run(

        cmd,

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True

    )


    print(
        result.stdout
    )



    if result.returncode != 0:

        raise RuntimeError(
            "Vina failed"
        )



# =====================================================
# Parse Vina score
# =====================================================

def parse_score(
        pdbqt
):


    scores=[]


    with open(
        pdbqt,
        "r"
    ) as f:


        for line in f:


            if "REMARK VINA RESULT" in line:


                try:

                    score=float(
                        line.split()[3]
                    )

                    scores.append(
                        score
                    )

                except:

                    pass



    if scores:

        return min(scores)


    return None



# =====================================================
# Save result
# =====================================================

def save_result(
        file,
        protein_id,
        ligand_id,
        score,
        pose
):


    file=Path(file)


    write_header = not file.exists()



    with open(
        file,
        "a",
        newline=""
    ) as f:


        writer=csv.writer(f)



        if write_header:

            writer.writerow(

                [
                    "protein_id",
                    "ligand_id",
                    "score",
                    "pose"
                ]

            )



        writer.writerow(

            [
                protein_id,
                ligand_id,
                score,
                pose
            ]

        )



# =====================================================
# Main
# =====================================================

def main():



    parser=argparse.ArgumentParser(
        description="Run AutoDock Vina docking"
    )


    parser.add_argument(
        "--protein_id",
        required=True
    )


    parser.add_argument(
        "--ligand_id",
        required=True
    )


    parser.add_argument(
        "--protein_dir",
        default="receptor_pdbqt"
    )


    parser.add_argument(
        "--ligand_dir",
        default="ligands_pdbqt"
    )


    parser.add_argument(
        "--pocket_dir",
        default="pocket"
    )


    parser.add_argument(
        "--pocket_rank",
        type=int,
        default=1
    )


    parser.add_argument(
        "--outdir",
        default="docking_results"
    )


    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=16
    )


    args=parser.parse_args()



    try:


        # --------------------------
        # receptor
        # --------------------------

        receptor=find_file(

            Path(args.protein_dir)
            /
            f"{args.protein_id}.pdbqt"

        )



        # --------------------------
        # ligand
        # --------------------------

        ligand=find_file(

            Path(args.ligand_dir)
            /
            f"{args.ligand_id}.pdbqt"

        )



        # --------------------------
        # pocket
        # --------------------------

        pocket_csv=find_pocket_csv(

            args.pocket_dir,

            args.protein_id

        )



        print("====================")

        print(
            "Protein:",
            receptor
        )

        print(
            "Ligand:",
            ligand
        )

        print(
            "Pocket:",
            pocket_csv
        )

        print("====================")



        # --------------------------
        # center
        # --------------------------

        center=get_pocket_center(

            pocket_csv,

            args.pocket_rank

        )


        print(
            "Center:",
            center
        )



        # --------------------------
        # output
        # --------------------------

        outdir=Path(
            args.outdir
        )


        outdir.mkdir(
            parents=True,
            exist_ok=True
        )


        pose=(

            outdir
            /
            f"protein_{args.protein_id}_ligand_{args.ligand_id}.pdbqt"

        )



        # --------------------------
        # docking
        # --------------------------

        run_vina(

            receptor,

            ligand,

            center,

            pose,

            args.exhaustiveness

        )



        score=parse_score(
            pose
        )


        print(
            "\nDocking score:",
            score
        )



        save_result(

            outdir /
            "summary.csv",

            args.protein_id,

            args.ligand_id,

            score,

            pose

        )



        print(
            "\nFinished"
        )



    except Exception as e:


        print(
            "\nERROR:"
        )

        print(
            e
        )

        sys.exit(1)



if __name__=="__main__":

    main()