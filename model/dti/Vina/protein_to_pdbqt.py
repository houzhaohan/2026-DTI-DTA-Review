#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch protein PDB -> receptor PDBQT

Pipeline:

protein.pdb

    |
    | keep ATOM only
    |
    v

clean_protein.pdb

    |
    | Open Babel
    | -xr
    | -h
    | --partialcharge gasteiger
    |
    v

receptor.pdbqt


Requirements:

conda install -c conda-forge openbabel

"""


import argparse
from pathlib import Path
import subprocess
import pandas as pd



# =====================================================
# Extract protein atoms
# =====================================================

def extract_protein(
        input_pdb,
        output_pdb,
        chain=None
):

    """
    Keep protein ATOM records.

    Remove:
        HETATM
        CONECT
        ANISOU
        ligand
        water

    """

    atom_count = 0


    with open(
        input_pdb,
        "r",
        errors="ignore"
    ) as fin, open(
        output_pdb,
        "w"
    ) as fout:


        for line in fin:


            if not line.startswith(
                "ATOM"
            ):
                continue



            # select chain

            if chain is not None:

                pdb_chain = line[21].strip()

                if pdb_chain != chain:

                    continue



            fout.write(line)

            atom_count += 1



        fout.write(
            "END\n"
        )


    if atom_count == 0:

        raise RuntimeError(
            f"No protein ATOM found in {input_pdb}"
        )


    return atom_count



# =====================================================
# Open Babel convert
# =====================================================

def obabel_receptor(
        pdb,
        pdbqt,
        obabel="obabel"
):


    cmd=[

        obabel,

        str(pdb),

        "-O",

        str(pdbqt),

        "-xr",

        "-h",

        "--partialcharge",

        "gasteiger"

    ]


    result=subprocess.run(

        cmd,

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True

    )


    if result.returncode != 0:

        raise RuntimeError(
            result.stdout
        )


    if not pdbqt.exists():

        raise RuntimeError(
            "PDBQT not generated"
        )



# =====================================================
# Validate receptor
# =====================================================

def check_pdbqt(
        pdbqt
):


    if not pdbqt.exists():

        return False


    atom_count=0


    with open(
        pdbqt,
        "r",
        errors="ignore"
    ) as f:


        for line in f:

            if line.startswith(
                "ATOM"
            ):

                atom_count+=1



    return atom_count > 0



# =====================================================
# Main
# =====================================================

def main():


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--pdb_dir",
        required=True,
        help="input pdb directory"
    )


    parser.add_argument(
        "--outdir",
        default="receptor_pdbqt"
    )


    parser.add_argument(
        "--chain",
        default=None,
        help="protein chain, e.g. A"
    )


    parser.add_argument(
        "--obabel",
        default="obabel"
    )


    args=parser.parse_args()



    pdb_dir=Path(
        args.pdb_dir
    )


    outdir=Path(
        args.outdir
    )


    outdir.mkdir(
        parents=True,
        exist_ok=True
    )



    clean_dir=outdir/"clean_pdb"

    clean_dir.mkdir(
        exist_ok=True
    )



    pdb_files=sorted(
        pdb_dir.glob("*.pdb")
    )


    print(
        "Total proteins:",
        len(pdb_files)
    )



    success=[]

    failed=[]



    for i,pdb in enumerate(pdb_files):


        name=pdb.stem


        print(
            f"\n[{i+1}/{len(pdb_files)}] {name}"
        )


        try:


            clean_pdb=(

                clean_dir /
                f"{name}.pdb"

            )


            receptor=(

                outdir /
                f"{name}.pdbqt"

            )


            atom_count=extract_protein(

                pdb,

                clean_pdb,

                args.chain

            )


            print(
                "ATOM kept:",
                atom_count
            )



            obabel_receptor(

                clean_pdb,

                receptor,

                args.obabel

            )



            if not check_pdbqt(
                receptor
            ):

                raise RuntimeError(
                    "Invalid receptor pdbqt"
                )



            success.append(
                name
            )


            print(
                "[SUCCESS]"
            )



        except Exception as e:


            print(
                "[FAILED]"
            )

            print(
                str(e)
            )


            failed.append(
                {
                    "protein":name,
                    "error":str(e)
                }
            )



    print("\n================")

    print(
        "Success:",
        len(success)
    )


    print(
        "Failed:",
        len(failed)
    )


    print(
        "================"
    )



    if failed:


        pd.DataFrame(
            failed
        ).to_csv(

            outdir/
            "failed.csv",

            index=False

        )



if __name__=="__main__":

    main()