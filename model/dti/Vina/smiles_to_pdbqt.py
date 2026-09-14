#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch SMILES -> ligand PDBQT

Input:
    CSV file

Example:
    SMILES_index,SMILES
    0,CCO
    1,c1ccccc1


Output:

    ligands_pdbqt/

        0.pdbqt
        1.pdbqt
        ...


Requirements:

    conda install -c conda-forge rdkit

    pip install meeko


"""


import argparse
from pathlib import Path
import subprocess
import tempfile
import pandas as pd
import sys


from rdkit import Chem
from rdkit.Chem import AllChem



# =====================================================
# SMILES -> 3D SDF
# =====================================================

def smiles_to_sdf(
        smiles,
        sdf_file,
        seed=42
):

    """
    Generate 3D molecule using RDKit
    """

    mol = Chem.MolFromSmiles(
        smiles
    )


    if mol is None:
        raise ValueError(
            "Invalid SMILES"
        )


    # add hydrogens
    mol = Chem.AddHs(
        mol
    )


    # ETKDG
    params = AllChem.ETKDGv3()

    params.randomSeed = seed


    result = AllChem.EmbedMolecule(
        mol,
        params
    )


    # retry
    if result != 0:

        params.useRandomCoords = True

        result = AllChem.EmbedMolecule(
            mol,
            params
        )


    if result != 0:

        raise RuntimeError(
            "3D conformer generation failed"
        )


    # Geometry optimization

    try:

        if AllChem.MMFFHasAllMoleculeParams(
            mol
        ):

            AllChem.MMFFOptimizeMolecule(
                mol,
                maxIters=500
            )

        else:

            AllChem.UFFOptimizeMolecule(
                mol,
                maxIters=500
            )


    except Exception:

        # keep embedded structure
        pass



    writer = Chem.SDWriter(
        str(sdf_file)
    )


    writer.write(
        mol
    )

    writer.close()



# =====================================================
# Meeko SDF -> PDBQT
# =====================================================

def sdf_to_pdbqt(
        sdf_file,
        pdbqt_file
):


    cmd = [

        "mk_prepare_ligand.py",

        "-i",
        str(sdf_file),

        "-o",
        str(pdbqt_file)

    ]


    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    if result.returncode != 0:

        raise RuntimeError(
            result.stderr
        )



    if not pdbqt_file.exists():

        raise RuntimeError(
            "PDBQT not generated"
        )



# =====================================================
# main
# =====================================================

def main():


    parser = argparse.ArgumentParser(
        description="Convert SMILES CSV to PDBQT"
    )


    parser.add_argument(
        "--csv",
        required=True,
        help="SMILES csv file"
    )


    parser.add_argument(
        "--outdir",
        default="ligands_pdbqt",
        help="Output directory"
    )


    parser.add_argument(
        "--id-column",
        default="SMILES_index"
    )


    parser.add_argument(
        "--smiles-column",
        default="SMILES"
    )


    args = parser.parse_args()



    csv_file = Path(
        args.csv
    )


    outdir = Path(
        args.outdir
    )


    outdir.mkdir(
        parents=True,
        exist_ok=True
    )



    df = pd.read_csv(
        csv_file
    )


    print(
        "="*60
    )

    print(
        f"Total molecules: {len(df)}"
    )

    print(
        "="*60
    )



    success = 0

    failed = []



    for index,row in df.iterrows():


        mol_id = str(
            row[args.id_column]
        )


        smiles = str(
            row[args.smiles_column]
        )


        output_pdbqt = (
            outdir /
            f"{mol_id}.pdbqt"
        )



        print(
            f"[{index+1}/{len(df)}] "
            f"{mol_id}"
        )



        try:


            with tempfile.TemporaryDirectory() as tmp:


                sdf_file = (
                    Path(tmp) /
                    "ligand.sdf"
                )


                # SMILES -> SDF

                smiles_to_sdf(
                    smiles,
                    sdf_file
                )


                # SDF -> PDBQT

                sdf_to_pdbqt(
                    sdf_file,
                    output_pdbqt
                )



            success += 1


        except Exception as e:


            print(
                f"[FAILED] {mol_id}"
            )

            print(
                e
            )


            failed.append(
                {
                    "SMILES_index":mol_id,
                    "SMILES":smiles,
                    "error":str(e)
                }
            )



    print("\n")
    print("="*60)

    print(
        "Finished"
    )

    print(
        f"Success: {success}"
    )

    print(
        f"Failed : {len(failed)}"
    )


    print("="*60)



    if failed:


        failed_file = (
            outdir /
            "failed_smiles.csv"
        )


        pd.DataFrame(
            failed
        ).to_csv(
            failed_file,
            index=False
        )


        print(
            "Failed list:"
        )

        print(
            failed_file
        )



if __name__ == "__main__":


    try:

        main()


    except KeyboardInterrupt:

        print(
            "\nInterrupted"
        )

        sys.exit(1)