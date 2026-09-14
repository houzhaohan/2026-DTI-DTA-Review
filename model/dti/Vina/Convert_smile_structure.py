#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMILES csv -> individual 3D sdf files

Input:
    smile.csv

Example:
    SMILES_index,SMILES
    0,CCO
    1,CCN

Output:

    ligands_sdf/

        0.sdf
        1.sdf
        ...

"""


import argparse
from pathlib import Path
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem



def smiles_to_3d_sdf(
        smiles,
        output_file,
        seed=42
):


    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:
        raise ValueError(
            "Invalid SMILES"
        )


    # 加氢
    mol = Chem.AddHs(
        mol
    )


    # 3D构象
    params = AllChem.ETKDGv3()

    params.randomSeed = seed


    status = AllChem.EmbedMolecule(
        mol,
        params
    )


    if status != 0:

        params.useRandomCoords = True

        status = AllChem.EmbedMolecule(
            mol,
            params
        )


    if status != 0:

        raise RuntimeError(
            "3D generation failed"
        )


    # 优化
    try:

        if AllChem.MMFFHasAllMoleculeParams(mol):

            AllChem.MMFFOptimizeMolecule(
                mol,
                maxIters=500
            )

        else:

            AllChem.UFFOptimizeMolecule(
                mol,
                maxIters=500
            )

    except:

        pass


    writer = Chem.SDWriter(
        str(output_file)
    )


    writer.write(
        mol
    )

    writer.close()



def main():


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--csv",
        required=True
    )


    parser.add_argument(
        "--outdir",
        default="ligands_sdf"
    )


    parser.add_argument(
        "--id",
        default="SMILES_index"
    )


    parser.add_argument(
        "--smiles",
        default="SMILES"
    )


    args = parser.parse_args()



    df = pd.read_csv(
        args.csv
    )


    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        exist_ok=True
    )


    success = 0
    failed = []


    total = len(df)


    for i,row in df.iterrows():


        mol_id = str(
            row[args.id]
        )


        smiles = row[
            args.smiles
        ]


        outfile = (
            outdir /
            f"{mol_id}.sdf"
        )


        print(
            f"[{i+1}/{total}] {mol_id}"
        )


        try:


            smiles_to_3d_sdf(
                smiles,
                outfile
            )


            # 写入额外信息
            mol = Chem.SDMolSupplier(
                str(outfile),
                removeHs=False
            )[0]


            if mol:

                mol.SetProp(
                    "SMILES_index",
                    mol_id
                )

                mol.SetProp(
                    "SMILES",
                    smiles
                )


                writer = Chem.SDWriter(
                    str(outfile)
                )

                writer.write(mol)
                writer.close()


            success += 1


        except Exception as e:


            print(
                "[FAILED]",
                mol_id,
                e
            )

            failed.append(
                [
                    mol_id,
                    smiles,
                    str(e)
                ]
            )



    print("\nFinished")

    print(
        "Success:",
        success
    )

    print(
        "Failed:",
        len(failed)
    )


    if failed:

        pd.DataFrame(
            failed,
            columns=[
                "id",
                "smiles",
                "error"
            ]
        ).to_csv(
            "failed_smiles.csv",
            index=False
        )



if __name__=="__main__":

    main()