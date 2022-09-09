import torch
import os

import pandas as pd
import numpy as np

import random

import pickle

from src.invalidSMILES import get_invalid_smiles
from src.preprocess import standardization_pipeline, remove_smiles_duplicates
from src.modelling import initialize_model, correct_SMILES

if __name__ == "__main__":
    # set random seed, used for error generation & initiation transformer
    SEED = 42
    random.seed(SEED)

    name = 'selective_ki'
    errors_per_molecule = 1000
    if not os.path.exists(f"Data/explore"):
        os.makedirs(f"Data/explore")
    error_source = "Data/explore/%s_with_%s_errors_index.csv" % (
        name, errors_per_molecule)

    folder_raw = "RawData/"
    folder_out = "Data/"
    invalid_type = 'multiple'
    num_errors = 12
    threshold = 200
    data_source = f"PAPYRUS_{threshold}"

    # introduce = True

    standardize = False
    if standardize:
        df = pd.read_csv('%s%s.csv' % (folder_out, name), usecols=['SMILES']).dropna()
        df["STD_SMILES"] = df.apply(
            lambda row: standardization_pipeline(row["SMILES"]),
            axis=1).dropna()
        df = df.drop(columns=['SMILES'])
        df.to_csv('%s%s.csv' % (folder_out, name), index=None)
    else:
        df = pd.read_csv('%s%s.csv' % (folder_out, name),
                         usecols=['STD_SMILES']).dropna()

    introduce = False
    if introduce:
        df_frag = pd.read_csv(f"{folder_raw}gbd_8.csv",
                              names=["FRAGMENT"],
                              usecols=[0],
                              header=0).dropna()

        # duplicate SMILES to create multiple errors per molecule
        df = df.append([df] * (errors_per_molecule - 1), ignore_index=False)
        index_list = list(df.index.values)
        smiles = list(df['STD_SMILES'].values)
        df = get_invalid_smiles(df,
                                df_frag,
                                SEED,
                                invalid_type="all",
                                num_errors=1)
        df = df.drop(columns=['FRAGMENT']).reset_index(drop=True)
        df = df.rename(columns={"ERROR": "SMILES"})
        df['SMILES_TARGET'] = df["SMILES"]
        df.index = index_list
        df['ORIGINAL_SMILES'] = smiles
        df = df.drop_duplicates(subset=['SMILES'])
        df.to_csv(error_source)

    correct = False
    if correct:
        # define this in test.py
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        initialize_source = 'Data/papyrus_rnn_S.csv'

        #device = torch.device('cpu')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, out, SRC = initialize_model(folder_out,
                                           data_source,
                                           error_source=initialize_source,
                                           device=device,
                                           threshold=threshold,
                                           epochs=30,
                                           layers=3,
                                           batch_size=16,
                                           invalid_type=invalid_type,
                                           num_errors=num_errors)
        print(out)
        valids, df_output = correct_SMILES(model, out, error_source, device,
                                           SRC)
        df_output.to_csv(
            f"Data/explore/{error_source.split('/')[2].split('.')[0]}_fixed.csv",
            index=False)

    df_new = pd.read_csv(f"Data/explore/{error_source.split('/')[2].split('.')[0]}_fixed.csv", usecols = ['CORRECT']).dropna()
    df_new["STD_SMILES"] = df_new.apply(
                lambda row: standardization_pipeline(row["CORRECT"]), axis=1
            ).dropna()
    df_new = remove_smiles_duplicates(df_new, subset="STD_SMILES")