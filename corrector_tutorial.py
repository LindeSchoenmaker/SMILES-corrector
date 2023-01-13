import torch
import os
import argparse

import pandas as pd
import numpy as np

import random

from src.preprocess import standardize, train_valid_test_split, remove_long_sequences
from src.invalidSMILES import get_invalid_smiles
from src.modelling import initialize_model, train_model, correct_SMILES


def ArgParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-r', '--folder_raw', type=str, default='RawData/',
                        help="Directory containing input files data_source & gdb8.csv")
    parser.add_argument('-o', '--folder_out', type=str, default='Data/',
                        help="Directory for saving output files")
    parser.add_argument('-d', '--data_source', type=str, default='PAPYRUS.csv',
                        help="data source to base synthetic errors on")
    parser.add_argument('-es', '--error_source', type=str, default='Data/papyrus_rnn_S.csv',
                        help="file with invalid SMILES to fix")
    parser.add_argument('-ran', '--random_state', type=int, default=42, help="Seed for the random state")
    parser.add_argument('-i', '--input', type=str, default='dataset',
                        help="tsv file name that contains SMILES, target accession & corresponding data")
    parser.add_argument('-th', '--threshold', type=int, default=200,
                        help="maximum sequence length")
    parser.add_argument('-n', '--num_errors', type=int, default=1,
                        help="Batch size")
    parser.add_argument('-type', '--invalid_type', type=str, default='all',
                        help='type of error to introduce, ["all", "exists", "par", "permut", "ring", "syntax", "valence", "arom"] for num_errors = 1 & "multiple" for num_errors > 1')
    parser.add_argument('-train', '--training', action='store_true',
                        help='If on, corrector is trained')
    parser.add_argument('-fix', '--fixing', action='store_true',
                        help='If on, model is loaded and used for fixing SMILES from error_source')
    parser.add_argument('-gpu', '--gpu', type=str, default='1',
                        help="GPU to use") 
    parser.add_argument('-bs', '--batch_size', type=int, default=16,
                        help="Batch size")
    parser.add_argument('-l', '--layers', type=int, default=3,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help="Number of epochs")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = ArgParser()

    folder_raw = args.folder_raw
    folder_out = args.folder_out
    data_source = args.data_source
    error_source = args.error_source
    threshold =args.threshold
    invalid_type = args.invalid_type
    num_errors = args.num_errors

    # set random seed, used for error generation & initiation transformer
    SEED = args.random_state
    random.seed(SEED)

    # create standardized dataset if not already present
    if os.path.exists(
            f"{folder_out}{data_source.split('.')[0]}_{threshold}_standardized.csv"
    ):
        # Load dataset of standardized SMILES & of fragments
        df = pd.read_csv(
            f"{folder_out}{data_source.split('.')[0]}_{threshold}_standardized.csv",
            usecols=["STD_SMILES"],
            header=0,
            index_col=None,
        )

    else:
        # standardize
        df = standardize(folder_raw, data_source)
        # remove long sequences
        df = remove_long_sequences(df,
                                   subset="STD_SMILES",
                                   threshold=threshold)
        # save standardized dataframe
        df.to_csv(
            f"Data/{data_source.split('.')[0]}_{threshold}_standardized.csv",
            index=False)
        df = df['STD_SMILES']

    data_source = f"{data_source.split('.')[0]}_{threshold}"

    # create synthetic invalid SMILES if not already present
    if os.path.exists(
            f"{folder_out}errors/{data_source}_{invalid_type}_{num_errors}_errors.csv"
    ):
        # Load dataset of invalid and valid SMILES
        df = pd.read_csv(
            f"{folder_out}errors/{data_source}_{invalid_type}_{num_errors}_errors.csv",
            usecols=["STD_SMILES", "ERROR"],
            header=0,
            index_col=None,
        )

    else:
        df_frag = pd.read_csv(f"{folder_raw}gbd_8.csv",
                              names=["FRAGMENT"],
                              usecols=[0],
                              header=0).dropna()
        # takes few minutes when using ray on ~24 CPUs
        df = get_invalid_smiles(df, df_frag, SEED, invalid_type, num_errors)

        # remove long sequences
        df = remove_long_sequences(df,
                                   subset="STD_SMILES",
                                   threshold=threshold)
        df = remove_long_sequences(df, subset="ERROR", threshold=threshold)

        if not os.path.exists(f"{folder_out}errors"):
            os.makedirs(f"{folder_out}errors")

        df.to_csv(
            f"{folder_out}errors/{data_source}_{invalid_type}_{num_errors}_errors.csv",
            index=False)
        print(df)

    if not os.path.exists(
            f"{folder_out}errors/split/{data_source}_{invalid_type}_{num_errors}_errors_train.csv"
    ):
        # for splitting the data and turning it into a torchtext dataset
        train, valid, _ = train_valid_test_split(df, SEED=SEED)
        if not os.path.exists(f"{folder_out}errors/split"):
            os.makedirs(f"{folder_out}errors/split")
        train.to_csv(
            f"{folder_out}errors/split/{data_source}_{invalid_type}_{num_errors}_errors_train.csv",
            index=False)
        valid.to_csv(
            f"{folder_out}errors/split/{data_source}_{invalid_type}_{num_errors}_errors_dev.csv",
            index=False)

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # define this in test.py
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, out, SRC = initialize_model(
        folder_out,
        data_source,
        error_source,
        device,
        threshold=threshold,
        epochs=args.epochs,
        layers=args.layers,
        batch_size=args.batch_size,
        invalid_type=invalid_type,
        num_errors=num_errors,
    )
    if args.training:
        model = train_model(model, out, False)

    elif args.fixing:
        print(f"Fixing {error_source.split('/')[-1].split('.')[0]}")

        valids, df_output = correct_SMILES(model, out, error_source, device,
                                           SRC)
        df_output.to_csv(
            f"generated/{out.split('/')[-1]}_{error_source.split('/')[-1].split('.')[0]}_fixed.csv",
            index=False)
