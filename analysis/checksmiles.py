import argparse
import os
import sys
import inspect

from rdkit import Chem
import numpy as np
import pandas as pd

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.utils.tokenizer import smi_tokenizer

parser = argparse.ArgumentParser(description='Process SMILES')

parser.add_argument('-s', '--source', help='directory')
parser.add_argument('-f', '--file', help='filename')
parser.add_argument('-c', '--column', help='column name of column of interest')
parser.add_argument('-save', '--save', action='store_true',
    help="If on, save dataframe with errors and a sample of this dataframe")

args = parser.parse_args()
print(args)
df = pd.read_csv(f"{args.source}/{args.file}.csv", header=0, index_col=None)
df = df.drop_duplicates(subset=args.column)

with open(f'{args.source}/{args.file}_log.txt', 'w') as f:
    f.write(
        f'Number of unique SMILES (not the same as molecules): {len(df)} \n')
#df = df.dropna()
#print(df)
error = []
correct = []
for index, row in df.iterrows():
    smile = row[args.column]
    if isinstance(smile, str):
        a = Chem.MolFromSmiles(
            smile
        )  #sanitize = True reasonable, so octet-complete Lewis dot structures (valence, aromatic rings kekulize)
        if a is None:
            error.append(smile)
        else:
            correct.append(smile)
df_valid = pd.DataFrame(correct, columns=['SMILES'])
df_valid.to_csv(f"{args.source}/{args.file}_valid.csv", index=False)

if args.save:
    df_new = pd.DataFrame(error, columns=['SMILES'])
    print(df.shape[0])
    print(df_new.shape[0])
    df_new = df_new.drop_duplicates(subset='SMILES')
    with open(f'{args.source}/{args.file}_log.txt', 'a') as f:
        f.write(
            f'Percentage unique erroneous sequences: {len(df_new) / len(df) *100} %\n'
        )

# check if SMILES can be tokenized
def try_tokenizer(smiles):
    "returns nan if SMILES cannot be tokenized"
    try:
        return len(smi_tokenizer(smiles))
    except AssertionError:
        return np.nan


if args.save:
    df_new["tokens"] = df_new.apply(lambda row: try_tokenizer(row['SMILES']),
                                    axis=1)
    df_new = df_new.dropna(subset=["tokens"])
    df_new = df_new.loc[df_new['tokens'] < 200]
    df_new = df_new.drop(columns=["tokens"])
    df_new.to_csv(f'{args.source}/{args.file}_errors_200.csv', index=False)

    df_s = df_new.sample(10000, random_state=42)
    df_s['SMILES_TARGET'] = df_s['SMILES']
    df_s.to_csv(f'{args.source}/{args.file}_errors_200_S.csv', index=False)
    print(df)

    df_m = df_new.sample(20000, random_state=42)
    df_m['SMILES_TARGET'] = df_m['SMILES']
    df_m.to_csv(f'{args.source}/{args.file}_errors_200_M.csv', index=False)
    print(df)
