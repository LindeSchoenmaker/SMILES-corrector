from src.utils.tokenizer import smi_tokenizer, remove_floats
from sklearn.model_selection import train_test_split
from chembl_structure_pipeline import standardizer
from rdkit.Chem import MolStandardize
from rdkit import Chem
import modin.pandas as pd
import os

os.environ["MODIN_CPUS"] = "16"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray



def remove_smiles_duplicates(dataframe: pd.DataFrame,
                             subset: str) -> pd.DataFrame:
    return dataframe.drop_duplicates(subset=subset)


def remove_long_sequences(df: pd.DataFrame, subset: str, threshold: int):
    """Removes long sequences from the list based on the number of tokens they contain"""
    df = remove_floats(df, subset)

    # tokenize entries and get length of resulting list
    df["length"] = df.apply(lambda row: len(smi_tokenizer(row[subset])),
                            axis=1)

    df = df[df.length < threshold]

    return df.drop(columns=['length'])


def standardization_pipeline(smile):
    desalter = MolStandardize.fragment.LargestFragmentChooser()
    std_smile = None
    if not isinstance(smile, str): return None
    m = Chem.MolFromSmiles(smile)
    # skips smiles for which no mol file could be generated
    if m is not None:
        # standardizes
        std_m = standardizer.standardize_mol(m)
        # strips salts
        std_m_p, exclude = standardizer.get_parent_mol(std_m)
        if not exclude:
            # choose largest fragment for rare cases where chembl structure
            # pipeline leaves 2 fragments
            std_m_p_d = desalter.choose(std_m_p)
            std_smile = Chem.MolToSmiles(std_m_p_d)
    return std_smile


def standardize(folder, data_source, short = False):
    """apply chembl standardization pipeline"""
    # Load dataset with molecules to use for training and evaluating corrector
    # done in chuncks with modin to speed up
    if short: 
        nrows = 10000 
        chunksize = 1000
    else: 
        nrows = None
        chunksize = 100000
    df_chunk = pd.read_csv(
        os.path.join(folder, data_source),
        engine="python",
        sep=";",
        header=0,
        index_col=None,
        dtype={"SMILES": "str"},
        nrows = nrows,
        chunksize=chunksize
    )
    chunk_list = []
    # Each chunk is in df format
    for i, chunk in enumerate(df_chunk):
        # keep SMILES info of entries with SMILES
        if 'Smiles' in chunk.columns:
            chunk = chunk[["Smiles"]].rename(columns={
                "Smiles": "SMILES"
            }).dropna()

        # Standardize SMILES
        # takes 5h for chembl
        chunk["STD_SMILES"] = chunk.apply(
            lambda row: standardization_pipeline(row["SMILES"]), axis=1)

        # remove rows for which standardization did not create standardized
        # SMILES
        chunk = chunk.dropna(subset=["STD_SMILES"])
        chunk_list.append(chunk)

    df_concat = pd.concat(chunk_list)

    # remove duplicate molecules from dataset using canonical smiles
    df_concat = remove_smiles_duplicates(df_concat, subset="STD_SMILES")

    return df_concat


def train_valid_test_split(dataframe: pd.DataFrame,
                           SEED: int,
                           frac_train=0.9,
                           frac_valid=0.1,
                           frac_test=0) -> list:
    """split data into train, validation and optionally test sets"""
    train_set, tmp_set = train_test_split(dataframe,
                                          train_size=frac_train,
                                          random_state=SEED)
    if frac_test > 0:
        valid_set, test_set = train_test_split(tmp_set,
                                               train_size=frac_valid /
                                               (frac_valid + frac_test),
                                               random_state=SEED)
        del tmp_set
    else:
        valid_set = tmp_set
        test_set = None
    return train_set, valid_set, test_set
