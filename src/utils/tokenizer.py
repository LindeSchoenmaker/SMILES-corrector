import re
import pandas as pd


def remove_floats(df: pd.DataFrame, subset: str):
    """Preprocessing step to remove any entries that are not strings"""
    df_subset = df[subset]
    df[subset] = df[subset].astype(str)
    # only keep entries that stayed the same after applying astype str
    df = df[df[subset] == df_subset].copy()

    return df


def smi_tokenizer(smi: str, reverse=False) -> list:
    """
    Tokenize a SMILES molecule
    """
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    # tokens = ['<sos>'] + [token for token in regex.findall(smi)] + ['<eos>']
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens[1:-1])
    assert smi == "".join(tokens[:])
    # try:
    #     assert smi == "".join(tokens[:])
    # except:
    #     print(smi)
    #     print("".join(tokens[:]))
    if reverse:
        return tokens[::-1]
    return tokens
