from src.utils.sanifix import AdjustAromaticNs
from src.utils.tokenizer import smi_tokenizer
from rdkit import RDLogger
from rdkit import Chem
import math
import re
import random
import pandas as pd

RDLogger.DisableLog("rdApp.*")


def num_in_list(tokens):
    """
    Create sequences with bond already exists error.
    """
    symbols = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    nums = set()
    for a in tokens:
        if a in symbols:
            nums.add(a)
    return list(nums)


def exists_error(smi: str) -> list:
    """
    Create sequences with bond already exists error.
    """
    tokens = smi_tokenizer(smi)
    random_value = random.random()
    nums = num_in_list(tokens)
    if len(nums) > 0:
        avail_nums = nums
        i = 0
        while i < len(nums):
            # get the value of ring closure symbol that will be altered
            num = random.choice(tuple(avail_nums))
            locs = [i for i, e in enumerate(tokens) if e == num]
            # check if ring token has already been removed
            if (len(locs) % 2) == 0:
                index = random.choice(range(0, len(locs), 2))
                break
            else:
                # one ring token has already been removed try for other
                avail_nums.remove(num)
                i += 1
        if 'index' in locals():
            if random_value < 0.67:
                random_removal = random.random()
                # removes ring closure symbol
                if random_removal < 0.5:
                    tokens.pop(locs[index + 1])

                # insert ring closure symbol two tokens after ring opening symbol, does not work when after ring opening come brackets (results in parse error)
                # add closing symbol one position after last syntax token
                # if there are tokens after ring symbol
                j = 1
                try:
                    while tokens[locs[index] + j] in ["(", "=", "#", "/", "-"]:
                        j += 1
                except IndexError:
                    pass
                tokens.insert(locs[index] + j + 1, num)
            else:
                # inserts ring opening and closure symbols to create double ring
                # closure and opening symbols for same atoms
                if num == "1":
                    random_num = 2
                else:
                    random_num = int(num) + random.choice([-1, 1])
                tokens.insert(locs[index] + 1, str(random_num))
                random_loc = random.random()
                if random_loc < 0.5:
                    tokens.insert(locs[index + 1] + 1, str(random_num))
                else:
                    tokens.insert(locs[index + 1] + 2, str(random_num))

    return "".join(tokens[:])


def par_error(smi: str) -> list:
    """
    Tokenize a SMILES molecule
    """
    tokens = smi_tokenizer(smi)
    random_value = random.random()
    if random_value < 0.2:
        # randomly add parenthesis
        # from 1 because at zero gives syntax error, if '(' inserted at
        # len(tokens) than also syntax error
        tokens.insert(random.randrange(1,
                                       len(tokens) + 1),
                      random.choice(["(", ")"]))
    elif random_value < 0.4:
        # randomly remove parenthesis
        index = [i for i, e in enumerate(tokens) if e == "(" or e == ")"]
        if index:
            tokens.pop(random.choice(index))
    elif random_value < 0.6:
        # switch pair of parentheses
        # does not always give error
        opening = [i for i, e in enumerate(tokens) if e == "("]
        closing = [i for i, e in enumerate(tokens) if e == ")"]
        if opening and closing:
            tokens[random.choice(opening)] = ")"
            tokens[random.choice(closing)] = "("
    elif random_value < 0.8:
        # change '(' into ')'
        opening = [i for i, e in enumerate(tokens) if e == "("]
        if opening:
            tokens[random.choice(opening)] = ")"
    elif random_value < 1:
        # change ')' into '('
        closing = [i for i, e in enumerate(tokens) if e == ")"]
        if closing:
            tokens[random.choice(closing)] = "("

    return "".join(tokens[:])


def permutation(smi: str, vocab) -> list:
    """
    Tokenize a SMILES molecule
    """
    tokens = smi_tokenizer(smi)
    random_value = random.random()
    if random_value < 0.17:
        # randomly remove one token
        tokens.pop(random.randrange(0, len(tokens)))
    elif random_value < 0.33:
        # insert at random location
        tokens.insert(random.randrange(0,
                                       len(tokens) + 1), random.choice(vocab))
    elif random_value < 0.5:
        # replace at random location, can lead to token being replaced by
        # itself
        tokens[random.randrange(0, len(tokens))] = random.choice(vocab)
        while smi == "".join(tokens[:]):
            tokens[random.randrange(0, len(tokens))] = random.choice(vocab)
    elif random_value < 0.66:
        i = 0
        try:
            # start could be any token from 0 to one that will pop 2 before the
            # end of the sequence
            start = random.randrange(0, len(tokens) - 1)
            # length of stretch
            stop = random.randrange(2, int(round(len(tokens) / 4) + 1))
            # randomly remove stretch that is max 25% of tokens
            while i in range(0, stop) and i + start <= len(tokens):
                tokens.pop(start)
                i = i + 1
        except ValueError:
            pass
    elif random_value < 0.83:
        # insert at random location, can also be after string has ended
        location = random.randrange(0, len(tokens) + 1)
        # chooses how many tokens to add, max length is 25%
        try:
            stop = random.randrange(2, int(round(len(tokens) / 4) + 1))
            for i in range(stop):
                tokens.insert(location, random.choice(vocab))
        except ValueError:
            pass
    else:
        # replace at random location, can lead to token being replaced by
        # itself
        try:
            start = random.randrange(0, len(tokens) - 1)
            # chooses how many tokens to add, (length: 2 - end of sequence)
            stop = start + random.randrange(2, int(round(len(tokens) / 4) + 1))
            while start in range(start, stop) and start < len(tokens):
                tokens[start] = random.choice(vocab)
                start = start + 1
        except ValueError:
            pass

    return "".join(tokens[:]), tokens


def ring_error(smi: str) -> list:
    """
    Create sequences with unclosed ring or duplicated ring opening symbol
    """
    tokens = smi_tokenizer(smi)
    nums = num_in_list(tokens)
    if len(nums) > 0:
        random_value = random.random()
        if random_value < 0.5:
            # get the value of ring symbol that will be altered
            num = random.choice(tuple(nums))
            locs = [i for i, e in enumerate(tokens) if e == num]
            loc = random.choice(locs)
            # removes chosen ring symbol
            tokens.pop(loc)
            # half of the time replaces removed ring symbol with different
            # number
            random_rep = random.random()
            if random_rep < 0.25:
                # replaces by 1 higher or lower
                if num == "1":
                    tokens.insert(loc, "2")
                else:
                    tokens.insert(loc, str(int(num) + random.choice([-1, 1])))
            elif random_rep < 0.5:
                # replaces by any of ring symbols that are used in the sequence
                # (+1)
                nums.append(str(len(nums) + 1))
                tokens.insert(loc, random.choice(nums))
                while smi == "".join(tokens[:]):
                    tokens.insert(loc, random.choice(nums))
        elif random_value < 0.75:
            # insert ring symbol
            nums.append(str(len(nums) + 1))
            loc = random.randrange(0, len(tokens))
            tokens[loc] = random.choice(nums)
        else:
            # duplicate ring opening symbol
            num = random.choice(tuple(nums))
            locs = [i for i, e in enumerate(tokens) if e == num]
            index = random.choice(range(0, len(locs), 2))
            tokens.insert(locs[index], num)

    return "".join(tokens[:])


def syntax_error(smi: str) -> list:
    """
    Create sequences with syntax error.
    """
    tokens = smi_tokenizer(smi)
    syn_sym = ["=", "#", "-", "(", ")"]
    # could also only use numbers present in the sequence
    random_value = random.random()
    if random_value < 0.1:
        # sequence starts with syntax token
        tokens.insert(0, random.choice(syn_sym))
    elif random_value < 0.2:
        # sequence ends with bond symbol or '('
        tokens.insert(len(tokens), random.choice(syn_sym[:4]))
    elif random_value < 0.3:
        locs = [i for i, e in enumerate(tokens) if e in syn_sym[:3]]
        try:
            # insert bond token before or after existing symbol token
            tokens.insert(
                random.choice(locs) + random.randint(0, 1),
                random.choice(syn_sym[:3]))
        except BaseException:
            pass
    elif random_value < 0.4:
        locs = [i for i, e in enumerate(tokens) if e == "("]
        try:
            # insert bond token before ring opening
            tokens.insert(random.choice(locs), random.choice(syn_sym[:3]))
        except BaseException:
            pass
    elif random_value < 0.6:
        # create ((
        locs = [i for i, e in enumerate(tokens) if e == "("]
        if locs:
            loc = random.choice(locs)
            tokens.insert(loc, "(")
            random_deletion = random.random()
            if random_deletion < 0.5:
                try:
                    locs2 = [i for i in locs if i > loc]
                    tokens.pop(random.choice(locs2) + 1)
                except BaseException:
                    pass
    else:
        nums = num_in_list(tokens)

    if 0.6 <= random_value < 0.8:
        # delete tokens before ring symbol, can only be done before the first ring symbol
        random_choice = random.random()
        try:
            ring_index = tokens.index("1")
        except ValueError:
            ring_index = 100
        if random_choice < 0.5 and ring_index < 5:
            tokens = tokens[ring_index:]
        # delete tokens inbetween opening brackets and ring symbol
        else:
            # search for pattern
            # find any substring that starts with a opening parenthesis, followed by 1 to 3 characters of which none is a closing parenthesis and then any number
            # randomly chose one fragment
            # delete characters inbetween opening par and number

            pattern = r"\([^\)\(\\\/]{1,3}[1-9]"
            string = smi
            matches = re.findall(pattern, string)
            # would give error if any other characters are used that are syntax
            # tokens for re module
            if matches:
                match = f"\\{random.choice(matches)}"
                replace = f"({match[-1]}"
                try:
                    tokens = re.sub(match, replace, string, count=1)
                except BaseException:
                    pass

    elif 0.8 < random_value < 1:
        # create empty brackets and half of the time fill with bond symbol or
        # ring symbol
        opening = [i for i, e in enumerate(tokens) if e == "("]
        closing = [i for i, e in enumerate(tokens) if e == ")"]
        random_insertion = random.random()
        i = 0
        if opening:
            if len(opening) == len(closing):
                while i < 5:
                    index = random.randint(0, len(opening) - 1)
                    i = +1
                    if index + 1 == len(opening):
                        tokens = tokens[:opening[index] +
                                        1] + tokens[closing[index]:]
                    elif closing[index] < opening[index + 1]:
                        tokens = tokens[:opening[index] +
                                        1] + tokens[closing[index]:]
                    # for brackets that open and close between brackets
                    elif index + 2 == len(opening):
                        tokens = tokens[:opening[index] +
                                        1] + tokens[closing[index + 1]:]
                    elif (closing[index] > opening[index + 1]
                          and closing[index + 1] < opening[index + 2]):
                        tokens = tokens[:opening[index] +
                                        1] + tokens[closing[index + 1]:]
                    else:
                        continue
                    if random_insertion < 0.5:
                        tokens.insert(opening[index] + 1,
                                      str(random.choice(syn_sym[:3] + nums)))
                    break
    return "".join(tokens[:])


def valence_error(smiles, fragment):
    """adds a fragment to an atom with a full valence, with a single, double or triple type bond
    the correct smile is changed to the correct smile + fragment, seperated by dot
    or changes bond order to be 1 or 2 orders higher
    """
    # get editable mol file of both core and fragment
    core = Chem.MolFromSmiles(smiles)
    corfrag = smiles
    random_value = random.random()
    if random_value < 0.5:
        # add fragment
        frag = Chem.MolFromSmiles(fragment)
        combo = Chem.CombineMols(core, frag)
        edcombo = Chem.EditableMol(combo)
        # get all matches of aliphatic atoms without implicit hydrogens
        smarts = "[A!h]"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        # when there are matches
        if str(match) != "()":
            # get a string that contains the correct smiles and fragment
            # together, seperated by .
            new_corfrag = smiles + "." + fragment
            if Chem.MolFromSmiles(new_corfrag) is not None:
                corfrag = new_corfrag
            # get an atom number for an atom with full valence without implicit
            # hydrogens
            core_num = str(random.choice(match))
            core_num = int(re.sub(r"(\(|,\))", "", core_num))
            frag_num = core.GetNumAtoms() + random.randrange(
                0, frag.GetNumAtoms())
            # combine with either single, double or triple bond
            bond_order = random.choice([
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
            ])
            edcombo.AddBond(core_num, frag_num, order=bond_order)
            back = edcombo.GetMol()
            try:
                smiles = Chem.MolToSmiles(back)
            except BaseException:
                pass
    else:
        # increase bond order
        # get single bond full atoms
        smarts = "[A!h]-,=*"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if str(match) != "()":
            center_num, neigh_num = random.choice(match)
            bond = core.GetBondBetweenAtoms(center_num, neigh_num)
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                bond_order = random.choice(
                    [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE])
                bond.SetBondType(bond_order)
            else:
                bond.SetBondType(Chem.BondType.TRIPLE)
            try:
                smiles = Chem.MolToSmiles(core)
            except BaseException:
                pass

    return smiles, corfrag


def arom_error(smiles, fragment, use_mol=True):
    """adds a fragment to an atom with a full valence, with single, double or triple type bond
    the correct smile is changed to the correct smile + fragment, seperated by dot
    or changes bond order to be 1 or 2 orders higher
    """
    if use_mol:
        random_value = random.random()
    else:
        random_value = random.uniform(0.5, 1)

    corfrag = smiles
    if random_value < 0.5:
        # get mol file of core
        core = Chem.MolFromSmiles(smiles)
    else:
        tokens = smi_tokenizer(smiles)

    if random_value < 0.16:
        # increase bond order
        smarts = "c-[Ah!H1]"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        # turn one bond into a double bond if matches are found
        if str(match) != "()":
            # get id's of a random match
            center_num, neigh_num = random.choice(match)
            bond = core.GetBondBetweenAtoms(center_num, neigh_num)
            # make single bond into double bond
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                bond.SetBondType(Chem.BondType.DOUBLE)
                try:
                    smiles = Chem.MolToSmiles(core)
                except BaseException:
                    pass
    elif random_value < 0.33:
        # add fragment
        frag = Chem.MolFromSmiles(fragment)
        combo = Chem.CombineMols(core, frag)
        edcombo = Chem.EditableMol(combo)
        # check for aromatic carbons without implicit hydrogens
        smarts = "[c!h,nD2]"
        match = core.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if str(match) != "()":
            # get an atom number for an atom with full valence without implicit
            # hydrogens
            try:
                # get a string that contains the correct smiles and fragment
                # together, seperated by .
                new_corfrag = smiles + "." + fragment
                if Chem.MolFromSmiles(new_corfrag) is not None:
                    corfrag = new_corfrag
                core_num = str(random.choice(match))
                core_num = int(re.sub(r"(\(|,\))", "", core_num))
                frag_num = core.GetNumAtoms()
                # combine with a single bond
                edcombo.AddBond(core_num,
                                frag_num,
                                order=Chem.rdchem.BondType.SINGLE)
                back = edcombo.GetMol()
                try:
                    smiles = Chem.MolToSmiles(back)
                except BaseException:
                    pass
            except IndexError:
                pass
    elif random_value < 0.5:
        mol = Chem.rdchem.Mol(core.ToBinary())
        mol = AdjustAromaticNs(mol)
        try:
            smiles = Chem.MolToSmiles(mol)
        except BaseException:
            pass

    elif random_value < 0.6:
        # turn uppercase into lowercase for non-aromatic outside of ring
        index = [
            i for i, e in enumerate(tokens)
            if e in ["C", "N", "O", "S", "P", "B"]
        ]
        if index:
            index = random.choice(index)
            tokens[index] = tokens[index].lower()
        smiles = "".join(tokens[:])
    elif random_value < 0.7:
        # changes aromatic atom with 1 pi electron to uppercase or different
        # aromatic atom
        index_c = [i for i, e in enumerate(tokens) if e == "c"]
        index_n = [i for i, e in enumerate(tokens) if e == "n"]
        # remove the ones that are followed up by '('
        # common atomatic atoms with 2 or 0 pi electrons, not an comprehensive list
        # does not exclude An(A)a so will not always lead to mistake
        options = [
            "[nH]",
            ["n", "(", "C", ")"],
            "o",
            "s",
            ["c", "(", "=", "O", ")"],
            ["c", "(", "=", "N", ")"],
            "C",
            "N",
            ["c", "c"],
            ["c", "n"],
            ["n", "c"],
            ["n", "n"],
        ]
        index = random.choice([index_c, index_n])
        if index:
            index = random.choice(index)
            tokens.pop(index)
            tokens[index:index] = random.choice(options)
        smiles = "".join(tokens[:])
    elif random_value < 0.8:
        # insert c or n into aromatic ring
        index = [i for i, e in enumerate(tokens) if e.islower()]
        if index:
            index = random.choice(index) + random.choice([0, 1])
            tokens.insert(index, random.choice(["c", "n"]))
        smiles = "".join(tokens[:])
    elif random_value < 0.9:
        # change aromatic atom with 2 pi into c or n
        index = [i for i, e in enumerate(tokens) if e in ["[nH]", "o", "s"]]
        if index:
            index = random.choice(index)
            tokens[index] = random.choice(["c", "n"])
        smiles = "".join(tokens[:])
    elif random_value < 1:
        # removes the token of an aromatic atom with 1 pi electron
        index = [i for i, e in enumerate(tokens) if e == "c" or e == "n"]
        if index:
            index = random.choice(index)
            tokens.pop(index)
        smiles = "".join(tokens[:])

    return smiles, corfrag


def introduce_error(smile,
                    fragment,
                    vocab,
                    invalid_type: str = "all",
                    num_errors: int = 1):
    """Introduces errors into valid SMILES

    param invalid_type: type of errors
    param smile: original SMILES
    param fragment: is fragment that could be added to this SMILES

    return: invalid SMILES and original SMILES (or original SMILES.Fragment in case a fragment is added to invalid SMILES)
    """

    corfrag = smile
    i = 0

    while Chem.MolFromSmiles(smile) is not None and i < 20:
        i += 1
        try:
            if invalid_type == "all" or invalid_type == "multiple":
                random_value = random.choice(range(1, 8))
                if random_value == 1:
                    smile = exists_error(smile)
                elif random_value == 2:
                    smile = par_error(smile)
                elif random_value == 3:
                    smile, tokens = permutation(smile, list(vocab))
                    vocab.update(tokens)
                elif random_value == 4:
                    smile = ring_error(smile)
                elif random_value == 5:
                    smile = syntax_error(smile)
                elif random_value == 6:
                    smile, corfrag = valence_error(smile, fragment)
                elif random_value == 7:
                    smile, corfrag = arom_error(smile, fragment)

            elif invalid_type == "exists":
                smile = exists_error(smile)
            elif invalid_type == "par":
                smile = par_error(smile)
            elif invalid_type == "permut":
                smile, tokens = permutation(smile, list(vocab))
                vocab.update(tokens)
            elif invalid_type == "ring":
                smile = ring_error(smile)
            elif invalid_type == "syntax":
                smile = syntax_error(smile)
            elif invalid_type == "valence":
                smile, corfrag = valence_error(smile, fragment)
            elif invalid_type == "arom":
                smile, corfrag = arom_error(smile, fragment)
        except ValueError:
            pass

    if invalid_type == "multiple":
        for j in range(num_errors - 1):
            original_smile = smile
            while smile == original_smile and len(smile) > 0:
                random_value = random.choice(range(1, 7))
                if random_value == 1:
                    smile = exists_error(smile)
                elif random_value == 2:
                    smile = par_error(smile)
                elif random_value == 3:
                    smile, tokens = permutation(smile, list(vocab))
                    vocab.update(tokens)
                elif random_value == 4:
                    smile = ring_error(smile)
                elif random_value == 5:
                    smile = syntax_error(smile)
                elif random_value == 6:
                    smile, _ = arom_error(smile, fragment, use_mol=False)

    if Chem.MolFromSmiles(smile) is not None:
        smile = None

    return smile, corfrag


def get_invalid_smiles(df: pd.DataFrame,
                       df_frag: pd.DataFrame,
                       SEED: int,
                       invalid_type: str = "all",
                       num_errors: int = 1):
    """Get invalid SMILES

    param invalid_type: type of errors

    return: dataframe with original and generated, corresponding invalid SMILES
    """
    random.seed(SEED)
    invalid_options = [
        "all",
        "multiple",
        "exists",
        "par",
        "permut",
        "ring",
        "syntax",
        "valence",
        "arom",
    ]
    if invalid_type not in invalid_options:
        raise ValueError(
            f"Invalid type not supported, must be one of {invalid_options}")

    df = pd.DataFrame(df).reset_index(drop=True)
    # reorder fragments dataset and merge with molecule dataset so that each molecule has an associated fragment
    # associated fragments are used for creating aromaticity & valence errors
    df_frag = df_frag.apply(
        lambda x: x.sample(frac=1, random_state=SEED).values)

    new_len = math.ceil(len(df) / len(df_frag))

    if new_len > 1:
        df_frag = pd.concat([df_frag] * new_len,
                            ignore_index=True).iloc[:len(df)]
    else:
        df_frag = df_frag.iloc[:len(df)]

    df = df.merge(df_frag, how="outer", left_index=True, right_index=True)

    # create a vocabulary of SMILES tokens, used as random input by
    # permutation error
    vocab = set()

    print(df)

    for i, row in df.sample(min(50, df.shape[0])).iterrows():
        smile = row["STD_SMILES"]
        tokens = smi_tokenizer(smile)
        vocab.update(tokens)

    df["ERROR"], df["STD_SMILES"] = zip(*df.apply(
        lambda row: introduce_error(row["STD_SMILES"], row["FRAGMENT"], vocab,
                                    invalid_type, num_errors),
        axis=1,
    ))

    print(df.isna().sum())
    return df.dropna()
