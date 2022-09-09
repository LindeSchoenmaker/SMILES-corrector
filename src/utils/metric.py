import itertools
import time
import statistics
from rdkit.Chem import GraphDescriptors, Lipinski, AllChem
from rdkit.Chem.rdSLNParse import MolFromSLN
from rdkit.Chem.rdmolfiles import MolFromSmiles
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from rdkit import rdBase, Chem

rdBase.DisableLog("rdApp.error")


def is_smiles(array,
              TRG,
              reverse: bool,
              return_output=False,
              src=None,
              src_field=None):
    """Turns predicted tokens within batch into smiles and evaluates their validity
    Arguments:
        array: Tensor with most probable token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        return_output (bool): True if output sequences and their validity should be saved
    Returns:
        df: dataframe with correct and incorrect sequences
        valids: list with booleans that show if prediction was a valid SMILES (True) or invalid one (False)
        smiless: list of the predicted smiles
    """
    trg_field = TRG
    valids = []
    smiless = []
    if return_output:
        df = pd.DataFrame()
    else:
        df = None
    batch_size = array.size(1)
    # check if the first token should be removed, first token is zero because
    # outputs initaliazed to all be zeros
    if int((array[0, 0]).tolist()) == 0:
        start = 1
    else:
        start = 0
    # for each sequence in the batch
    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[start:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # print(trg_tokens)
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        # determine how many valid smiles are made
        valid = True if MolFromSmiles(smiles) else False
        valids.append(valid)
        smiless.append(smiles)
        if return_output:
            if valid:
                df.loc[i, "CORRECT"] = smiles
            else:
                df.loc[i, "INCORRECT"] = smiles

    # add the original drugex outputs to the _de dataframe
    if return_output and src is not None:
        for i in range(0, batch_size):
            # turns sequence from tensor to list skipps first row as this is
            # <sos> for src
            sequence = (src[1:, i]).tolist()
            # goes from embedded to tokens
            src_tokens = [src_field.vocab.itos[int(t)] for t in sequence]
            # takes all tokens untill eos token, model would be faster if did
            # this one step earlier, but then changes in vocab order would
            # disrupt.
            rev_tokens = list(
                itertools.takewhile(lambda x: x != "<eos>", src_tokens))
            smiles = "".join(rev_tokens)
            df.loc[i, "ORIGINAL"] = smiles

    return df, valids, smiless


def is_unchanged(array,
                 TRG,
                 reverse: bool,
                 return_output=False,
                 src=None,
                 src_field=None):
    """Checks is output is different from input
    Arguments:
        array: Tensor with most probable token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        return_output (bool): True if output sequences and their validity should be saved
    Returns:
        df: dataframe with correct and incorrect sequences
        valids: list with booleans that show if prediction was a valid SMILES (True) or invalid one (False)
        smiless: list of the predicted smiles
    """
    trg_field = TRG
    sources = []
    batch_size = array.size(1)
    unchanged = 0

    # check if the first token should be removed, first token is zero because
    # outputs initaliazed to all be zeros
    if int((array[0, 0]).tolist()) == 0:
        start = 1
    else:
        start = 0

    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is <sos>
        # for src
        sequence = (src[1:, i]).tolist()
        # goes from embedded to tokens
        src_tokens = [src_field.vocab.itos[int(t)] for t in sequence]
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", src_tokens))
        smiles = "".join(rev_tokens)
        sources.append(smiles)

    # for each sequence in the batch
    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[start:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # print(trg_tokens)
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        # determine how many valid smiles are made
        valid = True if MolFromSmiles(smiles) else False
        if not valid:
            if smiles == sources[i]:
                unchanged += 1

    return unchanged


def molecule_reconstruction(array, TRG, reverse: bool, outputs):
    """Turns target tokens within batch into smiles and compares them to predicted output smiles
    Arguments:
        array: Tensor with target's token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        outputs: list of predicted SMILES sequences
    Returns:
         matches(int): number of total right molecules
    """
    trg_field = TRG
    matches = 0
    targets = []
    batch_size = array.size(1)
    # for each sequence in the batch
    for i in range(0, batch_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[1:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        targets.append(smiles)
    for i in range(0, batch_size):
        m = MolFromSmiles(targets[i])
        p = MolFromSmiles(outputs[i])
        if p is not None:
            if m.HasSubstructMatch(p) and p.HasSubstructMatch(m):
                matches += 1
    return matches


def complexity_whitlock(mol: Chem.Mol, includeAllDescs=False):
    """
    Complexity as defined in DOI:10.1021/jo9814546
    S: complexity = 4*#rings + 2*#unsat + #hetatm + 2*#chiral
    Other descriptors:
        H: size = #bonds (Hydrogen atoms included)
        G: S + H
        Ratio: S / H
    """
    mol_ = Chem.Mol(mol)
    nrings = Lipinski.RingCount(mol_) - Lipinski.NumAromaticRings(mol_)
    Chem.rdmolops.SetAromaticity(mol_)
    unsat = sum(1 for bond in mol_.GetBonds()
                if bond.GetBondTypeAsDouble() == 2)
    hetatm = len(mol_.GetSubstructMatches(Chem.MolFromSmarts("[!#6]")))
    AllChem.EmbedMolecule(mol_)
    Chem.AssignAtomChiralTagsFromStructure(mol_)
    chiral = len(Chem.FindMolChiralCenters(mol_))
    S = 4 * nrings + 2 * unsat + hetatm + 2 * chiral
    if not includeAllDescs:
        return S
    Chem.rdmolops.Kekulize(mol_)
    mol_ = Chem.AddHs(mol_)
    H = sum(bond.GetBondTypeAsDouble() for bond in mol_.GetBonds())
    G = S + H
    R = S / H
    return {"WhitlockS": S, "WhitlockH": H, "WhitlockG": G, "WhitlockRatio": R}


def complexity_baronechanon(mol: Chem.Mol):
    """
    Complexity as defined in DOI:10.1021/ci000145p
    """
    mol_ = Chem.Mol(mol)
    Chem.Kekulize(mol_)
    Chem.RemoveStereochemistry(mol_)
    mol_ = Chem.RemoveHs(mol_, updateExplicitCount=True)
    degree, counts = 0, 0
    for atom in mol_.GetAtoms():
        degree += 3 * 2**(atom.GetExplicitValence() - atom.GetNumExplicitHs() -
                          1)
        counts += 3 if atom.GetSymbol() == "C" else 6
    ringterm = sum(map(lambda x: 6 * len(x), mol_.GetRingInfo().AtomRings()))
    return degree + counts + ringterm


def calc_complexity(array,
                    TRG,
                    reverse,
                    valids,
                    complexity_function=GraphDescriptors.BertzCT):
    """Calculates the complexity of inputs that are not correct.
    Arguments:
        array: Tensor with target's token for each location for each sequence in batch
            [trg len, batch size]
        TRG: target field for getting tokens from vocab
        reverse (bool): True if the target sequence is reversed
        valids: list with booleans that show if prediction was a valid SMILES (True) or invalid one (False)
        complexity_function: the type of complexity measure that will be used
            GraphDescriptors.BertzCT
            complexity_whitlock
            complexity_baronechanon
    Returns:
         matches(int): mean of complexity values
    """
    trg_field = TRG
    sources = []
    complexities = []
    loc = torch.BoolTensor(valids)
    # only keeps rows in batch size dimension where valid is false
    array = array[:, loc == False]
    # should check if this still works
    # array = torch.transpose(array, 0, 1)
    array_size = array.size(1)
    for i in range(0, array_size):
        # turns sequence from tensor to list skipps first row as this is not
        # filled in in forward
        sequence = (array[1:, i]).tolist()
        # goes from embedded to tokens
        trg_tokens = [trg_field.vocab.itos[int(t)] for t in sequence]
        # takes all tokens untill eos token, model would be faster if did this
        # one step earlier, but then changes in vocab order would disrupt.
        rev_tokens = list(
            itertools.takewhile(lambda x: x != "<eos>", trg_tokens))
        if reverse:
            rev_tokens = rev_tokens[::-1]
        smiles = "".join(rev_tokens)
        sources.append(smiles)
    for source in sources:
        try:
            m = MolFromSmiles(source)
        except BaseException:
            m = MolFromSLN(source)
        complexities.append(complexity_function(m))
    if len(complexities) > 0:
        mean = statistics.mean(complexities)
    else:
        mean = 0
    return mean


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Recur:
    """Class for training and evaluating recurrent neural network
    
    Methods
    -------
    train_model()
        train model for initialized number of epochs
    evaluate(return_output)
        use model with validation loader (& optionally drugex loader) to get test loss & other metrics
    """

    def train_model(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        log = open(f"{self.out}.log", "a")
        best_error = np.inf
        last_save = 0
        for epoch in range(self.epochs):
            self.train()
            start_time = time.time()
            loss_train = 0
            for i, batch in enumerate(self.loader_train):
                optimizer.zero_grad()
                # changed src,trg call to match with bentrevett
                # src, trg = batch['src'], batch['trg']
                trg = batch.trg
                src, src_len = batch.src
                output, trg_index = self(src, src_len, trg, device=self.device)
                # feed the source and target into def forward to get the output
                # Xuhan uses forward for this, with istrain = true
                output_dim = output.shape[-1]
                # changed
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                # output = output[:,:,0]#.view(-1)
                # output = output[1:].view(-1, output.shape[-1])
                # trg = trg[1:].view(-1)
                loss = nn.CrossEntropyLoss(
                    ignore_index=self.TRG.vocab.stoi[self.TRG.pad_token])
                a, b = output.view(-1), trg.to(self.device).view(-1)
                # changed
                # loss = loss(output.view(0), trg.view(0).to(device))
                loss = loss(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                optimizer.step()
                loss_train += loss.item()
                # turned off for now, as not using voc so won't work, output is a tensor
                # output = [(trg len - 1) * batch size, output dim]
                # smiles, valid = is_valid_smiles(output, reversed)
                # if valid:
                #    valids += 1
                #    smiless.append(smiles)
            # added .dataset becaue len(iterator) gives len(self.dataset) /
            # self.batch_size)
            loss_train /= len(self.loader_train)
            info = f"Epoch: {epoch+1:02} step: {i} loss_train: {loss_train:.4g}"
            # model is used to generate trg based on src from the validation set to assess performance
            # similar to Xuhan, although he doesn't use the if loop
            if self.loader_valid is not None:
                return_output = False
                if epoch + 1 == self.epochs:
                    return_output = True
                (
                    valids,
                    loss_valid,
                    valids_de,
                    df_output,
                    df_output_de,
                    right_molecules,
                    complexity,
                ) = self.evaluate(return_output)
                reconstruction_error = 1 - right_molecules / len(
                    self.loader_valid.dataset)
                error = 1 - valids / len(self.loader_valid.dataset)
                complexity = complexity / len(self.loader_valid)
                info += f" loss_valid: {loss_valid:.4g} error_rate: {error:.4g} molecule_reconstruction_error_rate: {reconstruction_error:.4g} invalid_target_complexity: {complexity:.4g}"
                if self.loader_drugex is not None:
                    error_de = 1 - valids_de / len(self.loader_drugex.dataset)
                    info += f" error_rate_drugex: {error_de:.4g}"
                if reconstruction_error < best_error:
                    torch.save(self.state_dict(), f"{self.out}.pkg")
                    best_error = reconstruction_error
                    last_save = epoch
                else:
                    if epoch - last_save > 10:
                        (
                            valids,
                            loss_valid,
                            valids_de,
                            df_output,
                            df_output_de,
                            right_molecules,
                            complexity,
                        ) = self.evaluate(True)
                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(
                            start_time, end_time)
                        info += f" Time: {epoch_mins}m {epoch_secs}s"
                        print(info, file=log, flush=True)
                        print(info)
                        break
            elif error < best_error:
                torch.save(self.state_dict(), f"{self.out}.pkg")
                best_error = error
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            info += f" Time: {epoch_mins}m {epoch_secs}s"
            print(info, file=log, flush=True)
            print(info)
            # for i, smiles in enumerate(smiless):
            #     print(f'Valid SMILES {i}\t{smiles}', file=log)
        log.close()
        self.load_state_dict(torch.load(f"{self.out}.pkg"))
        df_output.to_csv(f"{self.out}.csv", index=False)
        df_output_de.to_csv(f"{self.out}_de.csv", index=False)

    def evaluate(self, return_output):
        self.eval()
        test_loss = 0
        df_output = pd.DataFrame()
        df_output_de = pd.DataFrame()
        valids = 0
        valids_de = 0

        right_molecules = 0
        complexity = 0
        with torch.no_grad():
            for _, batch in enumerate(self.loader_valid):
                src, src_len = batch.src
                trg = batch.trg
                output, trg_index = self.forward(src, src_len, trg,
                                                 0)  # turn off teacher forcing
                # checks the number of valid smiles
                df_batch, valid, smiless = is_smiles(
                    trg_index,
                    self.TRG,
                    reverse=True,
                    return_output=return_output)
                matches = molecule_reconstruction(trg,
                                                  self.TRG,
                                                  reverse=True,
                                                  outputs=smiless)
                right_molecules += matches
                complexity += calc_complexity(trg,
                                              self.TRG,
                                              reverse=True,
                                              valids=valid)
                # add new dataframe to existing one
                if df_batch is not None:
                    df_output = pd.concat([df_output, df_batch],
                                          ignore_index=True)
                valids += sum(valid)
                # changed
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = nn.CrossEntropyLoss(
                    ignore_index=self.TRG.vocab.stoi[self.TRG.pad_token])
                loss = loss(output, trg)
            test_loss += loss.item()
            if self.loader_drugex is not None:
                for _, batch in enumerate(self.loader_drugex):
                    # trg = batch.trg
                    src, src_len = batch.src
                    output, trg_index = self(src, src_len, None, 0)
                    # checks the number of valid smiles
                    df_batch, valid, smiless = is_smiles(
                        trg_index,
                        self.TRG,
                        reverse=True,
                        return_output=return_output)
                    valids_de += sum(valid)
                    if df_batch is not None:
                        df_output_de = pd.concat([df_output_de, df_batch],
                                                 ignore_index=True)
        return (
            valids,
            test_loss / len(self.loader_valid),
            valids_de,
            df_output,
            df_output_de,
            right_molecules,
            complexity,
        )


class Convo:
    """Class for training and evaluating transformer and convolutional neural network
    
    Methods
    -------
    train_model()
        train model for initialized number of epochs
    evaluate(return_output)
        use model with validation loader (& optionally drugex loader) to get test loss & other metrics
    translate(loader)
        translate inputs from loader (different from evaluate in that no target sequence is used)
    """

    def train_model(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        log = open(f"{self.out}.log", "a")
        best_error = np.inf
        for epoch in range(self.epochs):
            self.train()
            start_time = time.time()
            loss_train = 0
            for i, batch in enumerate(self.loader_train):
                optimizer.zero_grad()
                # changed src,trg call to match with bentrevett
                # src, trg = batch['src'], batch['trg']
                trg = batch.trg
                src = batch.src
                output, attention = self(src, trg[:, :-1])
                # feed the source and target into def forward to get the output
                # Xuhan uses forward for this, with istrain = true
                output_dim = output.shape[-1]
                # changed
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                # output = output[:,:,0]#.view(-1)
                # output = output[1:].view(-1, output.shape[-1])
                # trg = trg[1:].view(-1)
                loss = nn.CrossEntropyLoss(
                    ignore_index=self.TRG.vocab.stoi[self.TRG.pad_token])
                a, b = output.view(-1), trg.to(self.device).view(-1)
                # changed
                # loss = loss(output.view(0), trg.view(0).to(device))
                loss = loss(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                optimizer.step()
                loss_train += loss.item()
                # turned off for now, as not using voc so won't work, output is a tensor
                # output = [(trg len - 1) * batch size, output dim]
                # smiles, valid = is_valid_smiles(output, reversed)
                # if valid:
                #    valids += 1
                #    smiless.append(smiles)
            # added .dataset becaue len(iterator) gives len(self.dataset) /
            # self.batch_size)
            loss_train /= len(self.loader_train)
            info = f"Epoch: {epoch+1:02} step: {i} loss_train: {loss_train:.4g}"
            # model is used to generate trg based on src from the validation set to assess performance
            # similar to Xuhan, although he doesn't use the if loop
            if self.loader_valid is not None:
                return_output = False
                if epoch + 1 == self.epochs:
                    return_output = True
                (
                    valids,
                    loss_valid,
                    valids_de,
                    df_output,
                    df_output_de,
                    right_molecules,
                    complexity,
                    unchanged,
                    unchanged_de,
                ) = self.evaluate(return_output)
                reconstruction_error = 1 - right_molecules / len(
                    self.loader_valid.dataset)
                error = 1 - valids / len(self.loader_valid.dataset)
                complexity = complexity / len(self.loader_valid)
                unchan = unchanged / (len(self.loader_valid.dataset) - valids)
                info += f" loss_valid: {loss_valid:.4g} error_rate: {error:.4g} molecule_reconstruction_error_rate: {reconstruction_error:.4g} unchanged: {unchan:.4g} invalid_target_complexity: {complexity:.4g}"
                if self.loader_drugex is not None:
                    error_de = 1 - valids_de / len(self.loader_drugex.dataset)
                    unchan_de = unchanged_de / (
                        len(self.loader_drugex.dataset) - valids_de)
                    info += f" error_rate_drugex: {error_de:.4g} unchanged_drugex: {unchan_de:.4g}"

                if reconstruction_error < best_error:
                    torch.save(self.state_dict(), f"{self.out}.pkg")
                    best_error = reconstruction_error
                    last_save = epoch
                else:
                    if epoch - last_save >= 10 and best_error != 1:
                        torch.save(self.state_dict(), f"{self.out}_last.pkg")
                        (
                            valids,
                            loss_valid,
                            valids_de,
                            df_output,
                            df_output_de,
                            right_molecules,
                            complexity,
                            unchanged,
                            unchanged_de,
                        ) = self.evaluate(True)
                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(
                            start_time, end_time)
                        info += f" Time: {epoch_mins}m {epoch_secs}s"
                        print(info, file=log, flush=True)
                        print(info)
                        break
            elif error < best_error:
                torch.save(self.state_dict(), f"{self.out}.pkg")
                best_error = error
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            info += f" Time: {epoch_mins}m {epoch_secs}s"
            print(info, file=log, flush=True)
            print(info)
            # for i, smiles in enumerate(smiless):
            #     print(f'Valid SMILES {i}\t{smiles}', file=log)
        torch.save(self.state_dict(), f"{self.out}_last.pkg")
        log.close()
        self.load_state_dict(torch.load(f"{self.out}.pkg"))
        df_output.to_csv(f"{self.out}.csv", index=False)
        df_output_de.to_csv(f"{self.out}_de.csv", index=False)

    def evaluate(self, return_output):
        self.eval()
        test_loss = 0
        df_output = pd.DataFrame()
        df_output_de = pd.DataFrame()
        valids = 0
        valids_de = 0
        unchanged = 0
        unchanged_de = 0
        right_molecules = 0
        complexity = 0
        with torch.no_grad():
            for _, batch in enumerate(self.loader_valid):
                trg = batch.trg
                src = batch.src
                output, attention = self.forward(src, trg[:, :-1])
                pred_token = output.argmax(2)
                array = torch.transpose(pred_token, 0, 1)
                trg_trans = torch.transpose(trg, 0, 1)
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                src_trans = torch.transpose(src, 0, 1)
                df_batch, valid, smiless = is_smiles(
                    array, self.TRG, reverse=True, return_output=return_output)
                unchanged += is_unchanged(
                    array,
                    self.TRG,
                    reverse=True,
                    return_output=return_output,
                    src=src_trans,
                    src_field=self.SRC,
                )
                matches = molecule_reconstruction(trg_trans,
                                                  self.TRG,
                                                  reverse=True,
                                                  outputs=smiless)
                complexity += calc_complexity(trg_trans,
                                              self.TRG,
                                              reverse=True,
                                              valids=valid)
                if df_batch is not None:
                    df_output = pd.concat([df_output, df_batch],
                                          ignore_index=True)
                right_molecules += matches
                valids += sum(valid)
                # trg = trg[1:].view(-1)
                # output, trg = output[1:].view(-1, output.shape[-1]), trg[1:].view(-1)
                loss = nn.CrossEntropyLoss(
                    ignore_index=self.TRG.vocab.stoi[self.TRG.pad_token])
                loss = loss(output, trg)
            test_loss += loss.item()
            if self.loader_drugex is not None:
                for _, batch in enumerate(self.loader_drugex):
                    src = batch.src
                    output = self.translate_sentence(src, self.TRG,
                                                     self.device)
                    # checks the number of valid smiles
                    pred_token = output.argmax(2)
                    array = torch.transpose(pred_token, 0, 1)
                    src_trans = torch.transpose(src, 0, 1)
                    df_batch, valid, smiless = is_smiles(
                        array,
                        self.TRG,
                        reverse=True,
                        return_output=return_output,
                        src=src_trans,
                        src_field=self.SRC,
                    )
                    unchanged_de += is_unchanged(
                        array,
                        self.TRG,
                        reverse=True,
                        return_output=return_output,
                        src=src_trans,
                        src_field=self.SRC,
                    )
                    if df_batch is not None:
                        df_output_de = pd.concat([df_output_de, df_batch],
                                                 ignore_index=True)
                    valids_de += sum(valid)
        return (
            valids,
            test_loss / len(self.loader_valid),
            valids_de,
            df_output,
            df_output_de,
            right_molecules,
            complexity,
            unchanged,
            unchanged_de,
        )

    def translate(self, loader):
        self.eval()
        df_output_de = pd.DataFrame()
        valids_de = 0
        with torch.no_grad():
            for _, batch in enumerate(loader):
                src = batch.src
                output = self.translate_sentence(src, self.TRG, self.device)
                # checks the number of valid smiles
                pred_token = output.argmax(2)
                array = torch.transpose(pred_token, 0, 1)
                src_trans = torch.transpose(src, 0, 1)
                df_batch, valid, smiless = is_smiles(
                    array,
                    self.TRG,
                    reverse=True,
                    return_output=True,
                    src=src_trans,
                    src_field=self.SRC,
                )
                if df_batch is not None:
                    df_output_de = pd.concat([df_output_de, df_batch],
                                             ignore_index=True)
                valids_de += sum(valid)
        return valids_de, df_output_de
