import torch
import torch.nn as nn
from torchtext.legacy.data import TabularDataset, Field, BucketIterator, Iterator

import random
import os

from src.utils.tokenizer import smi_tokenizer
from src.transformer import Encoder, Decoder, Seq2Seq


def init_weights(m: nn.Module):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def initialize_model(folder_out: str,
                     data_source: str,
                     error_source: str,
                     device: torch.device,
                     threshold: int,
                     epochs: int,
                     layers: int = 3,
                     batch_size: int = 16,
                     invalid_type: str = "all",
                     num_errors: int = 1,
                     validation_step=True):
    """Create encoder decoder models for specified model (currently only translator) & type of invalid SMILES

    param data: collection of invalid, valid SMILES pairs
    param invalid_smiles_path: path to previously generated invalid SMILES
    param invalid_type: type of errors introduced into invalid SMILES

    return:

    """

    # set fields
    SRC = Field(
        tokenize=lambda x: smi_tokenizer(x),
        init_token="<sos>",
        eos_token="<eos>",
        batch_first=True,
    )
    TRG = Field(
        tokenize=lambda x: smi_tokenizer(x, reverse=True),
        init_token="<sos>",
        eos_token="<eos>",
        batch_first=True,
    )

    if validation_step:
        train, val = TabularDataset.splits(
            path=f'{folder_out}errors/split/',
            train=f"{data_source}_{invalid_type}_{num_errors}_errors_train.csv",
            validation=
            f"{data_source}_{invalid_type}_{num_errors}_errors_dev.csv",
            format="CSV",
            skip_header=False,
            fields={
                "ERROR": ("src", SRC),
                "STD_SMILES": ("trg", TRG)
            },
        )
        SRC.build_vocab(train, val, max_size=1000)
        TRG.build_vocab(train, val, max_size=1000)
    else:
        train = TabularDataset(
            path=
            f'{folder_out}errors/{data_source}_{invalid_type}_{num_errors}_errors.csv',
            format="CSV",
            skip_header=False,
            fields={
                "ERROR": ("src", SRC),
                "STD_SMILES": ("trg", TRG)
            },
        )
        SRC.build_vocab(train, max_size=1000)
        TRG.build_vocab(train, max_size=1000)

    drugex = TabularDataset(
        path=error_source,
        format="csv",
        skip_header=False,
        fields={
            "SMILES": ("src", SRC),
            "SMILES_TARGET": ("trg", TRG)
        },
    )
    print(vars(random.choice(train.examples)))

    print(len(SRC.vocab))
    print(len(TRG.vocab))

    #SRC.vocab = torch.load('vocab_src.pth')
    #TRG.vocab = torch.load('vocab_trg.pth')

    # model parameters
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = layers
    DEC_LAYERS = layers
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    # add 2 to length for start and stop tokens
    MAX_LENGTH = threshold + 2

    # model name
    MODEL_OUT_FOLDER = f"{folder_out}performance/"

    MODEL_NAME = "transformer_%s_%s_%s_%s_%s" % (
        invalid_type, num_errors, data_source, BATCH_SIZE, layers)
    if not os.path.exists(MODEL_OUT_FOLDER):
        os.mkdir(MODEL_OUT_FOLDER)

    out = os.path.join(MODEL_OUT_FOLDER, MODEL_NAME)

    torch.save(SRC.vocab, f'{out}_vocab_src.pth')
    torch.save(TRG.vocab, f'{out}_vocab_trg.pth')

    # iterator is a dataloader
    # iterator to pass to the same length and create batches in which the
    # amount of padding is minimized
    if validation_step:
        train_iter, val_iter = BucketIterator.splits(
            (train, val),
            batch_sizes=(BATCH_SIZE, 256),
            sort_within_batch=True,
            shuffle=True,
            # the BucketIterator needs to be told what function it should use to
            # group the data.
            sort_key=lambda x: len(x.src),
            device=device,
        )
    else:
        train_iter = BucketIterator(
            train,
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            shuffle=True,
            # the BucketIterator needs to be told what function it should use to
            # group the data.
            sort_key=lambda x: len(x.src),
            device=device,
        )
        val_iter = None

    drugex_iter = Iterator(
        drugex,
        batch_size=64,
        device=device,
        sort=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
    )
    print(len(drugex_iter.dataset))

    # model initialization

    enc = Encoder(
        INPUT_DIM,
        HID_DIM,
        ENC_LAYERS,
        ENC_HEADS,
        ENC_PF_DIM,
        ENC_DROPOUT,
        MAX_LENGTH,
        device,
    )
    dec = Decoder(
        OUTPUT_DIM,
        HID_DIM,
        DEC_LAYERS,
        DEC_HEADS,
        DEC_PF_DIM,
        DEC_DROPOUT,
        MAX_LENGTH,
        device,
    )

    model = Seq2Seq(
        enc,
        dec,
        SRC_PAD_IDX,
        TRG_PAD_IDX,
        device,
        train_iter,
        out=out,
        loader_valid=val_iter,
        loader_drugex=drugex_iter,
        epochs=EPOCHS,
        TRG=TRG,
        SRC=SRC,
    ).to(device)

    print(f"Number of training examples: {len(train.examples)}")
    if validation_step:
        print(f"Number of validation examples: {len(val.examples)}")
    print(f"The model has {count_parameters(model):,} trainable parameters")

    return model, out, SRC


def train_model(model, out, assess):
    """Apply given weights (& assess performance or train further) or start training new model

    Args:
        model: initialized model
        out: .pkg file with model parameters
        asses: bool 

    Returns:
        model with (new) weights
    """

    if os.path.exists(f"{out}.pkg") and assess:
        print("Assessing performance of existing model")

        model.load_state_dict(torch.load(f=out + ".pkg"))
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
        ) = model.evaluate(True)

        print(valids_de)
        print(unchanged_de)

        # log = open('unchanged.log', 'a')
        # info = f'type: comb unchanged: {unchan:.4g} unchanged_drugex: {unchan_de:.4g}'
        # print(info, file=log, flush = True)
        # print(valids_de)
        # print(unchanged_de)

        # print(unchan)
        # print(unchan_de)
        # df_output_de.to_csv(f'{out}_de_new.csv', index = False)

        # error_de = 1 - valids_de / len(drugex_iter.dataset)
        # print(error_de)
        # df_output.to_csv(f'{out}_par.csv', index = False)

    elif os.path.exists(f"{out}.pkg"):
        print("Continue training of existing model")
        # starts from the model after the last epoch, not the best epoch
        model.load_state_dict(torch.load(f=out + "_last.pkg"))
        # need to change how log file names epochs
        model.train_model()
    else:
        print("Start training model")
        model = model.apply(init_weights)
        model.train_model()

    return model


def correct_SMILES(model, out, error_source, device, SRC):
    """Model that is given corrects SMILES and return number of correct ouputs and dataframe containing all outputs
    Args:
        model: initialized model
        out: .pkg file with model parameters
        asses: bool 

    Returns:
        valids: number of fixed outputs
        df_output: dataframe containing output (either correct or incorrect) & original input
    """
    ## account for tokens that are not yet in SRC without changing existing SRC token embeddings
    errors = TabularDataset(
        path=error_source,
        format="csv",
        skip_header=False,
        fields={"SMILES": ("src", SRC)},
    )

    errors_loader = Iterator(
        errors,
        batch_size=64,
        device=device,
        sort=False,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
    )
    model.load_state_dict(torch.load(f=out + ".pkg"))
    # add option to use different iterator maybe?
    print("Correcting invalid SMILES")
    print(f"Number of invalid inputs: {len(errors.examples)}")
    valids, df_output = model.translate(errors_loader)
    #df_output.to_csv(f"{error_source}_fixed.csv", index=False)
    print(f"Finished, number of fixed outputs: {valids}")

    return valids, df_output
