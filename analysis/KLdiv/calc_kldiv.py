import pandas as pd
import numpy as np
from rdkit import Chem
from KLdiv import descriptors_generator, ReferenceDistribution, KLdiv
import os
import matplotlib.pyplot as plt


def calc_score_oversample(df, df_t, smoothen=False, oversample=False):
    """Obtain KL divergences and aggregated score"""
    divs = []
    div_dict = {}
    for column in df:
        #get features of sample
        array = df[column].to_numpy()
        array_t = df_t[column].to_numpy()

        # set step size
        if column == 'BertzCT':
            step = 30
        elif column == 'MolLogP':
            step = 0.25
        elif column in ['MolWt']:
            step = 10
        elif column in ['TPSA']:
            step = 4
        elif column in [
                'NumHAcceptors', 'NumHDonors', 'NumAliphaticRings',
                'NumAromaticRings'
        ]:
            step = 1
        elif column == 'NumRotatableBonds':
            step = 2
        elif column == 'FractionCSP3':
            step = 0.05

        # for each array calculate reference distribution
        if column in ['BertzCT', 'MolLogP', 'MolWt',
                      'TPSA']:  # also include FractionCSP3?
            ref = ReferenceDistribution(array, step)
            #for floats oversample & smoothen bins
            if oversample:
                ref = ref.oversample(step / oversample)
            if smoothen:
                for i in range(smoothen):
                    ref = ref.smoothen()
        else:
            ref = ReferenceDistribution(array, step)

        if column in ['BertzCT', 'MolLogP', 'MolWt',
                      'TPSA']:  # also include FractionCSP3?
            sam = ReferenceDistribution(array_t, step)
            #for floats oversample & smoothen bins
            if oversample:
                sam = sam.oversample(step / oversample)
            if smoothen:
                for i in range(smoothen):
                    sam = sam.smoothen()
        else:
            sam = ReferenceDistribution(array_t, step)

        #save figure
        plt.bar(x=ref.distrib.bins[:-1],
                height=ref.distrib.values,
                width=np.diff(ref.distrib.bins),
                align='edge',
                fc='sandybrown',
                **{'alpha': 0.5})
        plt.bar(x=sam.distrib.bins[:-1],
                height=sam.distrib.values,
                width=np.diff(sam.distrib.bins),
                align='edge',
                fc='skyblue',
                **{'alpha': 0.5})

        #plt.hist(array, bins = len(ref.distrib.bins), density=True, label='ref', **{'alpha': 0.8})
        #plt.hist(array_t, bins = len(sam.distrib.bins), density=True, label='sample', **{'alpha': 0.8})
        plt.savefig(f'KLdiv/figure/hist_{column}.png')
        plt.cla()

        if column in ['BertzCT']:
            div = KLdiv.from_distribs(sam, ref, ignore_after=2500)
        elif column in ['MolLogP']:
            div = KLdiv.from_distribs(sam,
                                      ref,
                                      ignore_until=-5,
                                      ignore_after=10)
        elif column in ['MolWt']:
            div = KLdiv.from_distribs(sam,
                                      ref,
                                      ignore_until=100,
                                      ignore_after=800)
        else:
            div = KLdiv.from_distribs(sam, ref)
        div_dict[column] = div
        divs.append(div)
    # aggregate score based on lines 246-247 distribution_learning_benchmark.py https://github.com/BenevolentAI/guacamol/blob/8247bbd5e927fbc3d328865d12cf83cb7019e2d6/guacamol/distribution_learning_benchmark.py
    partial_scores = [np.exp(-score) for score in divs]
    score = sum(partial_scores) / len(partial_scores)

    return score, div_dict


def table_1():
    # reference set
    source = 'Data/'
    dest = 'KLdiv/'
    name = 'PAPYRUS_200_standardized'

    if os.path.exists(f'{dest}{name}_descriptors.csv'):
        df = pd.read_csv(f'{dest}{name}_descriptors.csv')
    else:
        df = pd.read_csv(f"{source}{name}.csv", usecols=['STD_SMILES'])
        mol_list = []
        for smiles in df['STD_SMILES']:
            if isinstance(smiles, str):
                mol_list.append(Chem.MolFromSmiles(smiles))
        descriptors = descriptors_generator(mol_list)
        df = pd.DataFrame(list(descriptors))
        df.to_csv(f'{dest}{name}_descriptors.csv', index=None)

    df_s = pd.read_csv(f"generated/rl_valid_s.csv")
    mol_list = []
    for smiles in df_s['SMILES']:
        if isinstance(smiles, str):
            mol_list.append(Chem.MolFromSmiles(smiles))
    descriptors = descriptors_generator(mol_list)
    df_s = pd.DataFrame(list(descriptors))

    score, divs = calc_score_oversample(df, df_s, smoothen=1, oversample=False)
    output = pd.DataFrame([divs])
    print(output)

    cases = ['RNN', 'RNN_target', 'VAE', 'GAN']

    fixed = [
        'generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_rnn_errors_200_M_fixed.csv',
        'generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_rl_errors_200_M_fixed.csv',
        'generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_vae_errors_200_M_fixed.csv',
        'generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_gan_ckpt100_M_errors_200_fixed.csv'
    ]
    generated = [
        'generated/rnn_valid.csv', 'generated/rl_valid.csv',
        'generated/vae_valid.csv', 'generated/gan_ckpt100_M_valid.csv'
    ]

    score_ft = []
    score_fg = []
    score_gt = []
    for i, case in enumerate(cases):
        # compare fixed to train
        df_fixed = pd.read_csv(fixed[i], usecols=['CORRECT']).dropna()
        mol_list = []
        for smiles in df_fixed['CORRECT']:
            if isinstance(smiles, str):
                mol_list.append(Chem.MolFromSmiles(smiles))
        descriptors = descriptors_generator(mol_list)
        df_fixed = pd.DataFrame(list(descriptors))

        score, divs = calc_score_oversample(df,
                                            df_fixed,
                                            smoothen=1,
                                            oversample=False)
        if case != 'RNN_target':
            score_ft.append(score)
        file = open(f"KLdiv/results/{case}_fixed_train.txt", "w")

        for key, value in divs.items():

            file.write('%s: %.4f\n' % (key, value))
        file.write('Score: %.4f\n' % (score))

        file.close()

        # compare fixed to generated (correct-in-one-go)
        df_gen = pd.read_csv(generated[i], usecols=['SMILES'])
        gen = df_gen.sample(100000, random_state=42).SMILES.tolist()
        mol_list = []
        for smiles in gen:
            if isinstance(smiles, str):
                mol_list.append(Chem.MolFromSmiles(smiles))
        descriptors = descriptors_generator(mol_list)
        df_gen = pd.DataFrame(list(descriptors))

        score, divs = calc_score_oversample(df_gen,
                                            df_fixed,
                                            smoothen=1,
                                            oversample=False)
        if case != 'RNN_target':
            score_fg.append(score)
        file = open(f"KLdiv/results/{case}_fixed_gen.txt", "w")

        for key, value in divs.items():

            file.write('%s: %.4f\n' % (key, value))
        file.write('Score: %.4f\n' % (score))

        file.close()

        # compare generated (correct-in-one-go) to train
        gen = df_gen.sample(10000, random_state=42)
        score, divs = calc_score_oversample(df,
                                            gen,
                                            smoothen=1,
                                            oversample=False)
        if case != 'RNN_target':
            score_gt.append(score)
        file = open(f"KLdiv/results/{case}_gen_train.txt", "w")

        for key, value in divs.items():

            file.write('%s: %.4f\n' % (key, value))
        file.write('Score: %.4f\n' % (score))

        file.close()

    for score_list in [score_ft, score_fg, score_gt]:

        file = open(f"KLdiv/results/{i}_mean_std.txt", "a")

        file.write('Score: %.3f ± %.4f\n' %
                   (sum(score_list) / len(score_list), np.std(score_list)))

        file.close()


def table_2():
    # reference set

    df = pd.read_csv('/zfsdata/data/linde/DrugEx_new/AURORA/data/dataset.tsv',
                     sep='\t',
                     usecols=['SMILES'])
    train = df.SMILES.tolist()
    mol_list = []
    for smiles in train:
        if isinstance(smiles, str):
            mol_list.append(Chem.MolFromSmiles(smiles))
    descriptors = descriptors_generator(mol_list)
    df = pd.DataFrame(list(descriptors))

    df_fixed = pd.read_csv(
        'generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_rl_errors_200_M_fixed.csv',
        usecols=['CORRECT']).dropna()
    df_explore = pd.read_csv(
        'Data/explore/selective_ki_with_1000_errors_index_fixed.csv',
        usecols=['CORRECT']).dropna()
    df_gen = pd.read_csv('Data/explore/scores/rl.tsv',
                         sep='\t',
                         usecols=['Smiles'])

    # compare fixed to train
    fix = df_fixed.sample(10000, random_state=42).CORRECT.tolist()
    mol_list = []
    for smiles in fix:
        if isinstance(smiles, str):
            mol_list.append(Chem.MolFromSmiles(smiles))
    descriptors = descriptors_generator(mol_list)
    df_fixed = pd.DataFrame(list(descriptors))

    score, divs = calc_score_oversample(df,
                                        df_fixed,
                                        smoothen=1,
                                        oversample=False)
    file = open(f"KLdiv/results/table2_explore_fixed_train.txt", "w")

    for key, value in divs.items():

        file.write('%s: %.4f\n' % (key, value))
    file.write('Score: %.4f\n' % (score))

    file.close()

    # compare exploreto train
    explore = df_explore.sample(10000, random_state=42).CORRECT.tolist()
    mol_list = []
    for smiles in explore:
        if isinstance(smiles, str):
            mol_list.append(Chem.MolFromSmiles(smiles))
    descriptors = descriptors_generator(mol_list)
    df_explore = pd.DataFrame(list(descriptors))

    score, divs = calc_score_oversample(df,
                                        df_explore,
                                        smoothen=1,
                                        oversample=False)
    file = open(f"KLdiv/results/table2_explore_explore_train.txt", "w")

    for key, value in divs.items():

        file.write('%s: %.4f\n' % (key, value))
    file.write('Score: %.4f\n' % (score))

    file.close()

    # compare generated (correct-in-one-go) to train
    gen = df_gen.sample(10000, random_state=42).Smiles.tolist()
    mol_list = []
    for smiles in gen:
        if isinstance(smiles, str):
            mol_list.append(Chem.MolFromSmiles(smiles))
    descriptors = descriptors_generator(mol_list)
    df_gen = pd.DataFrame(list(descriptors))

    score, divs = calc_score_oversample(df,
                                        df_gen,
                                        smoothen=1,
                                        oversample=False)
    file = open(f"KLdiv/results/table2_rnn_fixed_train.txt", "w")

    for key, value in divs.items():

        file.write('%s: %.4f\n' % (key, value))
    file.write('Score: %.4f\n' % (score))

    file.close()


if __name__ == "__main__":
    table_2()
