import pandas as pd
import numpy as np
from rdkit import Chem
from KLdiv import descriptors_generator, ReferenceDistribution, KLdiv
import os
import matplotlib.pyplot as plt


def calc_score_oversample(df, df_t, nbins, smoothen=False, oversample=False):
    divs = []
    div_dict = {}
    for column in df:
        # ignore fracrioCSP3 for comparison
        if column in ['FractionCSP3']: continue
        #if not column in ['MolWt']: continue

        #get features of sample
        array = df[column].to_numpy()
        array_t = df_t[column].to_numpy()
        np.save('mw_sample', array_t)
        np.save('mw_ref', array)

        # get step size resulting in around 100 overlapping bins
        if column in ['BertzCT', 'MolLogP', 'MolWt', 'TPSA']:
            #nbins = int(np.sqrt(len(array)))
            nbins = nbins
        else:
            nbins = 10
        lower_r = np.amin(array)
        upper_r = np.amax(array)
        lower_t = np.amin(array_t)
        upper_t = np.amax(array_t)
        lower = min(lower_r, lower_t)
        upper = max(upper_r, upper_t)
        step = abs(upper - lower) / nbins
        if step > 1:
            step = round(step)
        else:
            step_options = [0.0125, 0.25, 0.5, 1]
            print(step)
            step = min(step_options, key=lambda x: abs(x - step))
            print(step)

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
            if step > 1:
                ref = ReferenceDistribution(array, step)
            else:
                ref = ReferenceDistribution(array, 1)

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
            if step > 1:
                sam = ReferenceDistribution(array_t, step)
            else:
                sam = ReferenceDistribution(array_t, 1)

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
        plt.savefig(f'KLdiv/figure/hist_{column}_{nbins}.png')
        plt.cla()

        if column in ['BertzCT']:
            div = KLdiv.from_distribs(sam, ref, ignore_after=2500)
        elif column in ['MolWt']:
            div = KLdiv.from_distribs(sam,
                                      ref,
                                      ignore_until=100,
                                      ignore_after=800)
        elif column in ['MolLogP']:
            div = KLdiv.from_distribs(sam,
                                      ref,
                                      ignore_until=-5,
                                      ignore_after=10)
        else:
            div = KLdiv.from_distribs(sam, ref)
        div_dict[column] = div
        divs.append(div)
    # aggregate score based on lines 246-247 distribution_learning_benchmark.py https://github.com/BenevolentAI/guacamol/blob/8247bbd5e927fbc3d328865d12cf83cb7019e2d6/guacamol/distribution_learning_benchmark.py
    partial_scores = [np.exp(-score) for score in divs]
    score = sum(partial_scores) / len(partial_scores)

    return score, div_dict


if __name__ == "__main__":
    #train
    #baseline = {'BertzCT': 0.0012663525629169076, 'MolLogP': 0.0012065209678231415, 'MolWt': 0.0012795700186902514, 'TPSA': 0.0017940093339629782, 'NumHAcceptors': 0.00041051771644249013, 'NumHDonors': 0.0002586824098468902, 'NumRotatableBonds': 0.00020623681703676279, 'NumAliphaticRings': 0.0005457704253431826, 'NumAromaticRings': 0.0002883963658989537}
    #rl target
    baseline = {
        'BertzCT': 0.7404563100370731,
        'MolLogP': 0.1444840999399262,
        'MolWt': 0.47572564529454003,
        'TPSA': 0.5680360062815143,
        'NumHAcceptors': 0.11871301084751028,
        'NumHDonors': 0.13911422087920464,
        'NumRotatableBonds': 0.042620716054325825,
        'NumAliphaticRings': 0.022383544774584253,
        'NumAromaticRings': 0.25214489772632637
    }
    #gan
    #baseline = {'BertzCT': 5.11042570432477, 'MolLogP': 1.0242099272299707, 'MolWt': 4.317288089889052, 'TPSA': 1.5747177719963201, 'NumHAcceptors': 1.1634387959526278, 'NumHDonors': 0.04790876027999972, 'NumRotatableBonds': 0.029360601755958636, 'NumAliphaticRings': 0.06957515494625859, 'NumAromaticRings': 1.886575537064675}
    # reference set
    source = 'Data/'
    dest = 'KLdiv/'
    name = 'PAPYRUS_200_standardized_n'

    if os.path.exists(f'{dest}{name}_descriptors.csv'):
        df = pd.read_csv(f'{dest}{name}_descriptors.csv')
        #df = df.loc[df['MolWt'] <= 750]
    else:
        df = pd.read_csv(f"{source}{name}.csv", usecols=['STD_SMILES'])
        #df = df.sample(10000, random_state=42)
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
    #df_s = df_s.loc[df_s['MolWt'] <= 750]

    print(len(df_s))
    output = pd.DataFrame()
    info = pd.DataFrame()
    for smoothen in [False, 1, 8]:
        for oversample in [False, 2]:
            for nbins in range(10, 400, 20):
                #if nbins != 100: continue
                score, divs = calc_score_oversample(df,
                                                    df_s,
                                                    nbins=nbins,
                                                    smoothen=smoothen,
                                                    oversample=oversample)
                print(score)
                print(divs)
                df_dictionary = pd.DataFrame([divs])
                df_info = pd.DataFrame([[smoothen, oversample]],
                                       columns=['smoothen', 'oversample'])

                info = pd.concat([info, df_info], ignore_index=True)
                print(info)
                output = pd.concat([output, df_dictionary], ignore_index=True)

    info.to_csv('KLdiv/rl_compare_info.csv')
    print(output)
    output = output.sub(baseline.values(), axis='columns')
    output.to_csv('KLdiv/rl_compare.csv')
    output = pd.merge(output, info)
    output.to_csv('KLdiv/rl_compare.csv')
    # results = pd.DataFrame(data = divs)
    # results['SCORE'] = score
    #print(results)

    figure = False
    if figure:
        n_rows = 2
        n_cols = 2
        i = 0
        columns = ['BertzCT', 'MolLogP', 'MolWt', 'TPSA']
        fig, axes = plt.subplots(n_rows, n_cols)
        for row_num in range(n_rows):
            for col_num in range(n_cols):
                ax = axes[row_num][col_num]
                print(len(range(10, 400, 20)))
                print(output[columns[i]].shape)
                ax.scatter(range(10, 400, 20), output[columns[i]])
                ax.set_title(f'Plot ({columns[i]})')
                ax.set_ylabel('Normalized KLdiv')
                ax.set_xlabel('Number of bins')
                i += 1
        fig.suptitle('Effect number of bins on kldiv')
        fig.tight_layout()
        plt.savefig('KLdiv/rl_inskldiv_smooth_8.png')
