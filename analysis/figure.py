#!/usr/bin/env python
# This script is used for drawing figures shown in the manuscript.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from statistics import stdev 
from rdkit.Chem import MolFromSmiles
import seaborn as sns
import math
import re
from tueplots import axes
from tueplots import cycler
from tueplots.constants.color import palettes
from matplotlib import gridspec

plt.rcParams.update(axes.lines())
plt.rcParams.update(cycler.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']))

# configuration for drawing figures on linux
plt.switch_backend('Agg')


def get_metrics(df, calc_reconstructed = True):
    """get fraction of correct, reconstructed and changed SMILES for dataframe with column CORRECT, INCORRECT & STD_SMILES/ORIGINAL"""
    df_c = df.dropna(subset=['CORRECT'])
    correct = len(df_c)
    for index, row in df_c.iterrows():
        if calc_reconstructed:
            m = MolFromSmiles(row['CORRECT'])
            if "ERROR" in df.columns:
                p = MolFromSmiles(row['STD_SMILES'])
            else:
                p = MolFromSmiles(row['ORIGINAL'].split('<unk>')[0])
            df_c.loc[index, 'RECONSTRUCTED'] = m.HasSubstructMatch(p) and p.HasSubstructMatch(m)
        df.loc[index, 'CHANGED']  = row['CORRECT'] != row['ORIGINAL']

    if calc_reconstructed:
        reconstructed = (df_c.RECONSTRUCTED).sum()
    else:
        reconstructed = None

    df_i = df.dropna(subset=['INCORRECT'])
    for index, row in df_i.iterrows():
        if "ERROR" in df.columns:
            df.loc[index, 'CHANGED']  = row['INCORRECT'] != row['ERROR']
        else:
            df.loc[index, 'CHANGED']  = row['INCORRECT'] != row['ORIGINAL']

    changed = (df.CHANGED).sum()

    print(correct)
    if calc_reconstructed:
        print(reconstructed)
        reconstructed =  reconstructed/100
    print(changed)

    return correct/100, reconstructed, changed/100


def cat_errors(f):
    """categorize errors based on text file with RDKit error messages"""
    errors = f.readlines()
    error_list = []
    for error in errors:
        if error.find("Error") >= 1:
            #failed while parsing results in double mistake for one invalid smile
            #does however hold information on type of mistake
            #so not only pass those errors that do not contain while parsing
            if error.find("while parsing") == -1:
                error = re.split('Error:|for|while', error)
                if re.search('Failed', error[1]) != None:
                    error[1] = 'syntax error'
                elif re.search('ring closure', error[1]) != None:
                    error[1] = 'bond exists'
                # to merge parentheses errors together
                elif re.search('parentheses', error[1]) != None:
                    error[1] = 'parentheses error'
                error_list.append(error[1])
        elif error.find("valence") >= 1:
            error = 'valence error'
            error_list.append(error)
        elif error.find("kekulize") >= 1:
            error = 'aromaticity error'
            error_list.append(error)
        elif error.find("marked aromatic") >= 1:
            error = 'aromaticity error'
            error_list.append(error)
        elif error.find("exists") >= 1:
            error = 'bond exists'
            error_list.append(error)
        elif error.find("atoms") >= 1:
            #error = 'extra close parentheses'
            error = 'parentheses error'
            error_list.append(error)
    #to return type of errors
    legend = list(dict.fromkeys(error_list))
    print(legend)
    print(len(error_list))

    df = pd.DataFrame()
    df['Mistake'] = error_list
    return df, legend, len(error_list)


def freq(source_folder, df, legend, name, multiple=False, perc=True):
    """
    Frequency plot of different errors
    """
    # Bar plot
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    if multiple:
        if perc:
            sns.barplot(x="type",
                        y='perc',
                        hue='Name',
                        orient="v",
                        data=df,
                        order=[
                            'aromaticity error', ' unclosed ring ',
                            'parentheses error', 'valence error',
                            'syntax error', 'bond exists'
                        ])
        else:
            sns.barplot(x="Mistake",
                        y='Num',
                        hue='Name',
                        orient="v",
                        data=df,
                        estimator=lambda x: len(x) / len(df) * 100,
                        order=[
                            'aromaticity error', ' unclosed ring ',
                            'parentheses error', 'valence error',
                            'syntax error', 'bond exists'
                        ])
    else:
        sns.barplot(x="Mistake",
                    y='Num',
                    orient="v",
                    data=df,
                    estimator=lambda x: len(x) / len(df) * 100,
                    order=[
                        'aromaticity error', ' unclosed ring ',
                        'parentheses error', 'valence error', 'syntax error',
                        'bond exists'
                    ])
    len(legend)
    ax.set_xticks(np.arange(6))
    final_legend = [
        'Aromaticity \nError', 'Unclosed \nRing', 'Parentheses \nError',
        'Valence \nError', 'Syntax \nError', 'Bond \nAlready Exists'
    ]
   
    ax.set_xticklabels(final_legend)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_xlabel('')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:], framealpha=0.2)
    #ax.set(ylim=(0, 1000))
    fig.savefig(f'{source_folder}{name}.png', dpi=300)


def facet(source_folder, name, df, perc=True):
    """
    Facet plot with frequency of different errors for different number of errors used for training
    """
    sns.set_theme(style="ticks")
    print(df)
    df['type'] = df['type'].str.title()
    df['type'] = df['type'].replace([' Unclosed Ring '], 'Unclosed Ring')
    df['type'] = df['type'].replace(['Bond Exists'], 'Bond Already Exists')

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(
        df,
        col="type",
        hue='case',
        col_order=[
            'Aromaticity Error', 'Unclosed Ring', 'Parentheses Error',
            'Valence Error', 'Syntax Error', 'Bond Already Exists'
        ],
        hue_order=['RNN', 'RNN target-directed', 'VAE', 'GAN'],
        col_wrap=3,
        height=3)

    # Draw a line plot to show the trajectory of each random walk
    if perc:
        grid.map(plt.plot, "num", "perc", marker="o", alpha=0.8)
    else:
        grid.map(plt.plot, "num", "freq", marker="o", alpha=0.8)

    print(df.num.unique().tolist())
    grid.set_titles(col_template="{col_name}")
    grid.add_legend(title=' ',
                    loc='upper right',
                    bbox_to_anchor=(1.01, 1),
                    framealpha=0.2)

    # Adjust the tick positions and labels
    if perc:
        grid.set_axis_labels("Errors per input", 'Frequency (%)')
        grid.set(xlim=(0, max(df.num.unique().tolist()) + 0.5),
                 ylim=(0, math.ceil(max(df.perc) / 10) * 10))
    else:
        grid.set_axis_labels("Errors per input", 'Frequency (counts)')
        grid.set(xlim=(0, max(df.num.unique().tolist()) + 0.5),
                 ylim=(0, max(df.freq) + 100))

    grid.savefig(f'{source_folder}{name}.png', dpi=600)


def figure_1(source_folder : str = 'generated/error_analysis/', names : list = ['rnn', 'rl', 'vae', 'gan_ckpt100_M']):
    """
    For creating figure 1 
    Arguments:
        source_folder: folder that contains text files of rdkit error messages (created using checksmiles.py (optionally via setup.py)
        names: list of file names
    """
    df_complete = pd.DataFrame(columns=['Name'])
    for i, name in enumerate(names):
        f = open(f"{source_folder}{name}.txt", "r")
        df, legend, total = cat_errors(f)
        df_new = df['Mistake'].value_counts().to_frame()
        df_new = df_new.reset_index(level=0)
        df_new.columns = ['type', 'freq']
        df['Num'] = 1
        df_new['perc'] = df_new['freq'] / total * 100
        if name == 'rl':
            df_new['Name'] = 'RNN target-directed'
        elif name == 'rnn':
            df_new['Name'] = 'RNN'
        elif name == 'vae':
            df_new['Name'] = 'VAE'
        elif name == 'gan_ckpt100_M':
            df_new['Name'] = 'GAN'
        df_complete = pd.concat([df_complete, df_new])

    print(df_complete)
    freq(source_folder, df_complete, legend, 'Figure_1', multiple=True)


def figure_2(correct_name : str = 'Data/performance/transformer_all_1_PAPYRUS_200_16_3_PAPYRUS_200_correct_S_fixed.csv',
             incorrect_name : str = 'Data/performance/transformer_all_1_PAPYRUS_200_16_3_PAPYRUS_200_incorrect_S_fixed.csv',
             original_name : str = 'Data/errors/split/PAPYRUS_200_all_1_errors_dev.csv',
             cases : list = ['rnn', 'rl', 'gentlr', 'gan_ckpt100_M'],
             legends : list = ['RNN', 'RNN \ntarget-directed', 'VAE', 'GAN']):
    """
    For creating figure 2 
    Arguments:
        correct_name: csv file with outputs of transformer model trained on correct inputs
        incorrect_name: csv file with outputs of transformer model trained on incorrect inputs
        original_name: csv file with original inputs,
        cases: list of file names pointing to de novo generated outputs fixed by the generator
        legends: legend
    """
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 0.5, 2]) 
    changed_rate = []
    error_rate = []
    molecule_reconstruction_error_rate = []
    
    #performance on correct set
    df = pd.read_csv(correct_name)
    print(df.head(10))
    correct, reconstructed, changed = get_metrics(df)
    changed_rate.append(changed)
    error_rate.append(correct)
    molecule_reconstruction_error_rate.append(reconstructed)

    # performance on incorrect set
    df = pd.read_csv(incorrect_name)
    df_original = pd.read_csv(original_name)
    df_new = pd.merge(df, df_original, left_on = 'ORIGINAL', right_on = 'ERROR' )
    print(df_new)
    correct, reconstructed, changed = get_metrics(df_new)
    print(f'correct:{correct}')
    print(f'reconstructed:{reconstructed}')
    print(changed)
    changed_rate.append(changed)
    error_rate.append(correct)
    molecule_reconstruction_error_rate.append(reconstructed)


    print(changed_rate)

    legends = []
    # Bar plot
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])

    legends = ['Valid\n', 'Invalid']
    # zipped_lists = zip(legends, changed_rate, error_rate, molecule_reconstruction_error_rate)
    # sorted_pairs = sorted(zipped_lists)

    # tuples = zip(*sorted_pairs)
    # legends, changed_rate, error_rate, molecule_reconstruction_error_rate = [ list(tuple) for tuple in  tuples]
    loc = np.array([0.05, 0.95])
    ax1.bar(loc - 0.2, changed_rate, 0.2, label='Inputs\nAltered') #SMILES alteration rate
    ax1.bar(loc, error_rate, 0.2, label='Valid\nSMILES') #SMILES validation rate
    ax1.bar(loc + 0.2, molecule_reconstruction_error_rate, 0.2, label='Molecules\nReconstructed') #Molecule reconstruction rate, identical, correct 
    
    #loc = np.arange(len(legends)) 
    ax1.set_xticks(loc)
    ax1.set_xticklabels(legends)
    ax1.set_ylim(0.0, 100)
    ax1.set_xlabel('Validation Set')
    ax1.set_ylabel('Percentage')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.7, 1), facecolor='white', framealpha=0)

    changed_rate = []
    error_rate = []

    # performance on case studies
    for case in cases:
        df = pd.read_csv(f'generated/multi/transformer_all_1_PAPYRUS_200_16_3_{case}_errors_200_S_fixed.csv')
        correct, _, changed = get_metrics(df, calc_reconstructed = False)
        changed_rate.append(changed)
        error_rate.append(correct)

    # zipped_lists = zip(legends, changed_rate, error_rate)
    # sorted_pairs = zipped_lists

    # tuples = zip(*sorted_pairs)
    # legends, changed_rate, error_rate = [ list(tuple) for tuple in  tuples]

    ax2.bar(np.arange(len(changed_rate)) - 0.15, changed_rate, 0.3, label='Inputs\nAltered') #SMILES alteration rate
    ax2.bar(np.arange(len(legends)) + 0.15, error_rate, 0.3, label='Valid\nSMILES') #SMILES validation rate
    
    x = np.arange(len(legends)) 
    ax2.set_xticks(x)
    ax2.set_xticklabels(legends)
    ax2.set_ylim(0.0, 100)
    ax2.set_xlabel('Case Study\n')
    ax2.set_ylabel('Percentage')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1), facecolor='white', framealpha=0)

    #fig.tight_layout()
    fig.savefig('generated/Figure_2.png', dpi=300)    


def figure_3(source_folder : str = 'generated/multi/error_analysis/',
             cases : list = ['rnn', 'rl', 'gentlr_S', 'gan_ckpt100_M']):
    """
    For creating figure 3
    Arguments:
        source_folder: folder that contains text files of rdkit error messages (created using checksmiles.py (optionally via setup.py)
        cases: list of file names pointing to de novo generated outputs fixed by the generator
    """
    names : list = ['all_1']
    df_complete = pd.DataFrame()
    for i, name in enumerate(names):
        for case in cases:
            name_new = f'{name}_{case}'
            f = open(f"{source_folder}{name_new}.txt", "r")
            df, legend, total = cat_errors(f)
            df_new = df['Mistake'].value_counts().to_frame()
            df_new = df_new.reset_index(level=0)
            df_new.columns = ['type', 'freq']
            df_new['perc'] = df_new['freq'] / 100
            df_new['num'] = int(name.split('_')[1])
            if case == 'rl':
                df_new['case'] = 'RNN target-directed'
            elif case == 'rnn':
                df_new['case'] = 'RNN'
            elif case == 'gentlr_S':
                df_new['case'] = 'VAE'
            elif case == 'gan_ckpt100_M':
                df_new['case'] = 'GAN'
            print(name)
            print(case)
            print(total)
            df_complete = pd.concat([df_complete, df_new])

    names = [
        'multi_2', 'multi_3', 'multi_5', 'multi_8', 'multi_12', 'multi_20'
    ]
    print(names)
    for i, name in enumerate(names):
        for case in cases:
            name_new = f'{name}_{case}'
            f = open(f"{source_folder}{name_new}.txt", "r")
            df, legend, total = cat_errors(f)
            df_new = df['Mistake'].value_counts().to_frame()
            df_new = df_new.reset_index(level=0)
            df_new.columns = ['type', 'freq']
            df_new['perc'] = df_new['freq'] / 100
            df_new['num'] = int(name.split('_')[1])

            if case == 'rl':
                df_new['case'] = 'RNN target-directed'
            elif case == 'rnn':
                df_new['case'] = 'RNN'
            elif case == 'gentlr_S':
                df_new['case'] = 'VAE'
            elif case == 'gan_ckpt100_M':
                df_new['case'] = 'GAN'

            df_complete = pd.concat([df_complete, df_new])

    facet('generated/multi/', 'Figure_3', df_complete, perc=True)


def figure_4(name = 'explore_novelty_1000'):
    """
    For creating figure 3
    Arguments:
        name: name of file with pickled list of list for average novelty (created by exploration_novelty.py)
    """
    import pickle
    with open(name, 'rb') as fp:
        n_list = pickle.load(fp)
    turned = []
    made = []
    print(len(n_list[0]))
    lengths = []
    for parent in n_list:
        lengths.append(len(parent))
    print(lengths)
    print(min(lengths))
    print(max(lengths))
    avg = sum(lengths)/len(lengths)
    print(avg)
    for i in range(round(avg)):
        comb = []
        for item in n_list:
            try:
                comb.append(item[i])
            except IndexError:
                pass
        if len(comb) > 0:
            turned.append(comb)
            made.append(i+1)

    #print(len(turned))

    row_average = [sum(sub_list) / len(sub_list) for sub_list in turned]
    res = [i / j * 100 for i, j in zip(row_average, made)]
    #print(res)
    #print(res)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #print(made)

    ax1.scatter(made, res)
    ax1.set_ylabel("Novelty (%)")
    ax1.set_xlabel("Generated sequences")


    fig.tight_layout()
    fig.savefig('explore_novelty_1000.png', dpi=300)  
    

if __name__ == '__main__':
    figure_1()
    figure_2()
    figure_3()
    figure_4()