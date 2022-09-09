# adapted from https://github.com/molecularsets/moses/blob/7b8f83b21a9b7ded493349ec8ef292384ce2bb52/moses/metrics/metrics.py

import torch
import warnings
from multiprocessing import Pool
import numpy as np
from scipy.spatial.distance import cosine as cos_distance
from scipy.stats import wasserstein_distance

# scikit-learn bootstrap

from moses.dataset import get_dataset, get_statistics
from moses.utils import mapper
from moses.utils import disable_rdkit_log, enable_rdkit_log
from moses.metrics.utils import compute_fragments, average_agg_tanimoto, \
    compute_scaffolds, fingerprints, \
    get_mol, canonic_smiles, mol_passes_filters

import pandas as pd


def get_some_metrics(gen,
                     k=None,
                     n_jobs=1,
                     device='cpu',
                     batch_size=512,
                     pool=None,
                     test=None,
                     test_scaffolds=None,
                     ptest=None,
                     ptest_scaffolds=None):
    """
    Computes all available metrics between test (scaffold test)
    and generated sets of SMILES.
    Parameters:
        gen: list of generated SMILES
        k: int or list with values for unique@k. Will calculate number of
            unique molecules in the first k molecules. Default [1000, 10000]
        n_jobs: number of workers for parallel processing
        device: 'cpu' or 'cuda:n', where n is GPU device number
        batch_size: batch size for FCD metric
        pool: optional multiprocessing pool to use for parallelization
        test (None or list): test SMILES. If None, will load
            a default test set
        test_scaffolds (None or list): scaffold test SMILES. If None, will
            load a default scaffold test set
        ptest (None or dict): precalculated statistics of the test set. If
            None, will load default test statistics. If you specified a custom
            test set, default test statistics will be ignored
        ptest_scaffolds (None or dict): precalculated statistics of the
            scaffold test set If None, will load default scaffold test
            statistics. If you specified a custom test set, default test
            statistics will be ignored
        train (None or list): train SMILES. If None, will load a default
            train set
    Available metrics:
        * %valid
        * %unique@k
        * Frechet ChemNet Distance (FCD)
        * Fragment similarity (Frag)
        * Scaffold similarity (Scaf)
        * Similarity to nearest neighbour (SNN)
        * Internal diversity (IntDiv)
        * Internal diversity 2: using square root of mean squared
            Tanimoto similarity (IntDiv2)
        * %passes filters (Filters)
        * Distribution difference for logP, SA, QED, weight
        * Novelty (molecules not present in train)
    """
    if test is None:
        if ptest is not None:
            raise ValueError("You cannot specify custom test "
                             "statistics for default test set")
        test = get_dataset('test')
        ptest = get_statistics('test')

    if test_scaffolds is None:
        if ptest_scaffolds is not None:
            raise ValueError("You cannot specify custom scaffold test "
                             "statistics for default scaffold test set")
        test_scaffolds = get_dataset('test_scaffolds')
        ptest_scaffolds = get_statistics('test_scaffolds')

    if k is None:
        k = [1000, 10000]

    disable_rdkit_log()
    metrics = {}
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    metrics['valid'] = fraction_valid(gen, n_jobs=pool)
    gen = remove_invalid(gen, canonize=True)
    test = remove_invalid(test, canonize=True)
    if not isinstance(k, (list, tuple)):
        k = [k]
    for _k in k:
        metrics['unique@{}'.format(_k)] = fraction_unique(gen, _k, pool)

    if ptest is None:
        ptest = compute_intermediate_statistics(test,
                                                n_jobs=n_jobs,
                                                device=device,
                                                batch_size=batch_size,
                                                pool=pool)

    mols = mapper(pool)(get_mol, gen)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    metrics['SNN/Test'], metrics['Novelty/Test'] = SNNMetric(**kwargs)(
        gen=mols, pref=ptest['SNN'])
    metrics['Frag/Test'] = FragMetric(**kwargs)(gen=mols, pref=ptest['Frag'])
    metrics['Scaf/Test'] = ScafMetric(**kwargs)(gen=mols, pref=ptest['Scaf'])

    enable_rdkit_log()
    if close_pool:
        pool.close()
        pool.join()
    return metrics


def compute_intermediate_statistics(smiles,
                                    n_jobs=1,
                                    device='cpu',
                                    batch_size=512,
                                    pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)
    if close_pool:
        pool.terminate()
    return statistics


def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)


def internal_diversity(gen,
                       n_jobs=1,
                       device='cpu',
                       fp_type='morgan',
                       gen_fps=None,
                       p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(
        gen_fps, gen_fps, agg='mean', device=device, p=p)).mean()


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn("Can't compute unique@{}.".format(k) +
                          "gen contains only {} molecules".format(len(gen)))
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if x is not None]


class Metric:

    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


def average_agg_tanimoto(stock_vecs,
                         gen_vecs,
                         batch_size=5000,
                         agg='max',
                         device='cpu',
                         p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules
    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1 / p)
    novelty = np.count_nonzero(agg_tanimoto != 1) / len(agg_tanimoto)
    return np.mean(agg_tanimoto), novelty


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {
            'fps': fingerprints(mols, n_jobs=self.n_jobs, fp_type=self.fp_type)
        }

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref['fps'],
                                    pgen['fps'],
                                    device=self.device)


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:
     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


class FragMetric(Metric):

    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])


class ScafMetric(Metric):

    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])


class WassersteinMetric(Metric):

    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {'values': values}

    def metric(self, pref, pgen):
        return wasserstein_distance(pref['values'], pgen['values'])


if __name__ == "__main__":
    source = '../Data/'
    dest = 'KLdiv/'
    name = 'PAPYRUS_200_standardized'
    df = pd.read_csv(f"{source}{name}.csv", usecols=['STD_SMILES'])
    train = df.sample(100000, random_state=42).STD_SMILES.tolist()

    cases = ['RNN', 'RNN_target', 'VAE', 'GAN']

    fixed = [
        '../generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_rnn_errors_200_M_fixed.csv',
        '../generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_rl_errors_200_M_fixed.csv',
        '../generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_vae_errors_200_M_fixed.csv',
        '../generated/multi/transformer_multiple_12_PAPYRUS_200_16_3_gan_ckpt100_M_errors_200_fixed.csv'
    ]
    generated = [
        '../generated/rnn_valid.csv', '../generated/rl_valid.csv',
        '../generated/vae_valid.csv', '../generated/gan_ckpt100_M_valid.csv'
    ]

    df_ft = pd.DataFrame()
    df_fg = pd.DataFrame()
    df_gt = pd.DataFrame()
    for i, case in enumerate(cases):
        # compare fixed to train
        df_fixed = pd.read_csv(fixed[i], usecols=['CORRECT']).dropna()
        fix = df_fixed.sample(10000, random_state=42).CORRECT.tolist()
        dict_metrics_ft = get_some_metrics(gen=fix, test=train)
        if case != 'RNN_target':
            df_ft = pd.concat(
                [df_ft, pd.DataFrame(dict_metrics_ft, index=[i])])

        file = open(f"tanimoto/{case}_test_fixed_train.txt", "w")

        for key, value in dict_metrics_ft.items():

            file.write('%s: %.4f\n' % (key, value))

        file.close()

        # compare fixed to generated (correct-in-one-go)
        df_gen = pd.read_csv(generated[i], usecols=['SMILES'])
        gen = df_gen.sample(100000, random_state=42).SMILES.tolist()
        dict_metrics_fg = get_some_metrics(gen=fix, test=gen)
        if case != 'RNN_target':
            df_fg = pd.concat(
                [df_fg, pd.DataFrame(dict_metrics_fg, index=[i])])

        file = open(f"tanimoto/{case}_fixed_gen.txt", "w")

        for key, value in dict_metrics_fg.items():

            file.write('%s: %.4f\n' % (key, value))

        file.close()

        # compare generated (correct-in-one-go) to train
        gen = df_gen.sample(10000, random_state=42).SMILES.tolist()
        dict_metrics_gt = get_some_metrics(gen=gen, test=train)
        if case != 'RNN_target':
            df_gt = pd.concat(
                [df_gt, pd.DataFrame(dict_metrics_gt, index=[i])])
        file = open(f"tanimoto/{case}_gen_train.txt", "w")

        for key, value in dict_metrics_gt.items():

            file.write('%s: %.4f\n' % (key, value))

        file.close()

    names = ['fixed_train', 'fixed_generated', 'generated_train']
    for i, df in enumerate([df_ft, df_fg, df_gt]):
        df.to_csv(f'{i}.csv')
        print(df)
        file = open(f"tanimoto/{names[i]}_mean_std.txt", "a")

        for key, value in dict_metrics_gt.items():
            print(df[key].mean)
            file.write('%s: %.2f ± %.3f\n' %
                       (key, df[key].mean(), df[key].std()))

        file.close()
