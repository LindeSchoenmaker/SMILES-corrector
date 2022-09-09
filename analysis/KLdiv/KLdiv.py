# -*- coding: utf-8 -*-

# Allow self reference in class definitions
from __future__ import annotations

from collections import namedtuple
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np
from scipy.stats import entropy
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors

default_descs = [
    'BertzCT', 'MolLogP', 'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors',
    'NumRotatableBonds', 'NumAliphaticRings', 'NumAromaticRings',
    'FractionCSP3'
]


def descriptors_generator(mols: Iterable[Chem.Mol],
                          descs: List[str] = None) -> Iterator:
    """Calculate molecular descriptors.
    
    :param mols: (a generator of) RDKit molecules
    :param descs: list of RDKit descriptors to be comupted
                  (default: BertzCT, MolLogP, MolWt, TPSA,
                            NumHAcceptors, NumHDonors,
                            NumRotatableBonds, NumAliphaticRings,
                            NumAromaticRings and FractionCSP3
    """
    if descs is None:
        descs = default_descs
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    for mol in mols:
        if mol is not None:
            values = calc.CalcDescriptors(mol)
            yield {name: value for name, value in zip(descs, values)}


Distrib = namedtuple('Distribution', ['scaling_factor', 'bins', 'values'])


class ReferenceDistribution:

    def __init__(self, descriptor: np.array, bin_width: float = 0.5):
        """Instanciate a ReferenceDistribution for a given descriptor.
        
        :param descriptor: Values of the descriptor to derive the distribution from.
        :param bin_width: Width of bins.
        """
        # Allow different instanciation
        if descriptor is None and bin_width is None:
            self.bin_width, self.min, self.max, self.nbins = None, None, None, None
            self.distrib, self._prev_bin_widths = None, []
            return
        self.bin_width = bin_width
        # Transform to NumPy array
        descriptor = np.array(descriptor)
        # Compute min and max
        min_, max_ = descriptor.min(), descriptor.max()
        # Round min and max to the nearest 0.5
        self.min, self.max = np.floor(min_ / bin_width) * bin_width, np.ceil(
            max_ / bin_width) * bin_width
        # Obtain number of bins
        self.nbins = int((self.max + bin_width - self.min) / bin_width)
        # Calculate histograms
        bins = np.linspace(self.min, self.max, self.nbins)
        hist, bins = np.histogram(descriptor, bins=bins, density=False)
        # Normalize distribution
        scaling_factor = hist.sum()
        hist = hist / scaling_factor
        self.distrib = Distrib(scaling_factor, bins, hist)
        # History of previously applied bin widths
        self._prev_bin_widths = [bin_width]

    @staticmethod
    def new(min: float,
            max: float,
            bin_width: float,
            nbins: int,
            distribution: Optional[Distrib] = None,
            prev_bin_widths: Optional[List[int]] = None
            ) -> ReferenceDistribution:
        """Create a ReferenceDistribution from the underlying data."""
        ref_distrib = ReferenceDistribution(None, None)
        ref_distrib.min, ref_distrib.max = min, max
        ref_distrib.bin_width = bin_width
        ref_distrib.nbins = nbins
        if distribution is not None:
            ref_distrib.distrib = Distrib(distribution.scaling_factor,
                                          distribution.bins.copy(),
                                          distribution.values.copy())
        if prev_bin_widths is not None:
            ref_distrib._prev_bin_widths = prev_bin_widths
        return ref_distrib

    def scale(self) -> List[Distrib]:
        """Obtain the bins and count distribution."""
        return ReferenceDistribution.new(
            self.min, self.max, self.bin_width, self.nbins,
            Distrib(1. / self.distrib.scaling_factor, self.distrib.bins,
                    self.distrib.hist * self.distrib.scaling_factor),
            self._prev_bin_widths)

    def smoothen(self, window_size: int = 3) -> ReferenceDistribution:
        """Smoothen the distributions using a sliding window average.
        
        :param window: sliding window (must be an odd number).
        """
        # Create the new ref distribution
        new_ref = ReferenceDistribution.new(self.min, self.max, self.bin_width,
                                            self.nbins, None,
                                            self._prev_bin_widths)
        # Modify internal distribution
        distrib = Distrib(
            self.distrib.scaling_factor, self.distrib.bins.copy(),
            np.concatenate([self.distrib.values, [0] * (window_size - 1)]))
        window = np.array([0.] * window_size)
        for j in range(len(distrib.values)):
            window[-1] = distrib.values[j]
            distrib.values[j] = np.mean(window)
            window = np.append(window[1:], 0)
        # Clip tails
        distrib = Distrib(distrib.scaling_factor, distrib.bins,
                          distrib.values[window_size - 2:-window_size + 2])
        # Update distribs
        new_ref.distrib = distrib
        return new_ref

    def oversample(self, bin_width: int) -> ReferenceDistribution:
        """Oversample bins.
        
        :param nbins: New number of bins. The distribution with greater
                      current nbins than provided are not modified.
        """
        # Create the new ref distribution
        new_ref = ReferenceDistribution.new(self.min, self.max, bin_width, 0,
                                            None, self._prev_bin_widths)
        # Do not modify if nbins < distrib.nbins
        if bin_width == self.bin_width:
            new_ref.nbins = self.nbins
            new_ref.distrib = self.distrib
            return new_ref
        elif bin_width > self.bin_width:
            raise ValueError(
                f'new step ({bin_width}) is lower than the original ({self.bin_width})'
            )
        # Calculate bins
        new_ref.nbins = int((self.max + bin_width - self.min) / bin_width)
        bins = np.linspace(self.min, self.max, new_ref.nbins)
        # Copy old distrib bins and values for popping
        old_bins, old_values = self.distrib.bins.copy(
        ), self.distrib.values.copy()
        # Calculate new distribution
        distrib = []
        for j in range(len(bins) - 1):
            if bins[j] >= old_bins[0] and bins[j + 1] <= old_bins[1]:
                # New interval fully contained in old bin -> weighted value
                distrib.append(old_values[0] * (bins[j + 1] - bins[j]) /
                               (old_bins[1] - old_bins[0]))
            else:
                # New interval in between two old intervals -> weighted sum
                distrib.append(old_values[0] * (old_bins[1] - bins[j]) /
                               (old_bins[1] - old_bins[0]) + old_values[1] *
                               (bins[j + 1] - old_bins[1]) /
                               (old_bins[2] - old_bins[1]))
                # Remove first elements
                old_bins, old_values = old_bins[1:], old_values[1:]
        new_ref.distrib = Distrib(self.distrib.scaling_factor, bins,
                                  np.array(distrib))
        # Add previous bin_width to history
        new_ref._prev_bin_widths.append(bin_width)
        return new_ref

    def __repr__(self):
        return f'<{self.__class__.__name__}: nbins={self.nbins}, '\
               f'bin width={self.bin_width}, min={self.min}, max={self.max}>'


class KLdiv:

    @staticmethod
    def from_sample(
            sample_descriptor: np.array,
            ref_distrib: ReferenceDistribution,
            ignore_empty_bins: bool = False,
            ignore_until: Optional[float] = None,
            ignore_after: Optional[float] = None) -> Union[float, List[float]]:
        """Obtain Kullback-Leibler divergence for the given descriptor.
        
        :param sample_descriptor: Values of the descriptor to be computed.
        :param ref_distrib: ReferenceDistribution to compare to.
        :param ignore_empty_bins: Should bin ranges undefined in the sample
                                  or reference values be disregarded.
        :param ignore_until: Ignore bins until the given value included
        :param ignore_after: Ignore bins after the given value included
        """
        # Obtain a ReferenceDistribution
        sample_distrib = ReferenceDistribution(sample_descriptor,
                                               ref_distrib._prev_bin_widths[0])
        # Apply all oversampling steps applied to the reference distrib
        for bin_width in ref_distrib._prev_bin_widths:
            sample_distrib = sample_distrib.oversample(bin_width)
        return KLdiv.from_distribs(sample_distrib, ref_distrib,
                                   ignore_empty_bins, ignore_until,
                                   ignore_after)

    @staticmethod
    def from_distribs(sample_distrib: ReferenceDistribution,
                      ref_distrib: ReferenceDistribution,
                      ignore_empty_bins: bool = False,
                      ignore_until: Optional[float] = None,
                      ignore_after: Optional[float] = None) -> float:
        """Obtain Kullback-Leibler divergence for the given distributions.
        
        :param sample_distrib: ReferenceDistribution to be evaluated.
        :param ref_distrib: ReferenceDistribution to compare to.
        :param ignore_empty_bins: Should bin ranges undefined in the sample
                                  or reference values be disregarded.
        :param ignore_until: Ignore bins until the given bin location included
        :param ignore_after: Ignore bins after the given bin location included
        """
        if sample_distrib.bin_width != ref_distrib.bin_width:
            raise ValueError(
                'distributions do not have the same bin_width values')
        # Check some bins and are aligned between both distributions
        aligned_bins_sample = np.in1d(
            sample_distrib.distrib.bins.round(3),
            ref_distrib.distrib.bins.round(3)).nonzero()[0]
        if len(aligned_bins_sample) == 0:
            raise ValueError('distributions do not align')
        ## Align distributions
        # Create union of bin ids
        bin_ids = np.unique(
            np.concatenate(
                (sample_distrib.distrib.bins, ref_distrib.distrib.bins)))
        # Create 0-filled bin values
        bins_sample, bins_ref = np.full(bin_ids.shape,
                                        0.0), np.full(bin_ids.shape, 0.0)
        # Fill where defined bin values
        bins_sample[np.in1d(
            bin_ids,
            sample_distrib.distrib.bins[:-1])] = sample_distrib.distrib.values
        bins_ref[np.in1d(
            bin_ids,
            ref_distrib.distrib.bins[:-1])] = ref_distrib.distrib.values
        # Ignore left-tailed distributions
        if ignore_until is not None:
            idx = np.where(bin_ids > ignore_until)[0]
            bins_sample = bins_sample[idx]
            bins_ref = bins_ref[idx]
            bin_ids = bin_ids[idx]
        # Ignore right-tailed distributions
        if ignore_after is not None:
            idx = np.where(bin_ids < ignore_after)[0]
            bins_sample = bins_sample[idx]
            bins_ref = bins_ref[idx]
            bin_ids = bin_ids[idx]
        if ignore_empty_bins:
            # Discard values where ref or sample values are 0
            non_zero_idx = np.where(
                np.bitwise_and(bins_sample != 0, bins_ref != 0))[0]
            bins_sample = bins_sample[non_zero_idx]
            bins_ref = bins_ref[non_zero_idx]
        else:
            # Replace 0 by epsilon, such that KL is defined
            bins_sample = np.where(bins_sample == 0, 1e-10, bins_sample)
            bins_ref = np.where(bins_ref == 0, 1e-10, bins_ref)
        ## Calculate KL div
        kl_div = entropy(bins_sample, bins_ref)
        return kl_div
