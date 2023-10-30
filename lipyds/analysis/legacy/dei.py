"""
Lipid Depletion-Enrichment Index
================================

Classes
-------

.. autoclass:: LipidEnrichment
    :members:

"""

from typing import Union

import scipy
import numpy as np

from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.analysis import distances
from .base import LeafletAnalysisBase


class LipidEnrichment(LeafletAnalysisBase):
    r"""Calculate the lipid depletion-enrichment index around a protein
    by leaflet.

    The depletion-enrichment index (DEI) of a lipid around a protein
    indicates how enriched or depleted that lipid in the lipid annulus
    around the protein, with respect to the density of the lipid in the bulk
    membrane.

    The results of this analysis contain these values:

    * 'Near protein': the number of lipids :math:`L` within
        cutoff :math:`r` of the protein
    * 'Fraction near protein': the fraction of the lipids :math:`L`
        with respect to all lipids within cutoff :math:`r` of the
        protein: :math:`\frac{n(x_{(L, r)})}{n(x_r)}`
    * 'Enrichment': the depletion-enrichment index.

    The hard-cutoff, gaussian distribution algorithm was obtained from
    [Corradi2018]_. The soft-cutoff, binomial distribution algorithm
    was first published in [Wilson2021]_. Please cite them if you use 
    this analysis in published work.


    Parameters
    ----------
    universe: Universe or AtomGroup
        The atoms to apply this analysis to.
    select: str (optional)
        A :meth:`Universe.select_atoms` selection string
        for atoms that define the lipid head groups, e.g.
        "name PO4" or "name P*"
    select_protein: str (optional)
        Selection string for the protein.
    cutoff: float (optional)
        Cutoff in ångström
    buffer: float (optional)
        buffer zone length in ångström. If > 0, this means a
        soft cutoff is implemented.
    beta: float (optional)
        beta controls the sharpness of soft-cutoff.
    distribution: str (optional)
        Whether to use the binomial or gaussian distribution
    **kwargs
        Passed to :class:`~lipyds.analysis.base.LeafletAnalysisBase`.


    Attributes
    ----------
    dei_by_leaflet: list of dicts
        A list of dictionaries of time series data for each leaflet.
        The first dictionary is for the first leaflet, etc.
        Leaflets are sorted by z-coordinate; the first leaflet
        has the lowest z-coordinate.
    leaflets_summary: list of dicts
        A list of summary dictionaries for each leaflet.
        The first dictionary is for the first leaflet, etc.
    """

    def __init__(self, universe: Union[AtomGroup, Universe],
                 select_protein: str = "protein",
                 cutoff: float = 6,
                 distribution: str = "binomial",
                 buffer: float = 0, beta: float = 4,
                 **kwargs):
        super(LipidEnrichment, self).__init__(universe, **kwargs)

        self.distribution = distribution.lower()
        if self.distribution == "binomial":
            self._fit_distribution = self._fit_binomial
        elif self.distribution == "gaussian":
            self._fit_distribution = self._fit_gaussian
        else:
            raise ValueError("`distribution` should be either "
                             "'binomial' or 'gaussian'")

        self.protein = self.universe.select_atoms(select_protein)
        self.cutoff = cutoff
        self.buffer = buffer
        self.beta = beta

    def _prepare(self):
        # in case of change + re-run
        self.mid_buffer = self.buffer / 2.0
        self.max_cutoff = self.cutoff + self.buffer
        self._buffer_sigma = self.buffer / self.beta
        if self._buffer_sigma:
            self._buffer_coeff = 1 / (self._buffer_sigma * np.sqrt(2 * np.pi))
        self.unique_ids = np.unique(self.ids)

        # results
        self.near_counts = np.zeros((self.n_leaflets, len(self.ids),
                                     self.n_frames))
        self.residue_counts = np.zeros((self.n_leaflets, len(self.ids),
                                        self.n_frames))
        self.total_counts = np.zeros((self.n_leaflets, self.n_frames))

    def _update_leaflets(self):
        super()._update_leaflets()
        self._set_leaflets_with_outside()
        self._current_leaflet_residues = {}
        self._current_leaflet_ids = {}
        self._cache = {}
        for i, residues in enumerate(self.leaflet_residues):
            leaflet_residues = sum(residues) if len(residues) else self.residues[[]]
            self._current_leaflet_residues[i] = leaflet_residues
            self._current_leaflet_ids[i] = getattr(leaflet_residues, self.group_by_attr)

    def _single_frame(self):
        # initial scoop for nearby groups
        coords_ = self.selection.positions
        pairs = distances.capped_distance(self.protein.positions,
                                          coords_,
                                          self.cutoff, box=self.protein.dimensions,
                                          return_distances=False)
        if pairs.size > 0:
            indices = np.unique(pairs[:, 1])
        else:
            indices = []

        # now look for groups in the buffer
        if len(indices) and self.buffer:
            pairs2, dist = distances.capped_distance(self.protein.positions,
                                                     coords_, self.max_cutoff,
                                                     min_cutoff=self.cutoff,
                                                     box=self.protein.dimensions,
                                                     return_distances=True)

            # don't count things in inner cutoff
            mask = [x not in indices for x in pairs2[:, 1]]
            pairs2 = pairs2[mask]
            dist = dist[mask]

            if pairs2.size > 0:
                _ix = np.argsort(pairs2[:, 1])
                indices2 = pairs2[_ix][:, 1]
                dist = dist[_ix] - self.cutoff

                init_resix2 = self.selection.resindices[indices2]
                # sort through for minimum distance
                ids2, splix = np.unique(init_resix2, return_index=True)
                resix2 = init_resix2[splix]
                split_dist = np.split(dist, splix[1:])
                min_dist = np.array([x.min() for x in split_dist])

                # logistic function
                for i, leaflet_residues in self._current_leaflet_residues.items():
                    ids = self._current_leaflet_ids[i]
                    match, rix, lix = np.intersect1d(resix2, leaflet_residues.resindices,
                                                     assume_unique=True,
                                                     return_indices=True)
                    subdist = min_dist[rix]
                    subids = ids[lix]
                    for j, x in enumerate(self.ids):
                        mask = (subids == x)
                        xdist = subdist[mask]
                        exp = -0.5 * ((xdist/self._buffer_sigma) ** 2)
                        n = self._buffer_coeff * np.exp(exp)
                        self.near_counts[i, j, self._frame_index] += n.sum()

        soft = self.near_counts[:, :, self._frame_index].sum()

        init_resix = self.selection.resindices[indices]
        resix = np.unique(init_resix)

        for i, leaflet_residues in self._current_leaflet_residues.items():
            ids = self._current_leaflet_ids[i]
            _, ix1, ix2 = np.intersect1d(resix, leaflet_residues.resindices,
                                         assume_unique=True,
                                         return_indices=True)
            self.total_counts[i, self._frame_index] = len(ix1)
            subids = ids[ix2]
            for j, x in enumerate(self.ids):
                self.residue_counts[i, j, self._frame_index] += sum(ids == x)
                self.near_counts[i, j, self._frame_index] += sum(subids == x)

        both = self.near_counts[:, :, self._frame_index].sum()

    def _conclude(self):
        self.dei_by_leaflet = []
        self.leaflets_summary = []

        for i in range(self.n_leaflets):
            timeseries = {}
            summary = {}
            res_counts = self.residue_counts[i]
            near_counts = self.near_counts[i]

            near_all = near_counts.sum(axis=0)
            total_all = res_counts.sum(axis=0)
            n_near_tot = near_all.sum()
            n_all_tot = total_all.sum()
            d, s = self._collate(near_all, near_all, total_all,
                                 total_all, n_near_tot, n_all_tot)
            timeseries['all'] = d
            summary['all'] = s
            for j, resname in enumerate(self.ids):
                near_species = near_counts[j]
                total_species = res_counts[j]
                d, s = self._collate(near_species, near_all, total_species,
                                     total_all, n_near_tot, n_all_tot)
                timeseries[resname] = d
                summary[resname] = s
            self.dei_by_leaflet.append(timeseries)
            self.leaflets_summary.append(summary)

    def _fit_gaussian(self, data, *args, **kwargs):
        """Treat each frame as an independent observation in a gaussian
        distribution.

        Appears to be original method of [Corradi2018]_.

        .. note::

            The enrichment p-value is calculated from a two-tailed
            sample T-test, following [Corradi2018]_.

        """
        near = data['Near protein']
        frac = data['Fraction near protein']
        dei = data['Enrichment']
        summary = {
            'Mean near protein': near.mean(),
            'SD near protein': near.std(),
            'Mean fraction near protein': frac.mean(),
            'SD fraction near protein': frac.std(),
            'Mean enrichment': dei.mean(),
            'SD enrichment': dei.std()
        }

        return summary

    def _fit_binomial(self, data: dict, n_near_species: np.ndarray,
                      n_near: np.ndarray, n_species: np.ndarray,
                      n_all: np.ndarray, n_near_tot: int, n_all_tot: int):
        """
        This function computes the following approximate probability
        distributions and derives statistics accordingly.

        * The number of lipids near the protein is represented as a
        normal distribution.
        * The fraction of lipids near the protein follows a
        hypergeometric distribution.
        * The enrichment is represented as the log-normal distribution
        derived from the ratio of two binomial convolutions of the
        frame-by-frame binomial distributions.

        All these approximations assume that each frame or observation is
        independent. The binomial approximation assumes that:

        * the number of the lipid species near the protein is
        small compared to the total number of that lipid species
        * the total number of all lipids is large
        * the fraction (n_species / n_all) is not close to 0 or 1.

        .. note::

            The enrichment p-value is calculated from the log-normal
            distribution of the null hypothesis: that the average
            enrichment is representative of the ratio of
            n_species : n_all

        """

        summary = {"Total # lipids, all": n_all_tot,
                   "Total # lipids, shell": n_near_tot}
        p_time = data['Fraction near protein']
        summary['Total # species, shell'] = N = n_near_species.sum()
        summary['Total # species, all'] = N_sp = n_species.sum()
        if n_near_tot:  # catch zeros
            p_shell = N / n_near_tot
        else:
            p_shell = 0
        if n_all_tot:
            p_null = N_sp / n_all_tot
        else:
            p_null = 0

        # n events: assume normal
        summary['Mean # species, shell'] = n_near_species.mean()
        summary['SD # species, shell'] = sd = n_near_species.std()

        # actually hypergeometric, but binomials are easier
        # X ~ B(n_near_tot, p_shell)
        summary['Mean fraction of species, shell'] = p_shell
        summary['SD fraction of species, shell'] = sd_frac = sd / n_near.mean()

        if p_null == 0:
            summary['Mean enrichment'] = 1
            summary['SD enrichment'] = 0

        else:
            summary['Mean enrichment'] = p_shell / p_null
            summary['SD enrichment'] = sd_frac / p_null

        return summary

    def _collate(self, n_near_species: np.ndarray, n_near: np.ndarray,
                 n_species: np.ndarray, n_all: np.ndarray,
                 n_near_tot: int, n_all_tot: int):
        data = {}
        data['Near protein'] = n_near_species
        frac = np.nan_to_num(n_near_species / n_near, nan=0.0)
        data['Fraction near protein'] = frac
        data['Total number'] = n_species
        n_total = np.nan_to_num(n_species / n_all, nan=0.0)
        # data['Fraction total'] = n_total
        n_total[n_total == 0] = np.nan
        dei = np.nan_to_num(frac / n_total, nan=0.0)
        data['Enrichment'] = dei

        summary = self._fit_distribution(data, n_near_species, n_near,
                                         n_species, n_all, n_near_tot,
                                         n_all_tot)
        return data, summary

    def summary_as_dataframe(self):
        """Convert the results summary into a pandas DataFrame.

        This requires pandas to be installed.
        """

        if not self.leaflets_summary:
            raise ValueError('Call run() first to get results')
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('pandas is required to use this function '
                              'but is not installed. Please install with '
                              '`conda install pandas` or '
                              '`pip install pandas`.') from None

        dfs = [pd.DataFrame.from_dict(d, orient='index')
               for d in self.leaflets_summary]
        for i, df in enumerate(dfs, 1):
            df['Leaflet'] = i
        df = pd.concat(dfs)
        return df
