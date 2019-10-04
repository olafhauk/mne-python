# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)
"""Compute resolution matrix for LCMV beamformers."""
from mne import pick_channels_forward
from mne.utils import logger


def make_resolution_matrix_lcmv(forward, filters):
    """Compute resolution matrix for linear inverse operator.

    Parameters
    ----------
    forward: dict
        Forward Operator.
    filters: Instance of Beamformer
         Dictionary containing filter weights from LCMV beamformer
         (see mne.beamformer.make_lcmv).

    Returns
    -------
        resmat: array, shape (n_dipoles_lcmv, n_dipoles_fwd)
        Resolution matrix (filter matrix times forward operator).
        Numbers of rows (n_dipoles_lcmv) and columns (n_dipoles_fwd) may differ
        depending on orientation constraints of filter and forward solution,
        respectively.
    """
    # don't include bad channels from noise covariance matrix
    bads_filt = filters['noise_cov']['bads']
    ch_names = filters['noise_cov']['names']

    # good channels
    ch_names = [c for c in ch_names if (c not in bads_filt)]

    forward = pick_channels_forward(forward, ch_names, ordered=True)

    # get leadfield matrix from forward solution
    leadfield = forward['sol']['data']

    # get the filter weights for beamformer as matrix
    filtmat = _get_matrix_from_lcmv(filters)

    resmat = filtmat.dot(leadfield)

    shape = resmat.shape

    logger.info('Dimensions of LCMV resolution matrix: %d by %d.' % shape)

    return resmat


def _get_matrix_from_lcmv(filters):
    """Helper got get the filter matrix from LCMV beamformer object.

    filters : instance of Beamformer
            Dictionary containing filter weights from LCMV beamformer.

    Returns
    -------
    filtmat: array, (n_dipoles, n_channels)
    The beamformer filter weights as matrix.
    """
    # filters as to be applied to unwhitened data or leadfield, hence include
    # whitening here
    filtmat = filters['weights'].dot(filters['whitener'])

    return filtmat
