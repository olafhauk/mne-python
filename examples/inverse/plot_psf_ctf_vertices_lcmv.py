# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)
"""
Point-spread functions (PSFs) and cross-talk functions (CTFs) for beamformer.

Visualise PSF and CTF at one vertex for LCMV beamformer for different data
covariance matrices.
"""

import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv
from mne.minimum_norm.resolution_matrix import get_point_spread, get_cross_talk
from mne.beamformer.resolution_matrix_lcmv import make_resolution_matrix_lcmv

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects/'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evo = data_path + '/MEG/sample/sample_audvis-ave.fif'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

# Read raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)

picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')

# Find events
events = mne.find_events(raw)

event_id = {'aud/l': 1, 'aud/r': 2, 'vis/l': 3, 'vis/r': 4}

epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.05, tmax=0.25,
                    picks=picks, baseline=(0.05, 0.25), preload=True)

# compute separate covariance matrices
cov_aud = mne.cov.compute_covariance(epochs['aud/l', 'aud/r'], method='empirical')
cov_vis = mne.cov.compute_covariance(epochs['vis/l', 'vis/r'], method='empirical')

# read forward solution
forward = mne.read_forward_solution(fname_fwd)
# forward operator with fixed source orientations
forward = mne.convert_forward_solution(forward, surf_ori=True,
                                       force_fixed=True)

# noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# evoked data for info
evoked = mne.read_evokeds(fname_evo, 0)

# compute beamformer filters for auditory covariance matrix
filters_aud = make_lcmv(evoked.info, forward, cov_aud, reg=0.05,
                        noise_cov=noise_cov,
                        pick_ori=None, rank=None,
                        weight_norm=None,
                        reduce_rank=False,
                        verbose=False)

# compute beamformer filters for visual covariance matrix
filters_vis = make_lcmv(evoked.info, forward, cov_vis, reg=0.05,
                        noise_cov=noise_cov,
                        pick_ori=None, rank=None,
                        weight_norm=None,
                        reduce_rank=False,
                        verbose=False)

# compute resolution matrix for auditory covariance
rm_aud = make_resolution_matrix_lcmv(forward, filters_aud)

# compute resolution matrix for auditory covariance
rm_vis = make_resolution_matrix_lcmv(forward, filters_vis)

# get CTFs
sources = [3000]

stc_aud = get_cross_talk(rm_aud, forward['src'], sources, norm=True)

stc_vis = get_cross_talk(rm_vis, forward['src'], sources, norm=True)

# Visualise

# Which vertex corresponds to selected source
vertno_lh = forward['src'][0]['vertno']
verttrue = [vertno_lh[sources[0]]]  # just one vertex

brain_aud = stc_aud.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=1, title='Auditory Covariance',
                         clim=dict(kind='value', lims=(0, 0.1, 0.2)))

brain_vis = stc_vis.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=2, title='Visual Covariance',
                         clim=dict(kind='value', lims=(0, 0.1, 0.2)))

# True source location for PSF
brain_aud.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of PSF
brain_vis.add_foci(verttrue, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='green')
