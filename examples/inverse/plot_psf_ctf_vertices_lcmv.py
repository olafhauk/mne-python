"""
==============================================================
Compute point-spread functions for LCMV beamformers
==============================================================

Visualise CTF at one vertex for LCMV beamformer for covariance matrices
computed for pre- and post-stimulus intervals, respectively.
"""

import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv
from mne.minimum_norm.resolution_matrix import get_cross_talk
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

raw.set_eeg_reference('average', projection=True)
raw.info['bads'] += ['EEG 053']  # bads + 1 more

picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')

# Find events
events = mne.find_events(raw)

# event_id = {'aud/l': 1, 'aud/r': 2, 'vis/l': 3, 'vis/r': 4}
event_id = {'vis/l': 3, 'vis/r': 4}

tmin, tmax = -.2, .25  # epoch duration
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    picks=picks, baseline=(-.2, 0.), preload=True)

# covariance matrix for pre-stimulus interval
tmin, tmax = -.2, 0.
cov_pre = mne.cov.compute_covariance(epochs, tmin=tmin, tmax=tmax,
                                     method='empirical')
# covariance matrix for post-stimulus interval (around main evoked responses)
tmin, tmax = 0.05, .25
cov_post = mne.cov.compute_covariance(epochs, tmin=tmin, tmax=tmax,
                                      method='empirical')

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
filters_pre = make_lcmv(evoked.info, forward, cov_pre, reg=0.05,
                        noise_cov=noise_cov,
                        pick_ori=None, rank=None,
                        weight_norm=None,
                        reduce_rank=False,
                        verbose=False)

# compute beamformer filters for visual covariance matrix
filters_post = make_lcmv(evoked.info, forward, cov_post, reg=0.05,
                         noise_cov=noise_cov,
                         pick_ori=None, rank=None,
                         weight_norm=None,
                         reduce_rank=False,
                         verbose=False)

# compute resolution matrix for auditory covariance
rm_pre = make_resolution_matrix_lcmv(forward, filters_pre)

# compute resolution matrix for auditory covariance
rm_post = make_resolution_matrix_lcmv(forward, filters_post)

# get CTFs
sources = [3000]

stc_pre = get_cross_talk(rm_pre, forward['src'], sources, norm=True)

stc_post = get_cross_talk(rm_post, forward['src'], sources, norm=True)

# Visualise

# Which vertex corresponds to selected source
vertno_lh = forward['src'][0]['vertno']
verttrue = [vertno_lh[sources[0]]]  # just one vertex

brain_pre = stc_pre.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=1, title='Pre-stimulus Covariance',
                         clim=dict(kind='value', lims=(0, 0.1, 0.2)))

brain_post = stc_post.plot('sample', 'inflated', 'lh',
                           subjects_dir=subjects_dir,
                           figure=2, title='Post-stimulus Covariance',
                           clim=dict(kind='value', lims=(0, 0.1, 0.2)))

# True source location for PSF
brain_pre.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of PSF
brain_post.add_foci(verttrue, coords_as_verts=True, scale_factor=1.,
                    hemi='lh', color='green')

print('The pre-stimulus beamformer''s CTF has lower values in parietal regions'
      '(suppressed alpha activity?) but larger values in occipital regions'
      '(less suppression of visual activity)')
