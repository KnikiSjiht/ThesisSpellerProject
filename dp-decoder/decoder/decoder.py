import mne
from mnelab.io import read_raw
import numpy as np
import os
import pyntbci
import pyxdf
from decoder.utils.logging import logger
from scipy.signal import butter, sosfilt, resample
import pickle
from decoder.signal_controller import SignalEmbedder
import time
# from ...dp-speller.speller.speller import TRIAL_TIME, PR, CODE # Imports from another module? probably not the DarePlane way? Another way of ensuring important values such a s trial_time and stimulus presentation are correct?

data_path = "./data/training data"
classifier_path = "./data/classifiers"
task = "cvep"

TRIAL_TIME = 4.2
PR = 60
CODE = "mgold_61_6521"


tmin = 0.0  # start of trial [s]
tmax = TRIAL_TIME  # length of a trial [s]
l_freq = 2.0  # high pass cutoff frequency [Hz]
h_freq = 30.0  # low pass cutoff frequency [Hz]
n_channels = 32  # number of channels
fs = 120  # target sampling frequency to downsample to [Hz]
pr = PR  # stimulus presentation rate [Hz]
window = 1.0  # window of data to take before a trial to catch filter artefacts [s]


def create_classifier(subject, session, run):
    logger.setLevel(10)

    # Load EEG
    fn = os.path.join(data_path, f"sub-{subject}_ses-{session}_task-{task}_run-{run}_eeg.xdf")
    streams = pyxdf.resolve_streams(fn)
    names = [stream["name"] for stream in streams]
    raw = read_raw(fn, stream_ids=[streams[names.index("BioSemi")]["stream_id"]])

    # Adjust marker channel data
    raw._data[0, :] -= np.min(raw._data[0, :])
    raw._data[0, raw._data[0, :] > 0] = 1
    idx = np.where(raw._data[0, :] > 0)[0]
    idx = idx[np.concatenate(([False], np.diff(idx) < raw.info["sfreq"]))]
    raw._data[0, idx] = 0

    # Read events
    events = mne.find_events(raw, stim_channel="Trig1", verbose=False)

    # Slicing
    X = mne.Epochs(raw, events=events, tmin=tmin - window, tmax=tmax, baseline=None, picks="eeg", preload=True,
                   verbose=False).get_data(copy=True, verbose=False)
    logger.info(f"X shape: {X.shape} (trials x channels x samples)")

    # Spectral filtering
    sos = butter(N=2, Wn=[l_freq, h_freq], btype='bandpass', output='sos', fs=raw.info["sfreq"])
    X = sosfilt(sos, X, axis=2)

    # Resampling
    X = resample(X, int(np.round(X.shape[2] / raw.info["sfreq"] * fs)), axis=2)

    # Remove 500 ms around
    X = X[:, :, int(window * fs):]
    logger.info(f"X shape: {X.shape} (trials x channels x samples)")

    # Extract target labels
    fn = os.path.join(data_path, f"sub-{subject}_ses-{session}_task-{task}_run-{run}_eeg.xdf")
    streams = pyxdf.load_xdf(fn)[0]
    names = [stream["info"]["name"][0] for stream in streams]
    stream = streams[names.index("KeyboardMarkerStream")]
    y = np.array([int(marker[3]) for marker in stream["time_series"] if marker[2] == "target"])
    logger.info(f"y shape: {y.shape} (trials)")

    # Load codes
    V = np.repeat(np.load(f"C:/Users/Thijs/Documents/Studie/Thesis/Speller Project/dp-speller/speller/codes/{CODE}.npz")["codes"].T, int(fs / pr), axis=1)
    logger.info(f"V shape: {V.shape} (codes x samples)")

    # Cross-validation
    n_folds = 4
    n_trials = X.shape[0]
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))
    accuracy = np.zeros(n_folds)

    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event="duration", encoding_length=0.3, onset_event=True)

    for i_fold in range(n_folds):
        X_trn, y_trn = X[i_fold != folds, :, :], y[i_fold != folds]
        X_tst, y_tst = X[i_fold == folds, :, :], y[i_fold == folds]

        rcca.fit(X_trn, y_trn)

        yh = rcca.predict(X_tst)[:, 0]
        accuracy[i_fold] = np.mean(yh == y_tst)

    logger.info(f"Classifier created with accuracy: {accuracy.mean():.3f}")
    class_loc = os.path.join(classifier_path, f"rCCA_Classifier_sub-{subject}_ses-{session}.pkl")
    with open(class_loc, 'wb') as file:
        pickle.dump(rcca, file)
    logger.info(f"Classifier saved to {class_loc}")

def decode(subject="P001", session="S002"):
    class_loc = os.path.join(classifier_path, f"rCCA_Classifier_sub-{subject}_ses-{session}.pkl")
    with open(class_loc, 'rb') as file:
        classifier = pickle.load(file)
    decoder = SignalEmbedder(classifier,"mockup_random", "decoder", tmax+1, fs, (l_freq, h_freq), marker_stream_name="mockup_random_markers", markers="start_trial")
    decoder.init_all()
    decoder.run()

decode()