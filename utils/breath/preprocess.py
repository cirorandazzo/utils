# utils.breath.preprocess
#
# for loading raw data into useful data structures

from multiprocessing import Pool
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from scipy.signal import butter

import diptest

from .segment import (
    fit_breath_distribution,
    segment_breaths,
)
from .breath import make_notmat_vars

from ..audio import AudioObject, get_triggers_from_audio
from ..callbacks import call_mat_stim_trial_loader
from ..file import parse_birdname


def load_datasets(datasets, file_format, fs_dataset=None):
    """
    Load multiple datasets which have been created during preprocess_files.

    file_format look something like this: r"M:\randazzo\breathing\processed\{dataset}\{dataset}-{file}.pickle"

    where {dataset} will be filled iteratively with the elements in datasets, and {file} will be filled with "all_files" and "all_breaths"


    fs_dataset is a dict of the format {dataset_name: fs}. if it's passed, adds column "fs" to relevant rows in all_files/all_breaths

    """

    all_files = []
    all_breaths = []

    for dataset in datasets:
        all_files_path = file_format.format(dataset=dataset, file="all_files")
        all_breaths_path = file_format.format(dataset=dataset, file="all_breaths")

        with open(all_files_path, "rb") as f:
            files = pickle.load(f)
            files["dataset"] = dataset

        with open(all_breaths_path, "rb") as f:
            breaths = pickle.load(f)
            breaths["dataset"] = dataset

        all_files.append(files)
        all_breaths.append(breaths)

    all_files = pd.concat(all_files).sort_index()
    all_breaths = pd.concat(all_breaths).sort_index()

    # add fs
    if fs_dataset is not None:
        all_files["fs"] = all_files.dataset.map(fs_dataset)
        all_breaths["fs"] = all_breaths.dataset.map(fs_dataset)

    return all_files, all_breaths


def preprocess_files(
    files,
    output_folder,
    prefix,
    fs,
    channel_map,
    has_stims=True,
    stim_length=0.1,
    output_exist_ok=False,
    n_jobs=4,
):
    """
    Generalized preprocessing script for audio and breath data using multiprocessing.
    """
    st = time.time()  # timer
    print(f"======Starting {prefix} ({len(files)} files)======")

    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=output_exist_ok)

    # Butterworth filter parameters
    bw = dict(N=2, Wn=50, btype="low")
    b_br, a_br = butter(**bw, fs=fs)

    # multiprocess files
    print("Starting multiprocessing pool...")
    with Pool(n_jobs) as pool:
        results = pool.starmap(
            preprocess_single_file,
            [
                (
                    file,
                    fs,
                    channel_map,
                    has_stims,
                    stim_length,
                    output_folder,
                    b_br,
                    a_br,
                )
                for file in files
            ],
        )

        print(f"Finished processing files! ({time.time() - st}s since start)")
        print(f"Trying to join...")

        pool.close()
        pool.terminate()

    print(f"Pool closed. ({time.time() - st}s since start)")
    print("Gathering results...")

    # Collect results
    all_files = []
    all_breaths = []
    errors = {}

    for stim_trials, calls, error in results:
        if error:
            errors[error[0]] = error[1]
        else:
            all_files.append(
                stim_trials.reset_index().set_index(["audio_filename", "stims_index"])
            )
            all_breaths.append(
                calls.reset_index().set_index(["audio_filename", "calls_index"])
            )

    # Merge dataframes
    if all_files:
        all_files = pd.concat(all_files).sort_index()
    if all_breaths:
        all_breaths = pd.concat(all_breaths).sort_index()

    # Save metadata
    with open(output_folder.joinpath(f"{prefix}-all_files.pickle"), "wb") as f:
        pickle.dump(all_files, f)

    with open(output_folder.joinpath(f"{prefix}-all_breaths.pickle"), "wb") as f:
        pickle.dump(all_breaths, f)

    with open(output_folder.joinpath(f"{prefix}-errors.pickle"), "wb") as f:
        pickle.dump(errors, f)

    print(f"Processing complete! {len(all_files)} files processed successfully.")
    print(f"{len(errors)} files encountered errors.")
    print(f"Elapsed time: {time.time() - st}")
    print(f"======Finished {prefix}!======\n")

    return all_files, all_breaths


def preprocess_single_file(
    file,
    fs,
    channel_map,
    has_stims,
    stim_length,
    output_folder,
    b_br,
    a_br,
):
    """
    Helper function to process a single file. This will be used with multiprocessing. See preprocess_files for parameter descriptions.
    """
    try:
        file = Path(file)
        print(f"Processing {file.name}...")

        # Load audio and breath channels
        aos = AudioObject.from_file(file, channel="all")
        ao_audio = aos[channel_map["audio"]]
        ao_breath = aos[channel_map["breath"]]

        assert (
            ao_audio.fs == fs
        ), f"Wrong sample rate! Expected {fs}, but file {file} had fs={ao_audio.fs}"

        # Filter breath
        ao_breath.filtfilt(b_br, a_br)
        breath = ao_breath.audio_filt
        
        # test amplitude distr bimodality
        dipstat = diptest.diptest(breath)

        # Center and normalize breath
        x_dist, _, trough_ii, amplitude_ii = fit_breath_distribution(breath)
        zero_point, insp_peak = x_dist[[trough_ii, min(amplitude_ii)]]
        
        # Normalize breath
        breath -= zero_point
        insp_peak = insp_peak - zero_point  # rel. to new zero
        breath /= abs(insp_peak)

        # Segment breaths
        exps, insps = segment_breaths(
            breath,
            fs,
            threshold=lambda x: 0,
            do_filter=False,  # already filtered
        )

        # Create .notmat-like structure
        onsets, offsets, labels = make_notmat_vars(
            exps, insps, len(breath), exp_label="exp", insp_label="insp"
        )
        onsets, offsets = onsets / fs, offsets / fs

        # Get stims (or dummy stim)
        if has_stims:
            stim_channel = channel_map["stim"]
            stims = (
                get_triggers_from_audio(
                    aos[stim_channel].audio,
                    crossing_direction="down",
                    threshold_function=lambda x: 1000,
                    allowable_range=[
                        fs / 2,
                        np.inf,
                    ],  # range for allowable # of frames between subsequent stims.
                )
                / fs
            )

            if len(stims) == 0:
                raise NoStimulusError("Didn't find any stimuli!")
        else:
            stims = np.array([0.0])

        # mimic .not.mat format (for call_mat_stim_trial_loader)
        data = {
            "onsets": np.concatenate([onsets, stims]) * 1000,
            "offsets": np.concatenate([offsets, stims + stim_length]) * 1000,
            "labels": np.concatenate([labels, ["Stimulus"] * len(stims)]),
        }

        # Generate calls and trials dataframes
        calls, stim_trials, _, _, _ = call_mat_stim_trial_loader(
            file=None,
            data=data,
            from_notmat=True,
            verbose=False,
            acceptable_call_labels=["Stimulus", "exp", "insp"],
        )

        calls.drop(columns=["ici"], inplace=True)
        calls = calls.loc[calls["type"] != "Stimulus"]

        # Add metadata
        calls["amplitude"] = calls.apply(
            _get_amplitude,
            axis=1,
            args=[fs, breath],
        )

        # TODO: implement putative calls. figure out thresholding for new normalization
        # stim_trials["n_putative_calls"] = calls[calls["amplitude"] > 1.1].shape[0]
        stim_trials["breath_zero_point"] = zero_point
        stim_trials["insp_peak"] = insp_peak
        stim_trials["dipstat"] = dipstat

        # Save processed data as .npy
        np_file = Path(output_folder).joinpath(file.stem + ".npy")
        np.save(np_file, np.vstack([ao_audio.audio, breath]))

        # Save filenames
        calls["audio_filename"] = str(file)
        calls["numpy_filename"] = str(np_file)
        stim_trials["audio_filename"] = str(file)
        stim_trials["numpy_filename"] = str(np_file)

        # Add birdname
        try:
            birdname = parse_birdname(
                str(file.parent)
            )  # formerly: file.name. but parent usually includes a birdname folder
        except ValueError:
            birdname = ""

        calls["birdname"] = birdname
        stim_trials["birdname"] = birdname

        return stim_trials, calls, None  # No error

    except Exception as e:
        print(f"ERROR: {file.name}. ({e}).")
        return None, None, (str(file), e)  # Return error info


def _get_amplitude(x, fs, breath):

    s, e = (np.array([x.start_s, x.end_s]) * fs).astype(int)

    # dumb edge case
    if s == e:
        return breath[s]

    if x.type == "exp":
        amp = max(breath[s:e])
    elif x.type == "insp":
        amp = min(breath[s:e])
    else:
        raise ValueError(f"Unknown breath type: {x.type}")

    return amp


class NoStimulusError(LookupError):
    """
    Raised when a file unexpectedly has no stimulus triggers.
    """

    pass
