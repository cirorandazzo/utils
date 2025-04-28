# utils.breath.preprocess
# 
# for loading raw data into useful data structures

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter

from .segment import (
    fit_breath_distribution,
    segment_breaths,
)
from .breath import make_notmat_vars
from ..audio import AudioObject

from utils.callbacks import call_mat_stim_trial_loader
from utils.file import parse_birdname

from utils.video import get_triggers_from_audio

def preprocess_files(
    files,
    output_folder,
    prefix,
    fs,
    channel_map,
    has_stims=True,
    stim_length=0.1,
    output_exist_ok=False,
):
    """
    Generalized preprocessing script for audio and breath data.

    Args:
        input_folder (str): Path to the input folder containing `wav` or `cbin` files.
        output_folder (str): Path to the output folder for processed `npy` files.
        prefix (str): string to prepend to output files.
        fs (int): Sample rate for processing.
        channel_map (dict): Mapping of channels, e.g., {'audio': 0, 'breath': 1}.
        file_type (str, optional): file type to look for in `input_folder`. default: "wav"
        has_stims (bool, optional): whether to expect stim triggers in file. If true, looks for stim triggers (requires channel_map["stim"]. Errors if no triggers are found.) If false, presumes each recording is a single trial. default: True
        stim_length (float, optional): length of stimuli in seconds. default: 0.1
        output_exist_ok (bool, optional):

    """
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=output_exist_ok)

    # Initialize containers
    all_files = []
    all_breaths = []
    errors = {}

    # Butterworth filter parameters
    bw = dict(N=2, Wn=50, btype="low")
    b_br, a_br = butter(**bw, fs=fs)

    for i, file in enumerate(files):
        file = Path(file)
        print(f"{i + 1}/{len(files)}, Processing {file.name}...")

        try:
            # Load audio and breath channels
            aos = AudioObject.from_file(file, channel="all")
            ao_audio = aos[channel_map["audio"]]
            ao_breath = aos[channel_map["breath"]]

            assert (
                ao_audio.fs == fs
            ), f"Wrong sample rate! Expected {fs}, but file {file} had fs={ao_audio.fs}"

            # TODO: optionally filter audio
            audio = ao_audio.audio

            # Filter breath
            ao_breath.filtfilt(b_br, a_br)
            breath = ao_breath.audio_filt

            # Center and normalize breath
            x_dist, _, trough_ii, amplitude_ii = fit_breath_distribution(
                breath,
            )
            zero_point, insp_peak = x_dist[[trough_ii, min(amplitude_ii)]]
            breath -= zero_point
            breath /= abs(insp_peak)  # formerly: abs(np.min(breath))

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
                # read stim triggers from correct channel
                stim_channel = channel_map["stim"]
                stims = (
                    get_triggers_from_audio(
                        aos[stim_channel].audio, crossing_direction="down"
                    )
                    / fs
                )

                if len(stims) == 0:
                    raise NoStimulusError("Didn't find any stimuli!")
            else:
                # Treat each file as a single trial
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

            # drop stimulus from calls df
            calls = calls.loc[calls["type"] != "Stimulus"]

            # Add metadata
            calls["amplitude"] = calls.apply(
                _get_amplitude,
                axis=1,
                args=[fs, breath],
            )

            # TODO: add putative call
            # putative call
            #   - exps: amplitude
            #   - insps: next exp (shift index; assert type)
            #
            # putative song
            #   - exps: if this is call and -2 or +2 is a call, putative song.
            #   - insps: if this is call and -1 or +3 (exp) is a call, putative song.

            # TODO: add stim phase to stim_trials (if has_stims is True)

            stim_trials["n_putative_calls"] = calls[calls["amplitude"] > 1.1].shape[0]

            stim_trials["breath_zero_point"] = zero_point

            # Save processed data as .npy
            np_file = output_folder.joinpath(file.stem + ".npy")
            np.save(np_file, np.vstack([audio, breath]))

            # save filenames
            calls["audio_filename"] = str(file)
            calls["numpy_filename"] = str(np_file)

            stim_trials["audio_filename"] = str(file)
            stim_trials["numpy_filename"] = str(np_file)

            # Add birdname
            try:
                birdname = parse_birdname(file.name)
            except ValueError as e:
                # if parsing fails, leave birdname blank
                birdname = ""

            calls["birdname"] = birdname
            stim_trials["birdname"] = birdname

            # Append to containers
            all_files.append(
                stim_trials.reset_index().set_index(["audio_filename", "stims_index"])
            )
            all_breaths.append(
                calls.reset_index().set_index(["audio_filename", "calls_index"])
            )

            print(f"\tSuccess!")

        except Exception as e:
            errors[str(file)] = e
            print(f"\tError: {e}")
            continue

    # Merge dataframes
    all_files = pd.concat(all_files).sort_index()
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

