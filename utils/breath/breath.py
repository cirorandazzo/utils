# utils.breath.breath
#
# misfits left over from moving to submodules. largely related to dataframe manipulations.
#

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..umap import loc_relative as umap__loc_relative


def get_first_breath_segment(
    trial,
    breath_type,
    fs,
    earliest_allowed_onset=None,
    buffer_s=0,
    return_stim_aligned=True,
    return_unit="samples",
):
    """
    Returns the onset and offset of the first instance of a specified 'breath_type' call after a given
    earliest allowed onset, optionally adding a buffer. The result can be returned in different units
    (samples, frames, seconds, or milliseconds) and optionally adjusted for the stimulus-aligned time.

    Parameters:
    -----------
    trial : dict
        The trial data containing 'call_types' and 'call_times_stim_aligned'.
    breath_type : str
        The type of breath (or call) to search for.
    fs : float
        The sampling frequency (samples per second) of the audio data.
    earliest_allowed_onset : float, optional
        The earliest allowed onset time (in seconds) after stimulus presentation for the breath type to occur. Default is None.
    buffer_s : float, optional
        The amount of time (in seconds) to extend both the onset and offset of the detected breath. Default is 0.
    return_stim_aligned : bool, optional
        Whether to return the time in stimulus-aligned units. If False, the time is adjusted by the trial start time. Default is True.
    return_unit : str, optional
        The unit to return the onset and offset times in. Options are "samples", "frames", "seconds", or "milliseconds". Default is "samples".

    Returns:
    --------
    numpy.ndarray
        A 2-element array containing the onset and offset of the first matching breath type call,
        adjusted for the buffer and in the requested unit. Returns np.nan if no such call is found.

    Raises:
    -------
    ValueError
        If an invalid unit is provided in the `return_unit` parameter.

    Notes:
    ------
    - If no breath of the specified type is found after the `earliest_allowed_onset`, a warning is raised and np.nan is returned.
    - The resulting onset and offset times are either in stimulus-aligned or trial time, depending on `return_stim_aligned`.
    - The time is adjusted by the specified `buffer_s` and converted to the requested unit (`samples`, `frames`, `seconds`, or `ms`).

    Example:
    --------
    get_first_breath_segment(trial, 'inhalation', fs=1000, earliest_allowed_onset=1.0, buffer_s=0.5, return_unit='seconds')
    """

    # select call type of interest
    ii_breath_type = np.array(trial["call_types"]) == breath_type

    # get onsets & offsets
    breath_times = trial["call_times_stim_aligned"][ii_breath_type, :]

    # reject calls that are too early
    if earliest_allowed_onset is not None:
        breath_times = breath_times[breath_times[:, 0] >= earliest_allowed_onset]

    if len(breath_times) == 0:
        warnings.warn(
            f"No calls of type `{breath_type}` found after `{earliest_allowed_onset}s post-stimulus`."
        )
        return np.nan

    # get earliest matching call in samples
    i_earliest_onset = breath_times[:, 0].argmin()

    earliest_call = breath_times[i_earliest_onset, :]

    if not return_stim_aligned:
        earliest_call += trial["trial_start_s"]

    # add buffer
    earliest_call[0] -= buffer_s
    earliest_call[1] += buffer_s

    # convert to requested unit
    if return_unit in ["samples", "frames"]:
        earliest_call = (earliest_call * fs).astype(int)
    elif return_unit in ["seconds"]:
        pass
    elif return_unit in ["milliseconds", "ms"]:
        earliest_call = earliest_call / 1000
    else:
        raise ValueError(f"Unknown unit: {return_unit}")

    return earliest_call


def get_segment_duration(trial, df, rel_index=[-2, -1]):
    """
    Get the total duration of breath segments in rel_index (where
    current == 0).

    Eg, for an expiration (n=0), default [-2, -1] will return the
    duration of the previous expiration and previous inspiration.
    (-2 = prev exp, -1 = prev insp).
    """

    # duration for: [prev exp, prev insp]
    durations = np.array(
        [
            loc_relative(*trial.name, df, field="duration_s", i=i, default=np.nan)
            for i in rel_index
        ]
    )

    return durations.sum()


def get_wav_snippet_from_numpy(
    breath_trial,
    window_fr,
    fs,
    trace_folder,
    channel=None,
    alignment_col_name="start_s",
    error_value=pd.NA,
):
    """
    Given the breath in "breath_trial", load a requested window from a preprocessed numpy file. Window is centered on breath onset, with "post" time including breath length.

    Presumes numpy file is in trace_folder, which contain .npy copies of the .wav file in breath_trial.name sans folder structure.
    """

    # get file
    wav_file = breath_trial.name[0]
    np_file = trace_folder.joinpath(Path(wav_file).stem + ".npy")
    breath = np.load(np_file)

    # get indices
    # alignment_point_fr: usually onset or offset of this breath, in frames (0=file start)
    alignment_point_fr = int(fs * breath_trial[alignment_col_name])
    ii = np.arange(*window_fr) + alignment_point_fr

    try:
        # select channel
        if channel is None or breath.squeeze().ndim == 1:
            breath = breath[ii]
        else:
            breath = breath[channel, ii]

        return breath

    except IndexError:  # usually: this breath occurs at start/end of file
        return error_value


def loc_relative(*args, **kwargs):
    """
    alias for utils.umap > loc_relative()
    """

    return umap__loc_relative(*args, **kwargs)


def make_notmat_vars(
    exps,
    insps,
    last_offset,
    exp_label="e",
    insp_label="i",
):
    """
    Generates the onset, offset, and label information for inspiration and expiration phases.

    Given the onsets of inspiration and expiration, this function combines them into a sorted list of
    events, and generates corresponding offset times. It also assigns a label ("e" for expiration, "i" for
    inspiration, by defaukt) to each event.

    Args:
        exps (numpy.ndarray): A 1D array of indices where expiration phases start.
        insps (numpy.ndarray): A 1D array of indices where inspiration phases start.
        last_offset (int): The last offset time (in samples) to complete the offset list.
        exp_label, insp_label (string): string/character with which to label exps/insps in `labels`

    Returns:
        tuple: A tuple containing:
            - onsets (numpy.ndarray): A sorted array of onset times (inspiration and expiration).
            - offsets (numpy.ndarray): A corresponding array of offset times.
            - labels (numpy.ndarray): An array of labels ("e" for expiration, "i" for inspiration)
                                      corresponding to the onsets.

    Example:
        >>> exps = np.array([10, 50, 90])
        >>> insps = np.array([30, 70, 110])
        >>> last_offset = 120
        >>> onsets, offsets, labels = make_notmat_vars(exps, insps, last_offset)
        >>> print("Onsets:", onsets)
        >>> print("Offsets:", offsets)
        >>> print("Labels:", labels)
    """

    # Combine and sort the onsets
    onsets = np.concatenate([exps, insps])
    labels = np.array([exp_label] * len(exps) + [insp_label] * len(insps))

    # Sort by onset time
    ii_sort = onsets.argsort()
    onsets = onsets[ii_sort]
    labels = labels[ii_sort]

    offsets = np.append(onsets[1:] - 1, last_offset)

    return (onsets, offsets, labels)
