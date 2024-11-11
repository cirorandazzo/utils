# callbacks.py
# 2024.05.13 CDR
#
# Functions related to loading & preprocessing callback data
#
# Renamed callbacks.py from deepsqueak.py

ESA_LOOKUP = {"c": "Call", "s": "Stimulus", "n": "Song", "z": "Song"}


def call_mat_stim_trial_loader(
    file,
    data=None,
    acceptable_call_labels=["Call", "Stimulus"],
    from_notmat=False,
    min_latency=0,
    calls_index_name="calls_index",
    stims_index_name="stims_index",
    stim_type_label="Stimulus",
    verbose=True,
):
    """
    Given (1) a .mat from DeepSqueak, (2) a .not.mat from evsonganaly, or (3) a dictionary which looks like loaded data from one of these, make a trial-by-trial dataframe of callbacks.
    """
    import numpy as np
    import pandas as pd

    from pymatreader import read_mat

    if verbose:
        print(f"Reading file: {file}")

    if (data is None) == (file is None):  # both or neither provided.
        raise ValueError("Exactly one of `data` or `file` must be provided.")
    elif data is not None:
        pass
    else:
        data = read_mat(file)

    calls = _read_calls_from_mat(data, from_notmat=from_notmat)
    file_info = _read_file_info_from_mat(data, from_notmat=from_notmat)

    calls.index.name = calls_index_name

    del data  # don't store twice, it's already saved elsewhere

    # new dataframe, each row maps to a single stimulus + its reponses
    if verbose:
        print("Constructing stim trial dataframe")

    stim_trials = construct_stim_trial_df(
        calls,
        audio_duration=file_info["wav_duration_s"],
        stim_type_label=stim_type_label,
        min_latency=min_latency,
        calls_index_name=calls_index_name,
        stims_index_name=stims_index_name,
    )

    if stim_trials is None:
        return None

    stim_trials["wav_filename"] = file_info["wav_filename"]

    # reject rows of stim_trials with bad call types
    if verbose:
        print(f"Rejecting call types not in: {acceptable_call_labels}")

    assert (
        stim_type_label in acceptable_call_labels
    ), f"Warning! Using label `{stim_type_label}` used to align trials but not listed as an acceptable call type."

    stim_trials, rejected_trials, call_types = reject_stim_trials(
        stim_trials,
        calls,
        acceptable_call_labels,  # you can't just reject stimuli...
    )

    if verbose:
        print(f"\t- # trials rejected: {len(rejected_trials)}")
        print(f"\t- # trials accepted: {len(stim_trials)}")
        print("")
        print(call_types)

    return calls, stim_trials, rejected_trials, file_info, call_types


def _read_calls_from_mat(
    data,
    from_notmat,
):
    """
    Reads calls from a .mat containing callback labels (either deepsqueak or evsonganaly .not.mat)
    """

    import numpy as np
    import pandas as pd

    if from_notmat:
        calls = pd.DataFrame()
        calls["start_s"] = data["onsets"] / 1000
        calls["end_s"] = data["offsets"] / 1000
        calls["duration_s"] = calls["end_s"] - calls["start_s"]
        calls["type"] = [ESA_LOOKUP.get(l, l) for l in data["labels"]]

    else:
        assert "Calls" in data.keys()

        calls = pd.DataFrame(data["Calls"])
        calls = calls[["start_s", "end_s", "duration_s", "type"]]  # reorder columns

    onsets = np.array(calls["start_s"])
    time_from_prev_onset = [np.NaN] + list(onsets[1:] - onsets[:-1])
    calls["time_from_prev_onset_s"] = time_from_prev_onset
    calls["type_prev_call"] = [None] + list(calls["type"][:-1])

    ii = (calls["type_prev_call"] == "Call") & (calls["type"] == "Call")

    calls["ici"] = calls.loc[ii]["time_from_prev_onset_s"]

    return calls


def _read_file_info_from_mat(
    data,
    from_notmat,
) -> dict:
    """
    Reads file metadata from a .mat containing callback labels (either deepsqueak or evsonganaly .not.mat)
    """
    import numpy as np
    import pandas as pd

    if from_notmat:
        # TODO: deal with file info
        file_info = dict(
            wav_duration_s=np.inf,
            wav_filename=None,
            # birdname=None,
            # d=-1,
            # block=-1,
        )

    else:
        assert "file_info" in data.keys()
        file_info = data["file_info"]

    return file_info


def construct_stim_trial_df(
    calls,
    audio_duration,
    stim_type_label,
    min_latency,
    calls_index_name="calls_index",
    stims_index_name="stims_index",
):
    """
    TODO: document
    """
    import pandas as pd
    import numpy as np

    stims = calls[calls["type"] == stim_type_label]

    stim_trials = pd.DataFrame()

    if len(stims) == 0:
        import warnings

        warnings.warn("No stimuli found in this file!")

        return None

    # trial start: stimulus onset
    stim_trials["trial_start_s"] = stims["start_s"]

    # trial end: onset of following stimulus
    trial_ends = list(stims.start_s[1:])
    trial_ends.append(audio_duration)
    stim_trials["trial_end_s"] = pd.Series(trial_ends, index=stims.index)

    stim_trials["stim_duration_s"] = stims["duration_s"]

    # get all labeled 'calls' in this range (may include song, wing flaps, etc)
    get_calls = lambda row: _get_calls_in_range(
        calls, row["trial_start_s"], row["trial_end_s"], exclude_stimulus=True
    )

    stim_trials["calls_in_range"] = stim_trials.apply(get_calls, axis=1)

    stim_trials["call_types"] = stim_trials["calls_in_range"].apply(
        lambda trial: [calls.loc[i, "type"] for i in trial]
    )

    stim_trials["call_times_stim_aligned"] = stim_trials.apply(
        _get_call_times,
        calls_df=calls,
        stimulus_aligned=True,
        axis=1,
    )

    stim_trials["n_calls"] = [
        sum(calls[:, 0] > 0) if len(calls) > 0 else 0
        for calls in stim_trials["call_times_stim_aligned"]
    ]

    onsets = [
        calls[:, 0] if len(calls) > 0 else np.array([])
        for calls in stim_trials["call_times_stim_aligned"]
    ]  # get all onsets for each trial, nan if no calls.
    onsets = [trial[trial > min_latency] for trial in onsets]

    stim_trials["latency_s"] = [
        np.min(trial) if len(trial) > 0 else np.nan for trial in onsets
    ]

    # stim_trials['latency_s'] = [np.min(calls[:,0]) if len(calls)>0 else np.nan for calls in stim_trials['call_times_stim_aligned']]

    # NOTE: does not just count call indices in `calls_in_range`, which can include calls that have onset before stimulus.

    # reindex stim trials by stim # in block, but store call # for each stim
    stim_trials[calls_index_name] = stim_trials.index
    stim_trials.index = pd.Index(range(len(stim_trials)), name=stims_index_name)

    return stim_trials


def _get_calls_in_range(calls, range_start, range_end, exclude_stimulus=True):
    """
    TODO: document

    NOTE: range is exclusive to prevent inclusion of next stimulus, since range_end is defined by start of next stimulus
    """
    import numpy as np
    import pandas as pd

    # either start or end in range is sufficient.
    time_in_range = lambda t: (t > range_start) & (t < range_end)

    start_in_range = calls["start_s"].apply(time_in_range)
    end_in_range = calls["end_s"].apply(time_in_range)

    # or, 'call' starts before & lasts longer than trial (eg, a long song)
    check_encompass = lambda calls_row: (calls_row["start_s"] < range_start) & (
        calls_row["end_s"] > range_end
    )
    encompasses = calls.apply(check_encompass, axis=1)

    # check if any of these are true for each song
    call_in_range = start_in_range | end_in_range | encompasses

    if exclude_stimulus:
        i_stim = calls["type"] == "Stimulus"
        call_in_range = call_in_range & ~i_stim

    # return indices of calls in range
    return list(calls[call_in_range].index.get_level_values("calls_index"))


def _get_call_times(trial, calls_df, stimulus_aligned=True):
    """
    TODO: document

    given one row/trial from stim_trials df and calls df, get on/off times for all calls in that trial. if stim_aligned is True, timings are adjusted to set stimulus onset as 0.
    """

    import numpy as np

    call_ii = trial["calls_in_range"]

    call_times_stim_aligned = np.array(
        [[calls_df.loc[i]["start_s"], calls_df.loc[i]["end_s"]] for i in call_ii]
    )

    if stimulus_aligned:
        trial_start_s = trial["trial_start_s"]
        call_times_stim_aligned -= trial_start_s

    return call_times_stim_aligned


def reject_stim_trials(
    stim_trials,
    calls,
    acceptable_call_labels,
):
    """
    Given stim-aligned dataframe, exclude all trials with call types other than those in acceptable_call_labels)
    """
    import numpy as np
    import pandas as pd

    call_types = stim_trials["calls_in_range"].apply(
        lambda x: calls["type"].loc[x].value_counts()
    )  # get call types

    keep_columns = [c for c in call_types.columns if c in acceptable_call_labels]
    call_types_rejectable = call_types.drop(
        columns=keep_columns
    )  # get only values of 'rejectable' calls

    to_reject = call_types_rejectable.apply(np.any, axis=1)

    rejected_trials = stim_trials[to_reject]
    stim_trials = stim_trials[~to_reject]

    return stim_trials, rejected_trials, call_types
