from multiprocessing import Pool
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd


def load_segment(
    row,
    fs=None,
    **cut_segment_kwargs,
):
    """
    Given a row of all_breaths df, return the (appropriately processed) trace. Useful for 1 trial (or standard loops).

    row: df row.
    fs: sample rate. if none, tries to fetch from row["fs"]
    **cut_segment_kwargs: kwargs passed on to cut_segment, which handles the logic + processing of traces
    """

    if fs is None:
        fs = row["fs"]

    data = np.load(row["numpy_filename"])

    segment = cut_segment(data, row, fs, **cut_segment_kwargs)

    return segment


def cut_segment(
    data,
    row,
    fs,
    interpolate_length=None,
    pad_frames=[0, 0],
):
    """
    Process a segment from a preloaded npy array.

    Args:
        data (np.array): timeseries from which to cut
        row (pd.Series): A row from the DataFrame containing metadata.
        fs (int): Sampling rate of the data.
        interpolate_length (int): Length to interpolate the segment to. None or 0 to prevent interpolation.
        pad_frames (int or list): [pad_start, pad_end] extra frames before & after call bounds. Both should be positive to expand window. Alternatively, pass one int that's applied to both ends.
        data_row (int): row to consider from npy file.

        TODO: option for overflow

    Returns:
        np.ndarray: The processed segment.
    """

    # parse pad_frames
    if isinstance(pad_frames, int):
        pad_start = pad_frames
        pad_end = pad_frames
    else:
        pad_start, pad_end = pad_frames

    # get indices
    start_idx = int(row["start_s"] * fs) - pad_start
    end_idx = int(row["end_s"] * fs) + pad_end

    # Extract the segment
    segment = data[start_idx:end_idx]

    # Interpolate to normalized length
    if interpolate_length is not None:
        l = len(segment)
        segment = np.interp(
            np.linspace(0, l, interpolate_length),
            np.arange(l),
            segment,
        )

    return segment


def process_all_segments(
    df,
    data_row,
    interpolate_length,
    pad_frames,
    pickle_save_directory=None,
    n_jobs=4,
):
    """
    Process all segments in the DataFrame using multiprocessing.

    Args:
        df (pd.DataFrame): DataFrame containing segment metadata. Should include columns: [numpy_filename, fs, start_s, end_s]
        n_jobs (int): Number of parallel processes.

        pickle_save_directory: Path to save a copy of df for each individual file. If None, skips saving individual files.
            TODO: check whether this file's df exists; if so, use that to save some time.

        See `load_segment` for other arguments.

    Returns:
        list: List of processed segments.
    """

    st = time.time()

    # assert necessary columns
    required_columns = ["numpy_filename", "fs", "start_s", "end_s"]
    for col in required_columns:
        assert col in df.columns, f"df missing required column: {col}"

    n_files = len(df["numpy_filename"].unique())

    # prepare output
    all_df = []

    print("Starting pool...")

    # load each and every
    with Pool(n_jobs) as pool:

        print(f"Pool started! [total elapsed: {time.time() - st}s]")

        for i, (file, df_file) in enumerate(df.groupby(by="numpy_filename")):
            fstem = Path(file).stem
            print(f"File {i}/{n_files} started. [total elapsed: {time.time() - st}s]")
            print(f"\t{fstem}")

            data = np.load(file)[data_row, :]

            print(f"\tMaking inputs... [total elapsed: {time.time() - st}s]")
            input = [  # goes into _load_segment_multiproc
                (
                    data,
                    record,
                    record["fs"],
                    interpolate_length,
                    pad_frames,
                )
                for record in df_file.reset_index().to_dict("records")
            ]

            print(f"\tGetting traces... [total elapsed: {time.time() - st}s]")
            out = pool.map(
                _load_segment_multiproc,
                input,
            )

            name, segments, errors = zip(*out)

            idx = pd.MultiIndex.from_tuples(
                name, names=("audio_filename", "calls_index")
            )
            df_file = pd.DataFrame(index=idx, data={"data": segments, "error": errors})

            if pickle_save_directory is not None:
                pickle_save_path = Path(pickle_save_directory) / f"{fstem}.pickle"
                df_file.to_pickle(pickle_save_path)

                print(f"\tSaved file df! [total elapsed: {time.time() - st}s]")
                print(f"\tSave path: {pickle_save_path}")

            all_df.append(df_file)
            print(f"\tFinished file! [total elapsed: {time.time() - st}s]")

        print(f"Closing pool... [total elapsed: {time.time() - st}s]")
        pool.close()
        pool.join()

    print(f"Pool closed! Saving... [total elapsed: {time.time() - st}s]")

    # return as a df
    return pd.concat(all_df).sort_index()


def _load_segment_multiproc(input):
    """
    input should contain information on a single record as an array. In order:
    - data: np array of entire file
    - record: pd.DataFrame.to_dict("record") of a single row
    - fs: sample rate
    - interpolate_length: see cut_segment
    - pad_frames: see cut_segment

    cut_segment(data, row, fs, interpolate_length=None, pad_frames=[0, 0])
    """

    data, record, fs, interpolate_length, pad_frames = input
    name = [record[c] for c in ["audio_filename", "calls_index"]]

    exc = None
    try:
        segment = cut_segment(*input)
    except Exception as e:
        segment = np.nan
        exc = e

    # # Report
    # if exc is None:
    #     shape = segment.shape
    # else:
    #     shape = "nan"

    # print(f"\t\t- {record['calls_index']}\t | {shape} \t | {exc}")

    return (name, segment, exc)


def load_df_from_segments(
    df, pickle_save_directory, filename_column="numpy_filename", skip_not_found=False
):
    """
    Given trace dfs for individual files in df, return a merged df of traces.
    """

    # get all filenames
    files = df[filename_column].unique()
    data = []

    # load individual file dfs
    # named like df[filename_column], but in traces folder with extension .pickle
    for fname in files:
        try:
            df_fname = Path(pickle_save_directory) / f"{Path(fname).stem}.pickle"

            with open(df_fname, "rb") as f:
                data.append(pickle.load(f))

        except FileNotFoundError as e:
            if not skip_not_found:
                raise e

    # return merged df
    df = pd.concat(data).sort_index()
    return df
