# file.py
#

import re


def multi_index_from_dict(df, index_dict, keep_current_index=True):
    """
    Add Multiindex columns to a pd dataframe given values from a dict. Useful before merging dataframes from separate timepoints, for example.
    """
    import pandas as pd

    df_indexed = df.copy()

    for k, v in index_dict.items():
        df_indexed[k] = v

    index_keys = list(index_dict.keys())
    if keep_current_index:
        index_keys.append(df.index)

    df_indexed.set_index(index_keys, inplace=True)

    return df_indexed


def load_syllable_mat(
    filename,
    parse_nextSyl=True,
):
    import pandas as pd

    from pymatreader import read_mat

    data = read_mat(filename)

    data = data["by_syllable"]

    data = pd.DataFrame.from_dict(data)

    data.set_index(keys=["syl", "i"], inplace=True)
    data.sort_index(inplace=True)
    data.drop(
        columns=[
            "MotifAudio",
            "deriv1_frame_aligned",
            "deriv2_frame_aligned",
        ],
        inplace=True,
    )

    if parse_nextSyl:
        data["nextSyl"] = data["postSyls"].apply(_get_next_syl)

    return data


def load_burst_mat(file):
    import numpy as np
    import pandas as pd

    from pymatreader import read_mat

    data = read_mat(file)

    bp_syl_num = data.pop("BPsylNum")

    index = pd.MultiIndex.from_arrays(
        arrays=(
            range(len(data["burst_roi"])),
            data["burst_roi"].astype(int),
            data["burst_time_sub"],
        ),
        names=("burst_id", "roi", "time"),
    )

    [data.pop(x) for x in ["burst_branch", "burst_time", "finalLocs"]]

    df = pd.DataFrame()

    for tr in range(data["burst_deriv"].shape[1]):
        df = pd.concat(
            [
                df,
                pd.DataFrame.from_dict(
                    {
                        "trial": tr,
                        "burst_dff": data["burst_dff"][:, tr],
                        "burst_deriv": data["burst_deriv"][:, tr],
                        "branch_id": int(data["ID"][tr]),
                    },
                ).set_index(["trial", index]),
            ]
        )

    df["syl"] = np.floor(df.index.get_level_values("time")).astype(int)

    # remove too-early stuff & post-branch stuff
    df = df.loc[(df["syl"] >= -1 * bp_syl_num) & (df["syl"] <= 0)]

    # and make positive for consistency with previous code
    df["syl"] = np.abs(df["syl"])

    return df


def _get_next_syl(postSyls):
    if len(postSyls) > 1:
        return postSyls[1]
    else:
        return "END"


def convert_image(
    filepath: str,
    convert_type="png",
    return_without_path=False,
) -> str:
    """
    Converts an image file to the specified format using Inkscape.

    This function takes a file path to an image and calls Inkscape to convert that file
    into the desired format (default is PNG). If the input file is already in the desired
    format, it simply returns the original file path. The function can also return just
    the filename without the path if specified.

    Parameters:
    ----------
    filepath : str
        The full path to the image file that needs to be converted.

    convert_type : str, optional
        The desired output format of the image. Defaults to "png". Other formats may be supported
        depending on Inkscape's capabilities.

    return_without_path : bool, optional
        If True, the function returns only the filename of the converted image, without the
        directory path. Defaults to False.

    Returns:
    -------
    str
        The full path to the converted image if `return_without_path` is False, or the
        filename of the converted image if `return_without_path` is True.

    Raises:
    ------
    subprocess.CalledProcessError
        If the Inkscape command fails to execute.

    Notes:
    -----
    This function requires Inkscape to be installed and accessible from the command line.
    Ensure that the file path provided points to a valid image file.
    """
    import os
    import subprocess

    if filepath.endswith("png"):  # is already png
        filepath_new = filepath
    else:
        filepath_new = os.path.splitext(filepath)[0] + "." + convert_type

        subprocess.check_call(
            [
                "inkscape",
                "--export-filename",
                filepath_new,
                filepath,
            ]
        )

    if return_without_path:  # return filename without path
        filepath_new = os.path.split(filepath_new)[-1]

    return filepath_new


def parse_parameter_from_string(
    string,
    parameter_name,
    chars_to_ignore=1,
    return_nan=False,
):
    """
    Note: rejects `chars_to_ignore` characters after parameter_name match, eg, 1 if there's a symbol there (eg, "max_features_7" --> "7")
    """
    import re

    import numpy as np

    whole_match = re.search(rf"({parameter_name})(\w\.?)+", string)

    if whole_match is not None:
        # return everything after "parameter_name"
        param = whole_match[0][len(parameter_name) + chars_to_ignore :]
    else:
        if return_nan:
            param = np.nan
        else:
            raise KeyError(
                f"Parameter `{parameter_name}` not found in string `{string}`"
            )

    return param


def parse_birdname(
    string,
    birdname_regex=r"([a-z]{1,2}[0-9]{1,2}){2}",
):
    """
    Cuts out typical bird identifier from a string. Default format: AA#(#)AA#(#), where A is a letter, # is an obligate number, and (#) is an optional number.
    """
    matches = re.findall(birdname_regex, string)

    if len(matches) == 0:
        raise BirdNameException(f"Could not parse birdname from string: {string}")

    # assert there's a single unique match
    matches = list(set(matches))

    if len(matches) > 1:
        raise BirdNameException(
            f"Found multiple birdname matches in string: {string}. Matches: {' | '.join(matches)}"
        )
    else:
        return matches[0]


class BirdNameException(ValueError):
    pass
