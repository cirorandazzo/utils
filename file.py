# file.py
#
#


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
