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
