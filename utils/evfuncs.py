"""
ev_funcs
Python implementations of functions used with EvTAF and evsonganaly.m

thanks annie :^)
"""

# from pathlib import Path

import numpy as np
import os


def readrecf(filename):
    """reads .rec files output by EvTAF
    and returns rec_dict with (approximately)
    the same structure as the rec struct
    returned by readrec.m

    Parameters
    ----------
    filename : str, Path
        name of .rec file, can include path

    Returns
    -------
    rec_dict : dict
        with following key, value pairs
            header : str
                header from .rec file
            time_before : float
                time in seconds of additional audio added to file by EvTAF
                from before sound amplitude went above threshold for recording
            time_after : float
                time in seconds of additional audio added to file by EvTAF
                after sound amplitude dropped below threshold for recording
            iscatch : bool
                list of whether each occurrence of auditory feedback triggered
                was a 'catch trial' (where feedback would have been triggered but
                was withheld so behavior on that trial could be measured)
            num_channels : int
                number of channels from DAQ board recored by EvTAF
            sample_freq : int
                sampling frequency of audio file associated with this .rec file, in Hertz
            num_samples : int
                number of audio samples in file
            outfile : str
                name of audio file played as auditory feedback (if there was any)
            thresholds : int
                thresholds used by ring counter to determine whether to trigger auditory feedback
            feedback_info : dict
                maps feedback type to time it occurred in milliseconds

    Examples
    --------
    >>> recf = 'gy6or6_baseline_230312_0808.138.rec'
    >>> rec_dict = readrecf(recf)
    >>> num_samples = rec_dict['num_samples'}
    >>> sample_freq = rec_dict['sample_freq'}
    >>> print(f"file duration in seconds: {num_samples / sample_freq:.3f}")
    file duration in seconds: 12.305
    """
    # filename = Path(filename)

    rec_dict = {}
    with open(filename, "r") as recfile:
        line_tmp = ""
        while 1:
            if line_tmp == "":
                line = recfile.readline()
            else:
                line = line_tmp
                line_tmp = ""

            if line == "":  # if End Of File
                break
            elif line == "\n":  # if blank line
                continue
            elif "Catch" in line:
                ind = line.find("=")
                rec_dict["iscatch"] = line[ind + 1 :]
            elif "Chans" in line:
                ind = line.find("=")
                rec_dict["num_channels"] = int(line[ind + 1 :])
            elif "ADFREQ" in line:
                ind = line.find("=")
                try:
                    rec_dict["sample_freq"] = int(line[ind + 1 :])
                except ValueError:
                    # if written with scientific notation
                    # first parse as float, then cast to int
                    sample_freq_float = float(line[ind + 1 :])
                    try:
                        rec_dict["sample_freq"] = int(sample_freq_float)
                    except ValueError:
                        raise ValueError(
                            "Couldn't convert following value for "
                            "ADFREQ in .rec file {} to an integer: "
                            "{}".format(filename, sample_freq_float)
                        )
            elif "Samples" in line:
                ind = line.find("=")
                rec_dict["num_samples"] = int(line[ind + 1 :])
            elif "T After" in line or "T AFTER" in line:
                ind = line.find("=")
                rec_dict["time_after"] = float(line[ind + 1 :])
            elif "T Before" in line or "T BEFORE" in line:
                ind = line.find("=")
                rec_dict["time_before"] = float(line[ind + 1 :])
            elif "Output Sound File" in line:
                ind = line.find("=")
                rec_dict["outfile"] = line[ind + 1 :]
            elif "Thresholds" in line or "THRESHOLDS" in line:
                th_list = []
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    try:
                        th_list.append(float(line))
                    except ValueError:  # because we reached next section
                        line_tmp = line
                        break
                rec_dict["thresholds"] = th_list
                if line == "":
                    break
            elif "Feedback information" in line:
                fb_dict = {}
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    elif line == "\n":
                        continue
                    ind = line.find("msec")
                    time = float(line[: ind - 1])
                    ind = line.find(":")
                    fb_type = line[ind + 2 :]
                    fb_dict[time] = fb_type
                rec_dict["feedback_info"] = fb_dict
                if line == "":
                    break
            elif "File created" in line:
                header = [line]
                for counter in range(4):
                    line = recfile.readline()
                    header.append(line)
                rec_dict["header"] = header
    return rec_dict


def load_cbin(filename, channel=0):
    """loads .cbin files output by EvTAF.

    Parameters
    ----------
    filename : str
        name of .cbin file, can include path
    channel : int
        Channel in file to load. Default is 0.

    Returns
    -------
    data : numpy.ndarray
        1-d vector of 16-bit signed integers
    sample_freq : int or float
        sampling frequency in Hz. Typically 32000.

    Examples
    --------
    >>> cbin_filename = 'gy6or6_baseline_230312_0808.138.cbin'
    >>> data, sample_freq = load_cbin(cbin_filename)
    >>> data
    array([-230, -223, -235, ...,   34,   36,   26], dtype=int16)
    """
    # filename = Path(filename)

    # .cbin files are big endian, 16 bit signed int, hence dtype=">i2" below
    filename = os.fspath(str(filename))
    data = np.fromfile(filename, dtype=">i2")
    recfile = os.path.splitext(filename)[0] + ".rec"
    rec_dict = readrecf(recfile)
    n_channels = rec_dict["num_channels"]

    data = data.reshape((-1, n_channels)).T

    if channel == "all":
        pass
    else:
        data = data[channel, :]

    sample_freq = rec_dict["sample_freq"]
    return data, sample_freq


def segment_notes(smooth, fs, min_int, min_dur, threshold):
    # % [ons,offs]=evsegment(smooth,Fs,min_int,min_dur,threshold);
    # % segment takes smoothed filtered song and returns vectors of note
    # % onsets and offsets values are in seconds
    #
    # 2024.11.11 CDR

    import numpy as np

    # threshold input
    notetimes = np.array(smooth) > threshold

    # get note onsets/offsets
    trans = np.convolve([1, -1], notetimes, "same")
    onsets = np.flatnonzero(trans > 0)
    offsets = np.flatnonzero(trans < 0)

    # return onsets, offsets
    if len(onsets) - len(offsets) == 1:  # extra onset, has to be at end
        onsets = onsets[:-1]
    elif len(offsets) - len(onsets) == 1:  # extra offset, has to be at start
        offsets = offsets[1:]
    elif len(onsets) == len(offsets):
        pass
    else:  # something funky.
        raise ValueError("Cannot rectify mismatch in length of onsets/offsets")

    # merge any calls closer than min_int ms
    temp_int = offsets[:-1] - onsets[1:] * 1000 / fs
    real_ints = temp_int > min_int
    onsets = onsets[np.insert(real_ints, 0, 1)]  # save first onset
    offsets = offsets[np.insert(real_ints, -1, 1)]  # save final offset

    # delete short calls (less than min_dur ms)
    temp_dur = (offsets - onsets) * 1000 / fs
    real_durs = temp_dur > min_dur
    onsets = onsets[real_durs]
    offsets = offsets[real_durs]

    return onsets / fs, offsets / fs
