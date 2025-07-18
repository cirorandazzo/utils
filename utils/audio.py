from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, ShortTimeFFT
from scipy.signal.windows import hamming
from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt

from .evfuncs import load_cbin


class AudioObject:
    def __init__(
        self,
        audio,
        fs,
        b=None,
        a=None,
        do_spectrogram=False,
    ):
        """
        b,a: Numerator (b) and denominator (a) polynomials of the IIR filter
        """

        self.audio = audio
        self.fs = fs
        self.audio_frs = None
        self.name = None

        self.file = None
        self.name = None
        self.channel = None

        if b is not None and a is not None:
            self.filtfilt(b, a)
        else:
            self.audio_filt = None

        if do_spectrogram and self.audio_filt is not None:
            self.make_spectrogram()
        else:
            self.spectrogram = None

    @classmethod
    def from_file(
        cls,
        filename,
        **kwargs,
    ):
        """
        Tries to determine file type from filename and call relevant loader.
        """

        ext = str(Path(filename).suffix).lower()

        if ext == ".wav":
            return cls.from_wav(filename=filename, **kwargs)
        elif ext == ".cbin":
            return cls.from_cbin(filename=filename, **kwargs)
        elif ext == "":
            raise ValueError("Received a directory! Must receive an audio file.")
        else:
            raise ValueError(f"Unknown file extension: {ext}")

    @classmethod
    def from_wav(
        cls,
        filename,
        channel=0,
        channel_names=None,
        **kwargs,
    ):
        """
        Create an AudioObject given a wav file & IIR filter details.

        Args should match default constructor.
        """

        fs, audio = wavfile.read(filename)

        if "channels" in kwargs.keys():
            channel = kwargs.pop("channels")
            raise Warning("`channels` has been renamed channel.")

        if channel == "all":
            channel = np.arange(audio.shape[1])
        else:
            channel = np.array([channel]).flatten()

        if channel_names is None:
            channel_names = [None] * len(channel)
        else:
            assert len(channel_names) == len(
                channel
            ), f"Loading {len(channel)} channels, but only {len(channel_names)} channel_names given."

        objs = []

        for i, c in enumerate(channel):
            new_obj = cls(
                audio[:, c],
                fs,
                **kwargs,
            )

            new_obj.file = filename
            new_obj.name = channel_names[i]
            new_obj.channel = c

            objs += [new_obj]

        if len(objs) == 1:
            objs = objs[0]

        return objs

    @classmethod
    def from_cbin(
        cls,
        filename,
        channel=0,
        **kwargs,
    ):
        """
        Create an AudioObject given a cbin file. Defaults to first channel (0) which can be changed as a keyword arg.

        Args should match default constructor.
        """

        audio, fs = load_cbin(filename, channel=channel)

        n_channels = audio.shape[0]

        if audio.ndim == 1 or n_channels == 1:
            new_obj = cls(audio, fs, **kwargs)
            new_obj.file = filename

        else:
            # if multiple channels, create a new object for each channel
            # and return a list of objects

            new_obj = [None] * n_channels

            for i in range(n_channels):
                new_obj[i] = cls(audio[i, :], fs, **kwargs)
                new_obj[i].file = filename

        return new_obj

    @classmethod
    def plot_file(cls, filename, channel="all"):
        aos = cls.from_file(filename, channel=channel)

        if not isinstance(aos, list):
            aos = [aos]

        fig, axs = plt.subplots(nrows=len(aos), sharex=True)

        for ch, ax in zip(aos, axs):
            ax.plot(ch.get_x(), ch.audio)

        axs[0].set_title(filename, wrap=True)
        axs[-1].set(xlabel="time (s)")

        return fig, axs, aos

    def filtfilt(self, b, a):
        self.audio_filt = filtfilt(b, a, self.audio)

    def filtfilt_butter_default(self, f_low=500, f_high=15000, poles=8):
        b, a = butter(poles, [f_low, f_high], btype="bandpass", fs=self.fs)

        self.filtfilt(b, a)

    def rectify_smooth(self, smooth_window_f):
        if self.audio_filt is None:
            raise UnfilteredException()

        rectified = np.power(self.audio_filt, 2)

        # cumsum = np.cumsum(np.insert(rectified, 0, 0))
        # smoothed = (cumsum[smooth_window_f:] - cumsum[:-smooth_window_f]) / float(smooth_window_f)

        wind = np.ones(smooth_window_f)
        smoothed = np.convolve(wind, rectified, "same")

        self.audio_frs = smoothed

    def make_spectrogram(
        self,
        n=1024,
        overlap=1020,
        normalize_range=(0, 1),
    ):
        if self.audio_filt is None:
            raise UnfilteredException()

        window = hamming(n)
        hop = n - overlap

        self.SFT = ShortTimeFFT(
            window,
            hop,
            self.fs,
            fft_mode="onesided",
        )

        spx = self.SFT.spectrogram(self.audio_filt)
        spx = np.log10(spx)

        if normalize_range is not None:
            spx = normalize(spx, normalize_range)

        self.spectrogram = spx

    def plot_spectrogram(self, **kwargs):
        if self.spectrogram is None:
            self.make_spectrogram()

        return plot_spectrogram(self.spectrogram, self.SFT, **kwargs)

    def get_sample_times(self):
        """
        Return an array of sample times; eg, to use when plotting.
        """
        return np.arange(len(self.audio)) / self.fs

    def get_x(self):
        """
        Alias for get_sample_times
        """
        return self.get_sample_times()

    def get_length_s(self):
        """
        Return length of audio in s.
        """

        return len(self.audio) / self.fs


def get_triggers_from_audio(
    audio: np.ndarray,
    threshold_function=lambda x: 10 * np.mean(x),
    crossing_direction="up",
    allowable_range=None,
    ignore_range_fr=None,
) -> np.ndarray:
    import numpy as np

    threshold = threshold_function(audio)

    if crossing_direction == "down":  # get downward crossings; eg, video frames
        a_thresholded = audio <= threshold
    elif crossing_direction == "up":  # get upward crossings
        a_thresholded = audio >= threshold
    else:
        raise ValueError(
            f"'{crossing_direction}' is not a valid crossing_direction. Must be 'up' or 'down'."
        )

    # one frame prior. don't allow first frame
    offset = np.append([1], a_thresholded[:-1])

    frames = np.nonzero(a_thresholded & ~offset)[0]

    if ignore_range_fr is not None:
        frames = np.array([f for f in frames if f not in range(ignore_range_fr)])

    if allowable_range is not None:
        # number of audio samples between subsequent frames
        deltas = frames[1:] - frames[:-1]

        check = False
        r = np.ptp(deltas)

        if np.isscalar(allowable_range):  # just one number
            check = r <= allowable_range
        else:
            check = (r >= allowable_range[0]) and (r <= allowable_range[1])

        assert (
            check
        ), "Warning! Frame timings might vary too much. You might need a stricter threshold function."

    return frames


def plot_spectrogram(
    spectrogram: np.ndarray,
    SFT: np.ndarray,
    ax=None,
    x_offset_s=0,
    **plot_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    # extent: times of audio signal (s) & frequencies (Hz). for correct axis labels
    extent = np.array(SFT.extent(SFT.hop * spectrogram.shape[1])).astype("float")
    extent[0:2] += x_offset_s  # offset x axis

    if "cmap" not in plot_kwargs.keys():
        plot_kwargs["cmap"] = "bone_r"

    ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        extent=extent,
        **plot_kwargs,
    )

    ax.set(
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
    )

    return ax


def normalize(x, range=(-1, 1)):
    flattened = minmax_scale(x.flatten(), feature_range=range).astype("float32")
    return flattened.reshape(x.shape)


class UnfilteredException(Exception):
    def __init__(self):
        self.message = "Filter audio with self.filtfilt before calling this method!"
