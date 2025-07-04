# ./utils/video.py
# 2024.05.13 CDR
#
# Functions related to processing callback experiment videos
#

import numpy as np
import cv2 as cv

from .audio import get_triggers_from_audio


def get_video_frames_from_callback_audio(
    camera_channel_audio: np.ndarray,
    threshold_function=lambda x: np.max(x) * 0.2,
    allowable_range=10,
    **kwargs,
) -> np.ndarray:
    """
    - takes ONE CHANNEL of audio as a numpy array
    - return all frames when audio starts dips below threshold (ie, frame i for i<=thresh iff (i-1)>thresh)
    - if do_check is True, ensures that range of inter-frame intervals <= 10
    """

    frames = get_triggers_from_audio(
        audio=camera_channel_audio,
        threshold_function=threshold_function,
        crossing_direction="down",
        allowable_range=allowable_range,
        **kwargs,
    )

    return frames


def open_video(video_path: str) -> cv.VideoCapture:

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(video_path))

    if not capture.isOpened():
        FileNotFoundError(f"Unable to open: {video_path}")

    return capture


def get_video_size(capture: cv.VideoCapture) -> np.ndarray:

    import numpy as np

    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    return np.array((width, height)).astype("int")


def get_video_fourcc(capture: cv.VideoCapture) -> str:

    h = int(capture.get(cv.CAP_PROP_FOURCC))

    codec = (
        chr(h & 0xFF)
        + chr((h >> 8) & 0xFF)
        + chr((h >> 16) & 0xFF)
        + chr((h >> 24) & 0xFF)
    )

    return h


def get_video_params(capture: cv.VideoCapture) -> dict:

    import numpy as np

    params = {
        "fourcc": get_video_fourcc(capture),
        "fps": int(capture.get(cv.CAP_PROP_FPS)),
        "frameSize": get_video_size(capture),
    }

    return params


def write_diff_video(in_file_name: str, out_file_name: str) -> None:

    import numpy as np

    capture = open_video(in_file_name)
    params = get_video_params(capture)

    cvtColor = cv.COLOR_BGR2GRAY  # cv.COLOR_RGB2GRAY uses weighted avg of RGB

    video_out = cv.VideoWriter()  # create new video
    video_out.open(filename=out_file_name, **params, isColor=0)

    ret, prev_frame = capture.read()
    prev_frame = cv.cvtColor(prev_frame, cvtColor)

    while True:
        ret, frame = capture.read()

        if frame is None:
            break
        else:
            frame = cv.cvtColor(frame, cvtColor)
            new_frame = cv.absdiff(frame, prev_frame)

            # THRESHOLD
            # new_frame = new_frame[new_frame > 10]

            video_out.write(new_frame)
            prev_frame = frame

    video_out.release()

    return


def play_video(video_path: str, max_frames=np.inf) -> None:

    import numpy as np

    cv.startWindowThread()

    capture = open_video(video_path)
    t_frame = round(1000 / capture.get(cv.CAP_PROP_FPS))

    fr_num = 0
    while fr_num < max_frames:
        ret, frame = capture.read()
        fr_num += 1

        if frame is None:
            break

        cv.imshow(video_path, frame)

        keyboard = cv.waitKey(t_frame)
        if keyboard == "q" or keyboard == 27:
            break

    cv.waitKey(1)
    cv.destroyAllWindows()
    cv.waitKey(1)
