# utils.breath.phase
#
# Functions related to phase computation.
#

import numpy as np


def get_phase(
    breathDur,
    avgExpDur,
    avgInspDur,
    wrap=True,
):
    """
    python implementation of ziggy phase computation code ("phase2.m")

    Note: algorithm errors on breathDur > 2 normal breath lengths (for consistency with ZK code)

    breathDur: time between preceding expiration & event for which to compute phase - usually onset of call expiration. (formerly t_nMin1Exp_to_Call)
    avgExpDur: duration of expiration for comparison (in same unit as breathDur). Usually, this is the mean duration of non-call expirations. breathDur in [0, avgExpDur] (time) is mapped to [0, pi].  breathDur in [0, avgExpDur] (time) is mapped to [0, pi].
    avgInspDur: duration of inspiration for comparison (in same unit as breathDur). Usually, this is the mean duration of non-call inspirations. breathDur in [0, avgExpDur] (time) is mapped to [0, pi].  breathDur in [avgExpDur, avgExpDur+avgInspDur] (time) is mapped to [pi, 2pi].
    wrap: Behavior for breaths longer than average breath. If True, computes based on (breathDur MOD avgDur) - ie, constrains to 2pi. If False, breathDur longer than avgBreathDur return phase greater than 2pi. Default True (as in ZK implementation).
    """

    phase = None

    avgBreathDur = avgExpDur + avgInspDur
    dur_ratio = breathDur / avgBreathDur  # get it? duration ratio?

    assert (
        dur_ratio <= 2
    ), "algorithm only implmented for callT within 2 normal breath lengths!"

    if wrap:
        n = 0
    else:
        n = 2 * np.pi * np.floor(dur_ratio)

    breathDur = breathDur % avgBreathDur

    # call happens before the expiration before the call... (ie, oops)
    if breathDur < 0:
        phase = 0.1

    # call happens during this expiration
    elif breathDur < avgExpDur:
        # expiration is [0, pi]
        phase = np.pi * (breathDur / avgExpDur)

    # call happens during inspiration after that
    elif breathDur >= avgExpDur and breathDur < avgBreathDur:
        # inspiration is [pi, 2pi]
        phase = np.pi * (1 + (breathDur - avgExpDur) / avgInspDur)

    else:
        ValueError("this really shouldn't happen...")

    return n + phase
