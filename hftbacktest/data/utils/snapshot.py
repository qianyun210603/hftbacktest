import numpy as np

from ... import HftBacktest, Linear, ConstantLatency
from ...reader import UNTIL_END_OF_DATA


def create_last_snapshot(
        data,
        tick_size,
        lot_size,
        initial_snapshot=None,
        output_snapshot_filename=None
):
    # Just to reconstruct order book from the given snapshot to the end of the given data.
    hbt = HftBacktest(
        data,
        tick_size,
        lot_size,
        0,
        0,
        ConstantLatency(0, 0),
        Linear,
        snapshot=initial_snapshot
    )

    # Go to the end of the data.
    hbt.goto(UNTIL_END_OF_DATA)

    snapshot = []
    snapshot += [[
        4,
        hbt.last_timestamp,
        -1,
        1,
        float(bid * tick_size),
        float(qty)
    ] for bid, qty in sorted(hbt.bid_depth.items(), key=lambda v: -float(v[0]))]
    snapshot += [[
        4,
        hbt.last_timestamp,
        -1,
        -1,
        float(ask * tick_size),
        float(qty)
    ] for ask, qty in sorted(hbt.ask_depth.items(), key=lambda v: float(v[0]))]

    snapshot = np.asarray(snapshot, np.float64)

    if output_snapshot_filename is not None:
        np.savez(output_snapshot_filename, data=snapshot)

    return snapshot
