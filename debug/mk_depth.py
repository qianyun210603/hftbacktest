from numba import njit
import pickle
from hftbacktest import HftBacktest, FeedLatency, Linear

# @njit
def print_3depth(hbt):
    while hbt.run:
        if not hbt.elapse(1 * 1e6):
            return False

        # a key of bid_depth or ask_depth is price in tick format.
        # (integer) price_tick = price / tick_size
        print('current_timestamp:', hbt.current_timestamp)
        i = 0
        for tick_price in range(hbt.best_ask_tick, hbt.high_ask_tick + 1):
            if tick_price in hbt.ask_depth:
                print(
                    'ask: ',
                    hbt.ask_depth[tick_price],
                    '@',
                    round(tick_price * hbt.tick_size, 3)
                )
                i += 1
                if i == 3:
                    break
        i = 0
        for tick_price in range(hbt.best_bid_tick, hbt.low_bid_tick - 1, -1):
            if tick_price in hbt.bid_depth:
                print(
                    'bid: ',
                    hbt.bid_depth[tick_price],
                    '@',
                    round(tick_price * hbt.tick_size, 3)
                )
                i += 1
                if i == 3:
                    break
    return True

if __name__ == "__main__":
    with open(r"D:\Documents\TradeResearch\Stock\convert_market_making\converted_data\sh110043_20230301.pkl",
              "rb") as f:
        df = pickle.load(f)
    hbt = HftBacktest(
        df,
        tick_size=0.1,
        lot_size=0.001,
        maker_fee=0.0002,
        taker_fee=0.0007,
        order_latency=FeedLatency(),
        asset_type=Linear,
    )

    print_3depth(hbt)