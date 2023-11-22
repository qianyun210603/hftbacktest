import pandas as pd
import numpy as np
from datetime import time
import pickle
from loguru import logger
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm


EVENT_DF_COLS = ["event", "exch_timestamp", "local_timestamp", "side", "price", "qty"]


class LobException(Exception):
    pass


def calculate_auction_fill(df, trade_df_price=np.nan):
    # Initialize empty orderbook and filled orders
    orderbook_buy = df.loc[[1]].reset_index(level=1).copy()
    orderbook_sell = df.loc[[-1]].reset_index(level=1).copy()
    filled_amount = 0

    if orderbook_buy.empty:
        return orderbook_sell.set_index("price", append=True), 0, trade_df_price
    if orderbook_sell.empty:
        return orderbook_buy.set_index("price", append=True).sort_index(ascending=False), 0, trade_df_price
    if orderbook_sell.iloc[0]["price"] > orderbook_buy.iloc[-1]["price"]:
        return (
            pd.concat(
                [orderbook_sell.set_index("price"), orderbook_buy.set_index("price").sort_index(ascending=False)],
                axis=0,
                keys=[-1, 1],
                names=["side"],
            ),
            0,
            trade_df_price,
        )

    orderbook_idx = len(orderbook_buy) - 1
    # Iterate over rows in dataframe
    idx = 0
    fill_price = np.nan
    for idx in range(len(orderbook_sell)):
        if orderbook_idx < 0 or orderbook_sell.iloc[idx]["price"] > orderbook_buy.iloc[orderbook_idx]["price"]:
            break
        # Iterate over rows in orderbook
        while (
            orderbook_sell.iloc[idx, -1] > 0
            and orderbook_idx >= 0
            and orderbook_sell.iloc[idx]["price"] <= orderbook_buy.iloc[orderbook_idx]["price"]
        ):
            this_fill_amount = min(orderbook_sell.iloc[idx, -1], orderbook_buy.iloc[orderbook_idx]["qty"])
            orderbook_sell.iloc[idx, -1] -= this_fill_amount
            orderbook_buy.iloc[orderbook_idx, -1] -= this_fill_amount
            fill_price = orderbook_buy.iloc[orderbook_idx]["price"]
            filled_amount += this_fill_amount
            while orderbook_buy.iloc[orderbook_idx, -1] == 0:
                orderbook_idx -= 1
        if orderbook_idx < 0 or orderbook_sell.iloc[idx]["price"] > orderbook_buy.iloc[orderbook_idx]["price"]:
            break
    orderbook = pd.concat(
        [
            orderbook_sell.iloc[idx:].set_index("price"),
            orderbook_buy.iloc[: orderbook_idx + 1].set_index("price").sort_index(ascending=False),
        ],
        axis=0,
        keys=[-1, 1],
        names=["side"],
    )
    if not (
        orderbook_sell.iloc[idx]["price"] + 1e-8 >= fill_price >= orderbook_buy.iloc[orderbook_idx]["price"] - 1e-8
    ):
        fill_price = (
            orderbook_buy.iloc[orderbook_idx].price
            if orderbook_buy.iloc[orderbook_idx].qty < orderbook_sell.iloc[idx].qty
            else orderbook_sell.iloc[idx].price
            if orderbook_buy.iloc[orderbook_idx].qty > orderbook_sell.iloc[idx].qty
            else (orderbook_sell.iloc[idx].price + orderbook_buy.iloc[orderbook_idx].price) / 2
        )
    if not (pd.isna(trade_df_price) or np.isclose(fill_price, trade_df_price)):
        # logger.warning(f"Fill price {fill_price} not equal to trade price {trade_df_price}!")
        fill_price = trade_df_price

    return orderbook, filled_amount, fill_price


def parse_collection_auctions(collection_auction_order_df, trade_df_price=np.nan):
    auction_df = (
        collection_auction_order_df[["委托代码", "委托价格", "委托数量"]]
        .groupby(["委托代码", "委托价格"], group_keys=False)
        .sum()
        .rename(columns={"委托数量": "qty"})
    )
    auction_df = auction_df[auction_df["qty"] > 0].sort_index(ascending=[False, True])
    auction_df.index = auction_df.index.rename({"委托代码": "side", "委托价格": "price"})
    open_orderbook, filled_amount, fill_price = calculate_auction_fill(auction_df, trade_df_price)
    auction_end_timestamp = int(
        collection_auction_order_df["自然日_时间"].iloc[0].replace(minute=25, second=0, microsecond=0).timestamp() * 1000
    )
    depth_events_df = open_orderbook.reset_index()
    depth_events_df.insert(0, "event", 4)
    depth_events_df.insert(1, "exch_timestamp", auction_end_timestamp)
    depth_events_df.insert(2, "local_timestamp", -1)
    trade_events_df = pd.DataFrame(
        [
            [
                2,
                auction_end_timestamp,
                -1,
                -1 if fill_price == open_orderbook.loc[1].index.max() else 1,
                fill_price,
                filled_amount,
            ]
        ],
        columns=EVENT_DF_COLS,
    )
    return open_orderbook, trade_events_df, depth_events_df


def parse_continuous_auctions(open_orderbook, continuous_auction_order_df):
    orderbook = open_orderbook.copy(deep=True)
    depth_events_c, trade_events_c = [], []
    for dt, row in continuous_auction_order_df.iterrows():
        timestamp = int(dt.timestamp() * 1000)
        if row["委托类型"] == "A":
            while row["委托数量"] > 0 and -row["委托代码"] in orderbook.index and row["委托价格"] * row["委托代码"] >= row["委托代码"] * orderbook.loc[-row["委托代码"]].index[0]:
                fill_price = orderbook.loc[-row["委托代码"]].index[0]
                if row["委托数量"] >= orderbook.loc[(-row["委托代码"], fill_price), "qty"]:
                    row["委托数量"] -= orderbook.loc[(-row["委托代码"], fill_price), "qty"]
                    trade_events_c.append(
                        [2, timestamp, -1, row["委托代码"], fill_price, orderbook.loc[(-row["委托代码"], fill_price), "qty"]]
                    )
                    orderbook.drop((-row["委托代码"], fill_price), axis=0, inplace=True)
                    depth_events_c.append([1, timestamp, -1, -row["委托代码"], fill_price, 0])
                else:
                    trade_events_c.append([2, timestamp, -1, row["委托代码"], fill_price, row["委托数量"]])
                    orderbook.loc[(-row["委托代码"], fill_price), "qty"] -= row["委托数量"]
                    row["委托数量"] = 0
                    depth_events_c.append(
                        [
                            1,
                            timestamp,
                            -1,
                            -row["委托代码"],
                            fill_price,
                            orderbook.loc[(-row["委托代码"], fill_price), "qty"],
                        ]
                    )


            if row["委托数量"] <= 0:
                if row["委托数量"] < 0:
                    raise LobException(
                        f"process order {row['交易所委托号']} gives negative qty of {row['代码']} at {dt.isoformat()}"
                    )
                continue

            if (row["委托代码"], row["委托价格"]) in orderbook.index:
                orderbook.loc[(row["委托代码"], row["委托价格"]), "qty"] += row["委托数量"]
                depth_events_c.append(
                    [1, timestamp, -1, row["委托代码"], row["委托价格"], orderbook.loc[(row["委托代码"], row["委托价格"]), "qty"]]
                )
            else:
                if row["委托代码"] in orderbook.index:
                    position = (-row["委托代码"] * orderbook.loc[row["委托代码"]].index).searchsorted(-row["委托代码"] * row["委托价格"])
                else:
                    position = 0
                if -1 in orderbook.index and row["委托代码"] == 1:
                    position += len(orderbook.loc[-1])
                new_row = pd.DataFrame(
                    [[row["委托数量"]]], index=pd.MultiIndex.from_tuples([(row["委托代码"], row["委托价格"])]), columns=["qty"]
                )
                orderbook = pd.concat([orderbook.iloc[:position], new_row, orderbook.iloc[position:]])
                depth_events_c.append(
                    [1, timestamp, -1, row["委托代码"], row["委托价格"], orderbook.loc[(row["委托代码"], row["委托价格"]), "qty"]]
                )
        elif row["委托类型"] == "D":
            if (row["委托代码"], row["委托价格"]) in orderbook.index:
                # if orderbook.loc[(row['委托代码'], row['委托价格']), 'qty'] >= row['委托数量']:
                orderbook.loc[(row["委托代码"], row["委托价格"]), "qty"] -= row["委托数量"]
                if orderbook.loc[(row["委托代码"], row["委托价格"]), "qty"] < 0:
                    raise LobException(
                        f"process order {row['交易所委托号']} gives negative qty of {row['代码']} at {dt.isoformat()}"
                    )
                    # orderbook.loc[(row['委托代码'], row['委托价格']), 'qty'] = 0
                depth_events_c.append(
                    [1, timestamp, -1, row["委托代码"], row["委托价格"], orderbook.loc[(row["委托代码"], row["委托价格"]), "qty"]]
                )
                if orderbook.loc[(row["委托代码"], row["委托价格"]), "qty"] == 0:
                    orderbook.drop((row["委托代码"], row["委托价格"]), axis=0, inplace=True)
            else:
                raise LobException(f"no order to cancel for {row['交易所委托号']} of {row['代码']} at {dt.isoformat()}")
    return pd.DataFrame(trade_events_c, columns=EVENT_DF_COLS), pd.DataFrame(depth_events_c, columns=EVENT_DF_COLS)


def convert(input_quote_file_name, input_trade_file_name, depth_file_name, trade_file_name, force_update=False):
    input_quote_file_name = Path(input_quote_file_name)
    input_trade_file_name = Path(input_trade_file_name)
    depth_file_name = Path(depth_file_name)
    trade_file_name = Path(trade_file_name)
    if not force_update and depth_file_name.exists() and trade_file_name.exists():
        logger.info(f"Files {depth_file_name.name} and {trade_file_name.name} already exist, skip.")
        return
    try:
        quote_df = pd.read_csv(input_quote_file_name, encoding="gbk", parse_dates=[["自然日", "时间"]])
    except:
        quote_df = pd.read_csv(input_quote_file_name, encoding="utf-8", parse_dates=[["自然日", "时间"]])
    try:
        trade_df = pd.read_csv(input_trade_file_name, encoding="gbk", parse_dates=[["自然日", "时间"]], dtype={"成交代码": str})
    except:
        trade_df = pd.read_csv(input_trade_file_name, encoding="utf-8", parse_dates=[["自然日", "时间"]], dtype={"成交代码": str})
    quote_df["委托代码"] = quote_df["委托代码"].replace({"B": 1, "S": -1})
    if quote_df["代码"].iloc[0].endswith("SH"):
        collection_auction_df = quote_df[quote_df["自然日_时间"].dt.time < time(9, 26)].copy()
        collection_auction_df["委托数量"] = collection_auction_df["委托数量"] * collection_auction_df["委托类型"].replace(
            {"D": -1, "A": 1}
        )
        trade_df_open = trade_df.loc[trade_df["自然日_时间"].dt.time < time(9, 26), "成交价格"].unique()
        if len(trade_df_open) > 1:
            raise ValueError("Open price not unique")
        if len(trade_df_open) == 0:
            trade_df_open = [np.nan]
        open_orderbook, trade_events_df, depth_events_df = parse_collection_auctions(
            collection_auction_df, trade_df_open[0]
        )
        try:
            con_trade_event_df, con_depth_event_df = parse_continuous_auctions(
                open_orderbook, quote_df[quote_df["自然日_时间"].dt.time >= time(9, 26)].set_index("自然日_时间").sort_index()
            )
        except LobException:
            return
    elif quote_df["代码"].iloc[0].endswith("SZ"):
        trade_df["BS标志"] = trade_df["BS标志"].replace({"B": 1, "S": -1})
        collection_canceled = set(
            trade_df.loc[
                (trade_df["自然日_时间"].dt.time < time(9, 26)) & (trade_df["成交代码"] == "C"), ["叫卖序号", "叫买序号"]
            ].unstack()
        )
        trade_df_open = trade_df.loc[
            (trade_df["自然日_时间"].dt.time < time(9, 26)) & (trade_df["成交代码"] == "0"), "成交价格"
        ].unique()
        if len(trade_df_open) > 1:
            raise ValueError("Open price not unique")
        if len(trade_df_open) == 0:
            trade_df_open = [np.nan]
        open_orderbook, trade_events_df, depth_events_df = parse_collection_auctions(
            quote_df.loc[
                (quote_df["自然日_时间"].dt.time < time(9, 26)) & ~quote_df["交易所委托号"].isin(collection_canceled)
            ].copy(),
            trade_df_open[0],
        )
        continious_bid = quote_df[quote_df["自然日_时间"].dt.time > time(9, 26)].copy()
        continious_bid["委托类型"] = "A"
        cancels = trade_df.loc[(trade_df["自然日_时间"].dt.time > time(9, 26)) & (trade_df["成交代码"] == "C")].copy()
        assert (
            (cancels["叫卖序号"] * cancels["叫买序号"] == 0) & (cancels["叫卖序号"] + cancels["叫买序号"] > 0)
        ).all(), "bid order code and offer order code must be one zero and one non-zero"
        cancels["交易所委托号"] = cancels["叫卖序号"] + cancels["叫买序号"]
        cancels["委托代码"] = (cancels["叫卖序号"] == 0).astype(int) * 2 - 1
        cancels["委托类型"] = "D"
        price_mapping = dict(quote_df[["交易所委托号", "委托价格"]].values)
        cancels["委托价格"] = cancels["交易所委托号"].map(price_mapping)
        continious_bid = (
            pd.concat(
                [
                    continious_bid,
                    cancels.drop(columns=["BS标志", "成交价格", "叫卖序号", "叫买序号", "成交编号", "成交代码"]).rename(
                        columns={"成交数量": "委托数量"}
                    ),
                ]
            )
            .sort_values(by=["自然日_时间", "交易所委托号", "委托类型"])
            .set_index("自然日_时间")
        )
        try:
            con_trade_event_df, con_depth_event_df = parse_continuous_auctions(open_orderbook, continious_bid)
        except LobException:
            return
    else:
        raise ValueError("Unknown stock exchange")
    trade_events_df = pd.concat([trade_events_df, con_trade_event_df])
    depth_events_df = pd.concat([depth_events_df, con_depth_event_df])
    with open(depth_file_name, "wb") as fd:
        pickle.dump(depth_events_df, fd)
    with open(trade_file_name, "wb") as ft:
        pickle.dump(trade_events_df, ft)


if __name__ == "__main__":
    base_folder = Path(r"D:\Documents\TradeResearch\Stock\convert_market_making\data")
    quote_folder = base_folder / "quote"
    trade_folder = base_folder / "trade"
    converted_folder = base_folder / ".." / "converted_data"
    if not converted_folder.exists():
        converted_folder.mkdir(exist_ok=True)
    pool = mp.Pool(mp.cpu_count() - 8)
    quote_files = list(quote_folder.glob("*.csv"))
    pbar = tqdm(total=len(quote_files))
    #quote_files = [quote_folder / "sz127037_20230322.csv"]
    mp_futures = []

    for quote_file in quote_files:
        base_name = quote_file.stem
        trade_file = trade_folder / f"{base_name}.csv"
        if trade_file.exists():
            # print(f"Converting {base_name}...")
            # convert(quote_file, trade_file, converted_folder / f"{base_name}.pkl", converted_folder / f"{base_name}_trades.pkl", force_update=False)
            # pbar.update(1)
            mp_futures.append(
                pool.apply_async(
                    convert,
                    args=(
                        quote_file,
                        trade_file,
                        converted_folder / f"{base_name}.pkl",
                        converted_folder / f"{base_name}_trades.pkl",
                    ),
                    callback=lambda _: pbar.update(1),
                )
            )
        else:
            pbar.update(1)

    pool.close()
    pool.join()
    pbar.close()
