import numpy as np
from numba import int64, float64
from numba.experimental import jitclass

from .proc import Proc, proc_spec
from ..order import BUY, SELL, NEW, CANCELED, FILLED, EXPIRED, NONE, Order, LIMIT, DAY, PARTIALLY_FILLED
from ..reader import COL_EVENT, COL_LOCAL_TIMESTAMP, COL_SIDE, COL_PRICE, COL_QTY, DEPTH_CLEAR_EVENT, DEPTH_EVENT, \
    DEPTH_SNAPSHOT_EVENT, TRADE_EVENT, USER_DEFINED_EVENT


class Local_(Proc):
    def __init__(
            self,
            reader,
            orders_to_exch,
            orders_from_exch,
            depth,
            state,
            order_latency,
            trade_list_size
    ):
        self._proc_init(
            reader,
            orders_to_exch,
            orders_from_exch,
            depth,
            state,
            order_latency
        )
        self.trade_len = 0
        self.last_trades = np.full((trade_list_size, self.data.shape[1]), np.nan, np.float64)
        self.user_data = np.full((20, self.data.shape[1]), np.nan, np.float64)

    def reset(
            self,
            start_position,
            start_balance,
            start_fee,
            maker_fee,
            taker_fee,
            tick_size,
            lot_size,
            snapshot,
            trade_list_size,
    ):
        self._proc_reset(
            start_position,
            start_balance,
            start_fee,
            maker_fee,
            taker_fee,
            tick_size,
            lot_size,
            snapshot
        )
        self.trade_len = 0
        if trade_list_size is not None:
            self.last_trades = np.full((trade_list_size, self.data.shape[1]), np.nan, np.float64)
        else:
            self.last_trades[:, :] = np.nan
        self.user_data[:, :] = np.nan

    def _next_data_timestamp(self):
        return self._next_data_timestamp_column(COL_LOCAL_TIMESTAMP)

    def _process_recv_order(self, order, recv_timestamp, wait_resp, next_timestamp):
        # Apply the received order response to the local orders.
        self.orders[order.order_id] = order
        if order.status == FILLED:
            self.state.apply_fill(order)

        # Bypass next_timestamp
        return next_timestamp

    def _process_data(self, row):
        # Process a depth event
        if row[COL_EVENT] == DEPTH_CLEAR_EVENT:
            self.depth.clear_depth(row[COL_SIDE], row[COL_PRICE])
        elif row[COL_EVENT] == DEPTH_EVENT or row[COL_EVENT] == DEPTH_SNAPSHOT_EVENT:
            if row[COL_SIDE] == BUY:
                self.depth.update_bid_depth(
                    row[COL_PRICE],
                    row[COL_QTY],
                    row[COL_LOCAL_TIMESTAMP]
                )
            else:
                self.depth.update_ask_depth(
                    row[COL_PRICE],
                    row[COL_QTY],
                    row[COL_LOCAL_TIMESTAMP]
                )

        # Process a trade event
        elif row[COL_EVENT] == TRADE_EVENT:
            if self.last_trades.shape[0] > 0:
                if self.trade_len < self.last_trades.shape[0] - 1:
                    self.last_trades[self.trade_len] = row[:]
                    self.trade_len += 1
                else:
                    raise IndexError('Insufficient trade list size.')

        # Process a user defined event
        elif row[COL_EVENT] >= USER_DEFINED_EVENT:
            i = int(row[COL_EVENT]) - USER_DEFINED_EVENT
            if i >= len(self.user_data):
                raise IndexError('USER_DEFINED_EVENT is out of range.')
            self.user_data[i] = row[:]
        return 0

    def submit_order(self, order_id, side, price, qty, order_type, time_in_force, current_timestamp):
        if order_id in self.orders:
            raise KeyError('Duplicate order_id')

        price_tick = round(price / self.depth.tick_size)
        order = Order(order_id, price_tick, self.depth.tick_size, qty, side, time_in_force, order_type)
        order.req = NEW
        exch_recv_timestamp = current_timestamp + self.order_latency.entry(current_timestamp, order, self)

        self.orders[order.order_id] = order
        self.orders_to.append(order.copy(), exch_recv_timestamp)

    def cancel(self, order_id, current_timestamp):
        order = self.orders.get(order_id)

        if order is None:
            raise KeyError('the given order_id does not exist.')
        if order.req != NONE:
            raise ValueError('the given order cannot be cancelled because there is a ongoing request.')

        order.req = CANCELED
        exch_recv_timestamp = current_timestamp + self.order_latency.entry(current_timestamp, order, self)

        self.orders_to.append(order.copy(), exch_recv_timestamp)

    def clear_inactive_orders(self):
        for order in list(self.orders.values()):
            if order.status == EXPIRED \
                    or order.status == FILLED \
                    or order.status == CANCELED:
                del self.orders[order.order_id]

    def clear_last_trades(self):
        self.trade_len = 0

    def get_user_data(self, event):
        return self.user_data[event - USER_DEFINED_EVENT]

    def at_close(self):
        # print("in local at_close")
        for order in list(self.orders.values()):
            if order.time_in_force == DAY and order.status in {NEW, PARTIALLY_FILLED}:
                order.status = EXPIRED
        self.depth.clear_depth(0, 0)
        # self.clear_inactive_orders()


def Local(
        reader,
        orders_to_exch,
        orders_from_exch,
        depth,
        state,
        order_latency,
        trade_list_size
):
    jitted = jitclass(
        spec=proc_spec(reader, state, order_latency) + [
            ('trade_len', int64),
            ('last_trades', float64[:, :]),
            ('user_data', float64[:, :]),
        ]
    )(Local_)
    return jitted(
        reader,
        orders_to_exch,
        orders_from_exch,
        depth,
        state,
        order_latency,
        trade_list_size
    )
