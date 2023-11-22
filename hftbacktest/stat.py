from numba.experimental import jitclass
from numba.typed import List
from numba.core.types import ListType
from numba import float64, int64
from matplotlib import pyplot as plt
from mplfinance._utils import IntegerIndexDateTimeFormatter
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


@jitclass
class Recorder:
    timestamp: ListType(int64)
    mid: ListType(float64)
    balance: ListType(float64)
    position: ListType(float64)
    fee: ListType(float64)
    trade_num: ListType(int64)
    trade_qty: ListType(float64)
    trade_amount: ListType(float64)

    def __init__(self, timestamp, mid, balance, position, fee, trade_num, trade_qty, trade_amount):
        self.timestamp = timestamp
        self.mid = mid
        self.balance = balance
        self.position = position
        self.fee = fee
        self.trade_num = trade_num
        self.trade_qty = trade_qty
        self.trade_amount = trade_amount

    def record(self, hbt):
        self.timestamp.append(hbt.current_timestamp)
        self.mid.append((hbt.best_bid + hbt.best_ask) / 2.0)
        self.balance.append(hbt.balance)
        self.position.append(hbt.position)
        self.fee.append(hbt.fee)
        self.trade_num.append(hbt.trade_num)
        self.trade_qty.append(hbt.trade_qty)
        self.trade_amount.append(hbt.trade_amount)


class Stat:
    def __init__(self, hbt, utc=True, unit='us', allocated=100000):
        self.hbt = hbt
        self.utc = utc
        self.unit = unit
        self.timestamp = List.empty_list(int64, allocated=allocated)
        self.mid = List.empty_list(float64, allocated=allocated)
        self.balance = List.empty_list(float64, allocated=allocated)
        self.position = List.empty_list(float64, allocated=allocated)
        self.fee = List.empty_list(float64, allocated=allocated)
        self.trade_num = List.empty_list(int64, allocated=allocated)
        self.trade_qty = List.empty_list(float64, allocated=allocated)
        self.trade_amount = List.empty_list(float64, allocated=allocated)

    @property
    def recorder(self):
        return Recorder(
            self.timestamp,
            self.mid,
            self.balance,
            self.position,
            self.fee,
            self.trade_num,
            self.trade_qty,
            self.trade_amount
        )

    def datetime(self):
        return pd.to_datetime(np.asarray(self.timestamp), utc=self.utc, unit=self.unit)

    def equity(self, resample=None, include_fee=True):
        if include_fee:
            equity = pd.Series(
                self.hbt.local.state.asset_type.equity(
                    np.asarray(self.mid),
                    np.asarray(self.balance),
                    np.asarray(self.position),
                    np.asarray(self.fee)
                ),
                index=self.datetime()
            )
        else:
            equity = pd.Series(
                self.hbt.local.state.asset_type.equity(
                    np.asarray(self.mid),
                    np.asarray(self.balance),
                    np.asarray(self.position),
                    0
                ),
                index=self.datetime()
            )
        if resample is None:
            return equity
        else:
            return equity.resample(resample).last()

    def sharpe(self, resample, include_fee=True, trading_days=365):
        pnl = self.equity(resample, include_fee=include_fee).diff()
        c = (24 * 60 * 60 * 1e9) / (pnl.index[1] - pnl.index[0]).value
        std = pnl.std()
        return np.divide(pnl.mean(), std) * np.sqrt(c * trading_days)

    def sortino(self, resample, include_fee=True, trading_days=365):
        pnl = self.equity(resample, include_fee=include_fee).diff()
        std = pnl[pnl < 0].std()
        c = (24 * 60 * 60 * 1e9) / (pnl.index[1] - pnl.index[0]).value
        return np.divide(pnl.mean(), std) * np.sqrt(c * trading_days)

    def riskreturnratio(self, include_fee=True):
        return self.annualised_return(include_fee=include_fee) / self.maxdrawdown(include_fee=include_fee)

    def drawdown(self, resample=None, include_fee=True):
        equity = self.equity(resample, include_fee=include_fee)
        max_equity = equity.cummax()
        drawdown = equity - max_equity
        return drawdown

    def maxdrawdown(self, denom=None, include_fee=True):
        mdd = -self.drawdown(None, include_fee=include_fee).min()
        if denom is None:
            return mdd
        else:
            return mdd / denom

    def daily_trade_num(self):
        return pd.Series(self.trade_num, index=self.datetime()).diff().rolling('1d').sum().mean()

    def daily_trade_volume(self):
        return pd.Series(self.trade_qty, index=self.datetime()).diff().rolling('1d').sum().mean()

    def daily_trade_amount(self):
        return pd.Series(self.trade_amount, index=self.datetime()).diff().rolling('1d').sum().mean()

    def annualised_return(self, denom=None, include_fee=True, trading_days=365):
        equity = self.equity(None, include_fee=include_fee)
        c = (24 * 60 * 60 * 1e9) / (equity.index[-1] - equity.index[0]).value
        if denom is None:
            return equity[-1] * c * trading_days
        else:
            return equity[-1] * c * trading_days / denom

    def summary(self, capital=None, resample='5min', trading_days=365, trading_hours_per_day=24, return_result=False):
        dt_index = self.datetime()
        raw_equity = self.hbt.local.state.asset_type.equity(
            np.asarray(self.mid),
            np.asarray(self.balance),
            np.asarray(self.position),
            np.asarray(self.fee)
        )
        raw_equity_wo_fee = self.hbt.local.state.asset_type.equity(
            np.asarray(self.mid),
            np.asarray(self.balance),
            np.asarray(self.position),
            0
        )
        equity = pd.Series(raw_equity, index=dt_index)
        rs_equity_wo_fee = pd.Series(raw_equity_wo_fee, index=dt_index).resample(resample, label='right').last().dropna()
        rs_equity = equity.resample(resample, label='right').last().dropna()
        rs_pnl = rs_equity.diff()

        c = (24 * 60 * 60 * 1e9) / (rs_pnl.index[1] - rs_pnl.index[0]).value
        sr = np.divide(rs_pnl.mean(), rs_pnl.std()) * np.sqrt(c * trading_days)

        std = rs_pnl[rs_pnl < 0].std()
        sortino = np.divide(rs_pnl.mean(), std) * np.sqrt(c * trading_days)

        max_equity = rs_equity.cummax()
        drawdown = rs_equity - max_equity
        mdd = -drawdown.min()

        if trading_hours_per_day is not None:
            ac = (trading_hours_per_day * 60 * 60 * 1e9) / (equity.index[-1] - equity.index[0]).value
        else:
            ac = 1/len(set(equity.index.date))
        ar = raw_equity[-1] * ac * trading_days
        rrr = ar / mdd

        dtn = pd.Series(self.trade_num, index=dt_index).diff().rolling('1d').sum().mean()
        dtq = pd.Series(self.trade_qty, index=dt_index).diff().rolling('1d').sum().mean()
        dta = pd.Series(self.trade_amount, index=dt_index).diff().rolling('1d').sum().mean()
        mkt_value = np.asarray(self.position) * np.asarray(self.mid)
        ret_result = None
        if not return_result:
            print('=========== Summary ===========')
            print('Sharpe ratio: %.1f' % sr)
            print('Sortino ratio: %.1f' % sortino)
            print('Risk return ratio: %.1f' % rrr)
            if capital is not None:
                print('Period return: %.2f %%' % (raw_equity[-1] / capital * 100))
                print('Annualised return: %.2f %%' % (ar / capital * 100))
                print('Max. draw down: %.2f %%' % (mdd / capital * 100))
            else:
                print('Period return: %.2f %%' % raw_equity[-1])
                print('Annualised return: %.2f' % ar)
                print('Max. draw down: %.2f' % mdd)
            print('The number of trades per day: %d' % dtn)
            print('Avg. daily trading volume: %d' % dtq)
            print('Avg. daily trading amount: %d' % dta)


            if capital is not None:
                print('Max leverage: %.2f' % (np.max(np.abs(mkt_value)) / capital))
                print('Median leverage: %.2f' % (np.median(np.abs(mkt_value)) / capital))
        else:
            ret_result = {
                'Sharpe': sr,
                'Sortino': sortino,
                'Risk/return': rrr,
                "Period return": raw_equity[-1] if capital is None else (raw_equity[-1] / capital),
                'Annualised return': ar if capital is None else (ar / capital),
                'Max D.D': mdd if capital is None else (mdd / capital),
                'Avg. Trades per Day': dtn,
                'Avg. Daily Vol.': dtq,
                'Avg. Daily Amount': dta,
            }
            if capital is not None:
                ret_result.update({
                    "Max leverage": np.max(np.abs(mkt_value)) / capital,
                    "Median leverage": np.median(np.abs(mkt_value)) / capital
                })

        fig, axs = plt.subplots(2, 1, sharex='all')
        fig.subplots_adjust(hspace=0)
        fig.set_size_inches(10, 6)

        def plot_dt_series(ax, dt_series):
            ax.plot(dt_series.values)
            formatter = IntegerIndexDateTimeFormatter([mdates.date2num(d) for d in dt_series.index], "%m-%d\n%H:%M")
            ax.xaxis.set_major_formatter(formatter)

        mid = pd.Series(self.mid, index=dt_index)

        if capital is not None:
            # ((mid / mid[0] - 1).resample(resample, label='right').last() * 100).plot(ax=axs[0], style='grey', alpha=0.5)
            # (rs_equity / capital * 100).plot(ax=axs[0])
            # (rs_equity_wo_fee / capital * 100).plot(ax=axs[0])
            plot_dt_series(axs[0], ((mid / mid[0] - 1).resample(resample, label='right').last().dropna() * 100))
            plot_dt_series(axs[0], (rs_equity / capital * 100).dropna())
            plot_dt_series(axs[0], (rs_equity_wo_fee / capital * 100).dropna())
        else:
            mid.resample(resample, label='right').last().plot(ax=axs[0], style='grey', alpha=0.5)
            (rs_equity * 100).plot(ax=axs[0])
            (rs_equity_wo_fee * 100).plot(ax=axs[0])

        # axs[0].set_title('Equity')
        axs[0].set_ylabel('Cumulative Returns (%)')
        axs[0].grid()
        axs[0].legend(['Trading asset', 'Strategy incl. fee', 'Strategy excl. fee'])

        # todo: this can mislead a user due to aggregation.
        position = pd.Series(self.position, index=dt_index).resample(resample, label='right').last().dropna()
        # position.plot(ax=axs[1])
        plot_dt_series(axs[1], position)
        # ax3 = ax2.twinx()
        # (position * mid).plot(ax=ax3, style='grey', alpha=0.2)
        # axs[1].set_title('Position')
        axs[1].set_ylabel('Position (Qty)')
        # ax3.set_ylabel('Value')
        axs[1].grid()

        if return_result:
            return ret_result, fig
