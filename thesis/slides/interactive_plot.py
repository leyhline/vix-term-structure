#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import colors as mcolors


# Load data in global scope
xm_settle = pd.read_csv("data/8_m_settle.csv", usecols=range(1,10),
                        parse_dates=[0], header=0, index_col=0, na_values=0)
xm_settle = xm_settle.loc["2006-10-23":]
vix_index = pd.read_csv("data/vix.csv", parse_dates=[0], header=0, index_col=0, na_values="null")
vix_index = vix_index.loc["2006-10-23":]
ts_170908 = pd.Series([13.275, 14.325, 14.875, 15.075, 16.125, 16.325, 16.715, 17.050],
                      index=["Sep 19", "Oct 17", "Nov 14", "Dec 19", "Jan 16", "Feb 13", "Mar 20", "Apr 17"])


# Plot VIX
def plot_vix(ax1):
    vix_index["Adj Close"]["2006":"2016"].plot.area(ax=ax1)
    ax1.set_xlabel("")
    ax1.set_ylabel("Index value")
    ax1.set_title("CBOE's Volatility Index (VIX)")
    ax1.set_ylim(10, 85)


# Plot futures
def plot_months(ax2, azim=-90, elev=10):
    xs = np.arange(len(xm_settle))
    zs = np.arange(0, 8)
    verts = []
    xm_settle_3dplot = xm_settle.copy()
    xm_settle_3dplot.index = xs
    for z in zs:
        ys = xm_settle_3dplot.iloc[:,int(z)].fillna(10)
        ys.iloc[0] = 10
        ys.iloc[-1] = 10
        verts.append(list(zip(ys.index.values, ys.values)))
    poly = PolyCollection(verts, linewidth=2.0, facecolors=[cm.winter(i, 1) for i in  np.linspace(0, 1, 8)])
    ax2.add_collection3d(poly, zs=zs, zdir='y')

    ax2.set_xlim3d(0, len(xm_settle))
    ax2.set_xticks([(xm_settle.index.year == year).argmax() for year in xm_settle.index.year.unique()[1::2]])
    ax2.set_xticklabels(xm_settle.index.year.unique()[1::2])
    ax2.set_ylim3d(0.0, 7.5)
    ax2.set_ylabel("Month")
    ax2.set_yticklabels(["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"])
    ax2.set_zlim3d(10, 50)
    ax2.set_zlabel("Futures price")
    ax2.view_init(azim=azim, elev=elev)
    ax2.set_title("Futures prices for the next eight expirations")


def plot_ts(ax, with_spreads=False):
    ax1 = ts_170908.plot(ax=ax, style="-o", label="Futures prices")
    ax.set_title("Term structure of futures from September 8th, 2017")
    ax.set_xlabel("Date of expiration")
    ax.set_ylabel("Futures price")
    for x, y in enumerate(ts_170908):
        if x < 4:
            plt.text(x, y, y, va="top", ha="left", color="#1f77b4")
        else:
            plt.text(x, y, y, va="bottom", ha="right", color="#1f77b4")
    if with_spreads:
        color = "green"
        spreads = ts_170908.aggregate(lambda x: pd.Series([np.nan] + [2*x[i] - x[i-1] - x[i+1] for i in range(1, len(x)-1)] + [np.nan],
                                      index=ts_170908.index))
        ax2 = spreads.plot(secondary_y=True, style="-o", color=color, label="Spread prices")
        for x, y in enumerate(spreads):
            if 1 <= x < 7:
                plt.text(x + 0.05, y, y, va="center", ha="left", color=color)
        plt.axhline(0, color="lightgreen")
        ax2.set_yticklabels([])
        lines = ax1.get_lines() + ax.right_ax.get_lines()
        ax.legend(lines, ["Futures prices", "Spread prices (long)"], loc="lower right")
    ax.grid(True)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_vix(ax)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_months(ax, None, None)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plot_vix(ax1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_months(ax2)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_ts(ax, False)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_ts(ax, True)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

