"""JKP-universe RavenPack DJ news motivation plot."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from constants import data_path, plot_path


_SIZE_GRP_ORDER = ["mega", "large", "small", "micro", "nano"]


def load_news():
    df = pd.read_parquet(Path(data_path) / "rpna_jkp_news.parquet").drop(columns=["permno"])
    df["rpa_date_utc"] = pd.to_datetime(df["rpa_date_utc"])

    em = pd.read_csv(Path(data_path) / "rpna_jkp_entity_map.csv")
    em["valid_from"] = pd.to_datetime(em["valid_from"])
    em["valid_to"]   = pd.to_datetime(em["valid_to"])
    em["permno"]     = em["permno"].astype(int)

    out = df.merge(em, on="rp_entity_id", how="inner")
    # Time-aware filter (defense in depth; SQL-side join in get_rpna_jkp_news already enforces this).
    out = out[(out["rpa_date_utc"] >= out["valid_from"])
              & (out["rpa_date_utc"] <= out["valid_to"])]
    # Overlapping entity-map intervals duplicate rows; keep one per news item per stock.
    out = out.drop_duplicates(["rp_story_id", "rp_story_event_index", "permno"])
    out = out.drop(columns=["valid_from", "valid_to"]).reset_index(drop=True)
    out["yyyymm"] = out["rpa_date_utc"].dt.year * 100 + out["rpa_date_utc"].dt.month
    return out


def load_lagged_betas():
    df = pd.read_csv(
        Path(data_path) / "monthly_lagged_betas.csv",
        usecols=["permno", "yyyymm", "beta_252d", "size_grp"],
    )
    df["permno"] = df["permno"].astype(int)
    df["yyyymm"] = df["yyyymm"].astype(int)
    return df.drop_duplicates(["yyyymm", "permno"], keep="last")


def load_data():
    return load_news(), load_lagged_betas()


def build_panel(events, betas, min_relevance=75, q_beta=10):
    """Per (permno, yyyymm): news metrics + within-month β decile + size_grp.
    Universe = betas with non-null beta_252d & size_grp; zero-filled for stock-months
    with no events at relevance >= min_relevance."""
    sub = events[events["relevance"] >= min_relevance].copy()
    sub["nip_pos"] = sub["nip"].clip(lower=0).fillna(0)
    sub["abs_css"] = sub["css"].abs().fillna(0)
    sub["abs_ess"] = sub["event_sentiment_score"].abs().fillna(0)
    monthly = sub.groupby(["permno", "yyyymm"]).agg(
        n_events    = ("relevance", "size"),
        sum_nip_pos = ("nip_pos",   "sum"),
        sum_abs_css = ("abs_css",   "sum"),
        sum_abs_ess = ("abs_ess",   "sum"),
    ).reset_index()

    df = (betas.dropna(subset=["beta_252d", "size_grp"])
                .merge(monthly, on=["permno", "yyyymm"], how="left"))
    df["n_events"]    = df["n_events"].fillna(0).astype(int)
    df["sum_nip_pos"] = df["sum_nip_pos"].fillna(0.0)
    df["sum_abs_css"] = df["sum_abs_css"].fillna(0.0)
    df["sum_abs_ess"] = df["sum_abs_ess"].fillna(0.0)

    df["beta_decile"] = df.groupby("yyyymm")["beta_252d"].transform(
        lambda x: pd.qcut(x, q_beta, labels=False, duplicates="drop") + 1
    )
    df = df.dropna(subset=["beta_decile"]).copy()
    df["beta_decile"] = df["beta_decile"].astype(int)
    return df


_METRICS = [
    ("n_events",    "News Count"),
    ("sum_nip_pos", "News Impact Projection"),
    ("sum_abs_css", "Composite Sentiment Score"),
    ("sum_abs_ess", "Event Sentiment Score"),
]


def plot_news_intensity_by_beta(events, betas, min_relevance=75):
    df = build_panel(events, betas, min_relevance=min_relevance)
    colors = dict(zip(_SIZE_GRP_ORDER, ["C0", "C1", "C2", "C3", "C4"]))
    bottom_groups = [g for g in _SIZE_GRP_ORDER if g != "mega"]
    xt = ["Low", "2", "3", "4", "5", "6", "7", "8", "9", "High"]

    # 2x2 layout: each measure is a thin mega row over a tall others row.
    # GAP_SMALL = within-measure gap; SPACER = empty row between measure blocks.
    GAP_SMALL, SPACER = 0.05, 0.8
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 3, SPACER, 1, 3],
                          hspace=GAP_SMALL, wspace=0.18)
    row_map = {0: (0, 1), 1: (3, 4)}   # measure-block -> (mega row, others row)
    for i, (col, title) in enumerate(_METRICS):
        blk, c = i // 2, i % 2
        mrow, brow = row_map[blk]
        ax_top, ax_bot = fig.add_subplot(gs[mrow, c]), fig.add_subplot(gs[brow, c])
        pv = (df.groupby(["yyyymm", "size_grp", "beta_decile"], observed=True)[col].mean()
                .groupby(["size_grp", "beta_decile"], observed=True).mean()
                .reset_index()
                .pivot(index="size_grp", columns="beta_decile", values=col)
                .reindex(_SIZE_GRP_ORDER))

        ax_top.plot(pv.columns, pv.loc["mega"].values, color=colors["mega"],
                    marker="o", lw=1.4, ms=5, label="mega")
        ax_top.set_xticks(range(1, 11)); ax_top.grid(False)
        ax_top.tick_params(labelbottom=False, labelsize=10)
        ax_top.set_title(title, fontsize=15)

        for gv in bottom_groups:
            ax_bot.plot(pv.columns, pv.loc[gv].values, color=colors[gv],
                        marker="o", lw=1.4, ms=5, label=gv)
        ax_bot.set_xticks(range(1, 11))
        ax_bot.set_xticklabels(xt, fontsize=11)
        ax_bot.grid(False)
        ax_bot.tick_params(axis="y", labelsize=10)

        if i == 0:
            ax_top.legend(loc="best", fontsize=10)
            ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax_bot.legend(title="size_grp", loc="best", fontsize=10, title_fontsize=11)

    fig.savefig(Path(plot_path) / "news_intensity_by_beta.pdf", bbox_inches="tight")
    plt.show()
