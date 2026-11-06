"""Statistics and clustering scores shared by the multi-seed paper tables.

    summarize            mean / sd / median / quartiles / range / t-CI of one arm's scores
    quartiles            (Q1, median, Q3) of one arm's scores
    fmt_iqr, tex_iqr     "median [Q1, Q3]" for the console and $\\iqr{median}{Q1}{Q3}$ for LaTeX
    iqr_errorbar         seaborn `errorbar=` callable: (Q1, Q3) whiskers for median bars
    kmeans_ari           k-means ARI at known k
    nc                   neighbor consistency of an embedding w.r.t. a target
    benjamini_hochberg   step-up adjusted p-values
    holm                 Holm-Bonferroni adjusted p-values

k-means uses fixed `random_state=0` and `n_init=10`, so scoring is
deterministic and all observed spread across runs is training-seed spread.
"""
import numpy as np
from scipy import stats
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def summarize(vals):
    """n, mean, sd (ddof=1), median, Q1/Q3, min/max and the 95% t-interval of the mean."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    n = v.size
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan,
                    lo=np.nan, hi=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    q1, md, q3 = quartiles(v)
    return dict(n=n, mean=float(v.mean()), sd=sd, median=md, q1=q1, q3=q3,
                lo=float(v.min()), hi=float(v.max()),
                ci_lo=float(v.mean() - half), ci_hi=float(v.mean() + half))


def quartiles(vals):
    """(Q1, median, Q3) of the finite values, numpy's default (linear-interpolation) percentiles."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    return float(q1), float(md), float(q3)


def _num(x, prec):
    """Fixed-point, with a rounded-to-zero negative printed as 0 (not -0.00)."""
    s = f"{x:.{prec}f}"
    return s[1:] if s == f"-{0:.{prec}f}" else s


def fmt_iqr(vals, prec=3):
    """`median [Q1, Q3]` of the finite values, or `--` when there are none."""
    q1, md, q3 = quartiles(vals)
    if not np.isfinite(md):
        return "--"
    return f"{_num(md, prec)} [{_num(q1, prec)}, {_num(q3, prec)}]"


def tex_iqr(vals, prec=2, bold=False):
    r"""LaTeX cell `$\iqr{median}{Q1}{Q3}$` (the paper's macro); `--` when there are no values.

    bold marks a significant entry ({\boldmath ...}).
    """
    q1, md, q3 = quartiles(vals)
    if not np.isfinite(md):
        return "--"
    cell = f"$\\iqr{{{_num(md, prec)}}}{{{_num(q1, prec)}}}{{{_num(q3, prec)}}}$"
    return f"{{\\boldmath{cell}}}" if bold else cell


# Preamble for a generated LaTeX table: defines the gray IQR and \iqr unless the paper already has.
TEX_IQR_DEFS = (r"\providecolor{iqrgray}{gray}{0.55}" "\n"
                r"\providecommand{\iqr}[3]{\ensuremath{#1\;{\color{iqrgray}[#2, #3]}}}")


def iqr_errorbar(x):
    """seaborn `errorbar=` callable: the (Q1, Q3) whisker of a median bar."""
    q1, _, q3 = quartiles(x)
    return q1, q3


def kmeans_ari(emb, labels, k, standardize=False):
    x = StandardScaler().fit_transform(emb) if standardize else emb
    pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
    return adjusted_rand_score(labels, pred)


def nc(emb, target, K=10, NB=20):
    """Neighbor consistency: fraction of the K nearest neighbors in the same bin."""
    target = np.asarray(target, dtype=float)
    bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, NB + 1)
    b = np.digitize(target, bins)
    _, idx = KDTree(emb).query(emb, k=K + 1)
    return float((b[idx[:, 1:]] == b[:, None]).mean())


def benjamini_hochberg(p):
    """Step-up adjusted p-values, monotone by construction."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = running
    return q


def holm(p):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(p, dtype=float)
    idx = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj
