"""Combined 2x3 embedding figure.

Row 1: linear-growth case -- parameter embedding + PC0 axis, gamma vs PC0;
       kl_rate embedding.
Row 2: oscillatory case   -- one H = 64 run (`run_training.sh grok-h64`):
       its epoch-40 embedding snapshot coloured by alpha*omega, then its
       epoch-200 snapshot coloured by alpha and by omega (_grok_common).
"""
import argparse
import sys
from pathlib import Path


import cmocean
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from graph_dyno.plotting.plot import load_run

from _ckpt_store import add_source_args, apply_source_args, ckpt_root
from _grok_common import EARLY, LATE, pick_snapshots
from _run_lookup import (  # noqa: E402  (importing from sibling after sys.path tweak)
    find_run_in_experiment,
    load_experiments_yaml,
    setup_mlflow_uri,
)

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def _resolve_run(*, run_name=None, experiment=None,
                 metric="best_val_loss", direction="min",
                 tag_filters=None, param_filters=None,
                 use_last=False):
    """Pick a run by explicit name or by experiment lookup."""
    if run_name and ckpt_root() is None:
        return load_run(run_name, use_last=use_last)
    if experiment:
        return find_run_in_experiment(
            experiment, metric=metric, direction=direction,
            tag_filters=tag_filters, param_filters=param_filters,
            use_last=use_last, run_name=run_name,
        )
    raise ValueError("Must specify a run name or an experiment name.")


def _pca_axis(embedding):
    """Unit PC0 direction and projection in 2D embedding space."""
    embedding = np.asarray(embedding)
    pca = PCA(n_components=1)
    projection = pca.fit_transform(embedding).flatten()
    axis = pca.components_[0]
    return axis, projection


def _compute_lin_nl_r2(embedding, hue_arr):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    embedding = np.asarray(embedding)
    hue_arr = np.asarray(hue_arr)
    r2_lin = LinearRegression().fit(embedding, hue_arr).score(embedding, hue_arr)
    pipe = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf"))
    param_grid = {
        "kernelridge__alpha": np.logspace(-3, 1, 5),
        "kernelridge__gamma": np.logspace(-2, 2, 5),
    }
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="r2")
    r2_nl = cross_val_score(gs, embedding, hue_arr,
                            cv=outer_cv, scoring="r2").mean()
    return r2_lin, r2_nl


def _truncate_cmap(name, start=0.25, stop=1.0, n=256):
    base = plt.get_cmap(name)
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        f"{name}_trunc", base(np.linspace(start, stop, n))
    )


_R2_LOC_MAP = {
    "top left":     (0.02, 0.98, "top",    "left"),
    "top right":    (0.98, 0.98, "top",    "right"),
    "bottom left":  (0.02, 0.02, "bottom", "left"),
    "bottom right": (0.98, 0.02, "bottom", "right"),
}


def _embedding_scatter(ax, embedding, hue_np, hue_label,
                       pca_axis=None, projection=None, line_color="magenta",
                       log_norm=False, plot_mi=False, log_mi=False, add_colorbar=True,
                       show_r2=False, colorbar_loc="bottom right", cmap="viridis",
                       r2_loc="bottom left", fit_line=False):
    embedding_full = np.asarray(embedding)
    if embedding_full.shape[1] > 2:
        embedding_2d = PCA(n_components=2).fit_transform(embedding_full)
        if pca_axis is not None and projection is not None \
                and np.asarray(pca_axis).shape[0] != 2:
            pca_axis = np.array([1.0, 0.0])
            projection = embedding_2d[:, 0]
    else:
        embedding_2d = embedding_full

    if fit_line and pca_axis is None and projection is None:
        from sklearn.linear_model import LinearRegression
        w = LinearRegression().fit(embedding_2d, np.asarray(hue_np)).coef_
        pca_axis = w / np.linalg.norm(w)
        projection = embedding_2d @ pca_axis
    if log_norm:
        norm = matplotlib.colors.LogNorm(vmin=hue_np.min(), vmax=hue_np.max())
    else:
        norm = plt.Normalize(hue_np.min(), hue_np.max())

    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1],
                    hue=hue_np, palette=cmap, hue_norm=norm,
                    legend=False, ax=ax, s=9)
    if pca_axis is not None and projection is not None:
        centroid = embedding_2d.mean(axis=0)
        c_proj = centroid @ pca_axis
        t_lo, t_hi = projection.min() - c_proj, projection.max() - c_proj
        p_lo = centroid + t_lo * pca_axis
        p_hi = centroid + t_hi * pca_axis
        ax.plot([p_lo[0], p_hi[0]], [p_lo[1], p_hi[1]],
                linewidth=2, color=line_color, zorder=3, alpha=0.4)

    if plot_mi:
        mi_target = np.log(np.asarray(hue_np)) if log_mi else np.asarray(hue_np)
        mi = mutual_info_regression(embedding_full, mi_target,
                                    n_neighbors=5, random_state=0).mean()
        ax.text(0.02, 0.95,
                f"$I(\\lambda;\\,{hue_label[1:-1]}) = {mi:.2f}$ nats",
                transform=ax.transAxes, fontsize=6, va="top")

    if embedding_full.shape[1] > 2:
        ax.set_xlabel(r"$PC_1$", labelpad=2)
        ax.set_ylabel(r"$PC_2$", labelpad=4)
    else:
        ax.set_xlabel(r"$\lambda_0$", labelpad=2)
        ax.set_ylabel(r"$\lambda_1$", rotation=0, labelpad=4)

    if show_r2:
        r2_lin, r2_nl = _compute_lin_nl_r2(embedding_full, hue_np)
        tx, ty, va, ha = _R2_LOC_MAP.get(r2_loc, _R2_LOC_MAP["bottom left"])
        ax.text(tx, ty,
                f"$R^2_\\mathrm{{lin}} = {r2_lin:.2f}$\n$R^2_\\mathrm{{nl}} = {r2_nl:.2f}$",
                transform=ax.transAxes, ha=ha, va=va, fontsize=6)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if colorbar_loc == "top right":
            cax = ax.inset_axes([0.94, 0.69, 0.035, 0.28])
            cb_side = "left"
        elif colorbar_loc == "bottom left":
            cax = ax.inset_axes([0.03, 0.25, 0.035, 0.28])
            cb_side = "right"
        else:
            cax = ax.inset_axes([0.94, 0.03, 0.035, 0.28])
            cb_side = "left"
        cb = ax.figure.colorbar(sm, cax=cax, location=cb_side)
        cb.set_label(hue_label, fontsize=7, labelpad=2)
        cb.ax.tick_params(labelsize=5, pad=1)


def _projection_correlation(ax, projection, hue_np, hue_label,
                            axis_name=r"PC_1",
                            scatter_color="#1c9d88", line_color="black",
                            axis_label_color="magenta", log_scale=False,
                            text_loc="upper left", embedding=None):
    sns.scatterplot(x=projection, y=hue_np, ax=ax, color=scatter_color, s=9)
    ax.set_xlabel(f"${axis_name}$ of $\\lambda$", color=axis_label_color, labelpad=2)
    ax.set_ylabel(hue_label, labelpad=2)

    if log_scale:
        ax.set_yscale("log")
        log_y = np.log(hue_np)
        z = np.polyfit(projection, log_y, 1)
        p = np.poly1d(z)
        x_fit = [projection.min(), projection.max()]
        ax.plot(x_fit, np.exp(p(x_fit)), dashes=[5, 2], linewidth=2, color=line_color)
        y_pred = p(projection)
        r2 = 1 - np.sum((log_y - y_pred) ** 2) / np.sum((log_y - np.mean(log_y)) ** 2)
    else:
        z = np.polyfit(projection, hue_np, 1)
        p = np.poly1d(z)
        x_fit = [projection.min(), projection.max()]
        ax.plot(x_fit, p(x_fit), dashes=[5, 2], linewidth=2, color=line_color)
        y_pred = p(projection)
        r2 = 1 - np.sum((hue_np - y_pred) ** 2) / np.sum((hue_np - np.mean(hue_np)) ** 2)
        ax.set_ylim([hue_np.min() - 0.1, hue_np.max() + 0.1])

    loc_map = {
        "upper left":  (0.02, 0.90), "upper right": (0.98, 0.90),
        "lower left":  (0.02, 0.18), "lower right": (0.98, 0.18),
    }
    tx, ty = loc_map.get(text_loc, (0.02, 0.90))
    ha = "right" if "right" in text_loc else "left"
    if embedding is not None:
        r2_lin, r2_nl = _compute_lin_nl_r2(embedding, hue_np)
        ty_top = 0.98 if "upper" in text_loc else 0.02
        va = "top" if "upper" in text_loc else "bottom"
        ax.text(tx, ty_top,
                f"$R^2_\\mathrm{{lin}} = {r2_lin:.2f}$\n$R^2_\\mathrm{{nl}} = {r2_nl:.2f}$",
                transform=ax.transAxes, ha=ha, va=va, fontsize=6)
    else:
        ax.text(tx, ty, f"$R^2 = {r2:.2f}$",
                transform=ax.transAxes, ha=ha, fontsize=6)


def _kl_rate_scatter(ax, model):
    embedding = model.parameter_embedding.detach().cpu().numpy()
    embedding_was_reduced = embedding.shape[1] > 2
    if embedding_was_reduced:
        print("REDUCED KL EMBEDDING TO 2D FOR SCATTER")
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(embedding)
    elif embedding.shape[1] < 2:
        raise ValueError(
            f"Embedding has only {embedding.shape[1]} dim; need >= 2 for a 2D scatter."
        )

    kls = model.get_metadata("kls")
    if kls is None:
        raise ValueError("Model has no 'kls' metadata.")
    kls = kls.detach().cpu().numpy()[: embedding.shape[0]]

    hue_label = r"${k}_\ell$"
    cb_palette = sns.color_palette("colorblind")
    two_color_cmap = matplotlib.colors.ListedColormap(
        [cb_palette[0], cb_palette[1]], name="colorblind_pair"
    )
    lo, hi = kls.min(), kls.max()
    mid = 0.5 * (lo + hi)
    discrete_norm = matplotlib.colors.BoundaryNorm([lo, mid, hi], two_color_cmap.N)

    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=kls, palette=two_color_cmap, hue_norm=discrete_norm,
        legend=False, ax=ax, s=9,
    )
    if embedding_was_reduced:
        ax.set_xlabel(r"$PC_1$", labelpad=2)
        ax.set_ylabel(r"$PC_2$", labelpad=4)
    else:
        ax.set_xlabel(r"$\lambda_0$", labelpad=2)
        ax.set_ylabel(r"$\lambda_1$", rotation=0, labelpad=4)

    sm = plt.cm.ScalarMappable(cmap=two_color_cmap, norm=discrete_norm)
    sm.set_array([])
    cax = ax.inset_axes([0.94, 0.72, 0.03, 0.25])
    cb = ax.figure.colorbar(sm, cax=cax, location="left", ticks=[lo, hi])
    cb.set_ticklabels(["$10^3$", "$10^6$"])
    cb.set_label(hue_label, fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=5, pad=1)


def make_figure(growth_run_name=None,
                grok_run_name=None,
                kl_run_name=None,
                growth_experiment=None,
                grok_experiment=None,
                kl_experiment=None,
                kl_filters=None,
                output_path="./figures/combined_embedding_figure.png",
                dpi=300,
                early=EARLY,
                late=LATE):
    sns.set_context("paper")
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    run_growth = _resolve_run(run_name=growth_run_name, experiment=growth_experiment)
    model_growth = run_growth["model"]
    run_kl = _resolve_run(run_name=kl_run_name, experiment=kl_experiment,
                          tag_filters=kl_filters, use_last=True)
    model_kl = run_kl["model"]
    snaps = pick_snapshots(grok_experiment, grok_run_name)
    print(f"oscillatory row: {snaps.run.info.run_name}, epochs {early} / {late}")

    # ---- Growth case ----
    growth_embedding = model_growth.parameter_embedding.detach().cpu().numpy()
    growth_rate = model_growth.get_metadata("growth_rates").detach().cpu().numpy()
    growth_pc_axis, growth_pc_proj = _pca_axis(growth_embedding)

    # ---- Oscillatory case: the early (epoch-40) snapshot ----
    osc_embedding = snaps[early]
    alpha_omega = snaps.targets["alpha_omega"]
    osc_pc_axis, osc_pc_proj = _pca_axis(osc_embedding)

    # ---- OSC grok case: the late (epoch-200) snapshot of the same run ----
    # the same as oscillatory case but i need a plot for alpha and a plot for omega separately
    osc_grok_embedding = snaps[late]
    amplitude_grok = snaps.targets["alpha"]
    omega_grok = snaps.targets["omega"]
    from sklearn.linear_model import LinearRegression

    def _best_linear_axis(emb, target):
        w = LinearRegression().fit(emb, target).coef_
        axis = w / np.linalg.norm(w)
        return axis, emb @ axis

    alpha_axis, alpha_proj = _best_linear_axis(osc_grok_embedding, amplitude_grok)
    omega_axis, omega_proj = _best_linear_axis(osc_grok_embedding, omega_grok)

    osc_grok_embedding_2d = osc_grok_embedding
    if osc_grok_embedding_2d.shape[1] > 2:
        osc_grok_embedding_2d = PCA(n_components=2).fit_transform(osc_grok_embedding_2d)
    osc_grok_embedding_2d = osc_grok_embedding_2d - osc_grok_embedding_2d.min(axis=0)


    # ---- Layout (5.5" wide total; square panels) ----
    fig_w = 5.5
    n_rows, n_cols = 2, 3
    left_m = 0.35
    right_m = 0.1
    col_gap = 0.55
    top_m = 0.15
    bottom_m = 0.3
    row_gap = 0.55
    panel_w = (fig_w - left_m - right_m - (n_cols - 1) * col_gap) / n_cols
    panel_h = panel_w
    fig_h = top_m + n_rows * panel_h + (n_rows - 1) * row_gap + bottom_m
    wspace = col_gap / panel_w
    hspace = row_gap / panel_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(
        n_rows, n_cols,
        wspace=wspace, hspace=hspace,
        left=left_m / fig_w, right=1 - right_m / fig_w,
        top=1 - top_m / fig_h, bottom=bottom_m / fig_h,
    )

    # ---- Row 0: growth case ----
    ax_a = fig.add_subplot(gs[0, 0])
    _embedding_scatter(ax_a, growth_embedding, growth_rate, r"$\gamma$",
                       pca_axis=growth_pc_axis, projection=growth_pc_proj)
    # One cell sits at λ0 ≈ 0.38, far right of the rest. Crop the x-range to
    # the bulk and mark off-scale cells with an arrowhead on the right edge.
    a_x_hi = 0.15
    ax_a.set_xlim(-0.05, a_x_hi)
    off_scale = growth_embedding[:, 0] > a_x_hi
    ax_a.scatter(np.full(off_scale.sum(), a_x_hi), growth_embedding[off_scale, 1],
                 c=growth_rate[off_scale], cmap="viridis",
                 norm=plt.Normalize(growth_rate.min(), growth_rate.max()),
                 marker=">", s=16, clip_on=False, zorder=4)

    ax_b = fig.add_subplot(gs[0, 1])
    _projection_correlation(ax_b, growth_pc_proj, growth_rate, r"$\gamma$",
                            log_scale=False, text_loc="upper right",
                            embedding=growth_embedding)

    ax_c = fig.add_subplot(gs[0, 2])
    _kl_rate_scatter(ax_c, model_kl)

    # ---- Row 1: oscillatory case ----
    ax_d = fig.add_subplot(gs[1, 0])
    _embedding_scatter(ax_d, osc_embedding, alpha_omega, r"$\alpha\omega$",
                       # pca_axis=osc_pc_axis, projection=osc_pc_proj,
                       log_norm=False, log_mi=True, show_r2=True,
                       r2_loc="bottom left")
    ax_d.set_ylim(-0.2, 0.2)
    ax_d.set_yticks([-0.2, 0, 0.2])
    ax_d.set_ylabel(r"$PC_2$", labelpad=-2)

    ax_e = fig.add_subplot(gs[1, 1])
    _embedding_scatter(ax_e, osc_grok_embedding_2d, amplitude_grok, r"$\alpha$",
                       log_norm=False, show_r2=True, colorbar_loc="bottom right",
                       cmap=_truncate_cmap("Blues", start=0.25))
    ax_e.set_xlabel(r"$PC_1$", labelpad=2)
    ax_e.set_ylabel(r"$PC_2$", labelpad=1, rotation="vertical")
    ax_e.locator_params(axis="both", nbins=3)

    # now scatter for omega separately
    ax_f = fig.add_subplot(gs[1, 2])
    _embedding_scatter(ax_f, osc_grok_embedding_2d, omega_grok, r"$\omega$",
                       log_norm=False, show_r2=True, colorbar_loc="bottom right",
                       cmap=_truncate_cmap("Greens", start=0.25))
    ax_f.set_xlabel(r"$PC_1$", labelpad=2)
    ax_f.set_ylabel(r"$PC_2$", labelpad=1, rotation="vertical")
    ax_f.locator_params(axis="both", nbins=3)

    # ---- Panel labels (figure1b.py style) ----
    top_row = [ax_a, ax_b, ax_c]
    bot_row = [ax_d, ax_e, ax_f]
    for col in range(n_cols):
        top_bbox = top_row[col].get_position()
        bot_bbox = bot_row[col].get_position()
        x = min(top_bbox.x0, bot_bbox.x0) - 0.015
        fig.text(x, top_bbox.y1 + 0.005, "abc"[col],
                 fontweight="bold", va="bottom", ha="left")
        fig.text(x, bot_bbox.y1 + 0.005, "def"[col],
                 fontweight="bold", va="bottom", ha="left")

    fig.patch.set_alpha(0.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", format="svg", transparent=True)
    pdf_path = output_path.with_name("figure3.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf", transparent=True)
    print(f"Saved {output_path}")
    print(f"Saved {svg_path}")
    print(f"Saved {pdf_path}")
    return fig


def _parse_args():
    cfg = load_experiments_yaml().get("figure3", {})
    p = argparse.ArgumentParser(description="Combined 2x3 embedding figure")
    # Experiment-name selectors (preferred). Defaults come from
    # paper_figures/experiments.yaml.
    p.add_argument("--growth-experiment", default=cfg.get("growth"))
    p.add_argument("--grok-experiment", default=None,
                   help="H = 64 experiment for row 2 (default: grok_h64 in experiments.yaml)")
    p.add_argument("--kl-experiment", default=cfg.get("kl"))
    # Explicit run-name overrides for picking a specific seed/run.
    p.add_argument("--growth-run", default=None)
    p.add_argument("--grok-run", default=None,
                   help="H = 64 run for row 2 (default: grok_h64.figure_seed)")
    p.add_argument("--kl-run", default=None)
    p.add_argument("--early", type=int, default=EARLY, help="snapshot epoch for panel d")
    p.add_argument("--late", type=int, default=LATE, help="snapshot epoch for panels e, f")
    p.add_argument("--output", default="./figures/combined_embedding_figure.png")
    p.add_argument("--dpi", type=int, default=300)
    add_source_args(p)
    args = apply_source_args(p.parse_args())
    args.kl_filters = cfg.get("kl_filters")
    if ckpt_root() is not None and args.kl_run is None and cfg.get("kl_ckpt_seed") is not None:
        args.kl_run = f"seed_{cfg['kl_ckpt_seed']}"
    return args


if __name__ == "__main__":
    setup_mlflow_uri()
    args = _parse_args()
    make_figure(
        growth_run_name=args.growth_run,
        grok_run_name=args.grok_run,
        kl_run_name=args.kl_run,
        growth_experiment=args.growth_experiment,
        grok_experiment=args.grok_experiment,
        kl_experiment=args.kl_experiment,
        kl_filters=args.kl_filters,
        output_path=args.output,
        dpi=args.dpi,
        early=args.early,
        late=args.late,
    )
