"""Plot SQL-returned LightGBM calibration and interpretation results."""

import matplotlib.pyplot as plt
import numpy as np
from vgi_notebooks import finish


def calibration(curve_frame, metrics_frame, N_ESTIMATORS):
    """Present worker-returned predictions and their evaluation population."""
    outside = metrics_frame[metrics_frame.sample_role == "Beyond training range"].iloc[0]
    fig, ax = plt.subplots(figsize=(11.3, 6))
    ax.axvspan(8, 12, color="#F3E1D5", alpha=0.75, label="Beyond training range")
    ax.plot(
        curve_frame.input_v,
        curve_frame.reference_mm,
        color="#6A7882",
        ls="--",
        lw=2,
        label="Authored reference: 2 + 3 × voltage",
    )
    ax.plot(
        curve_frame.input_v,
        curve_frame.predicted_mm,
        color="#327659",
        lw=2.4,
        drawstyle="steps-mid",
        label="Worker prediction on the grid",
    )
    training = curve_frame[curve_frame.sample_role == "Training"]
    ax.scatter(
        training.input_v,
        training.reference_mm,
        color="#243B2F",
        s=12,
        alpha=0.6,
        label="Training points",
        zorder=3,
    )
    ax.axvline(8, color="#B5693B", lw=1.3)
    ax.set(xlabel="Input voltage (V)", ylabel="Response (mm)", xlim=(-0.15, 12.2))
    ax.margins(y=0.13)
    ax.legend(loc="upper left", fontsize=9)
    ax.annotate(
        f"Outside prediction\n{outside.max_prediction_mm:.2f} mm",
        xy=(11, outside.max_prediction_mm),
        xytext=(9.8, outside.max_prediction_mm - 5),
        arrowprops={"arrowstyle": "-", "color": "#327659"},
        color="#327659",
        fontsize=10,
    )
    finish(
        fig,
        "Tree predictions level off beyond the training range",
        f"{N_ESTIMATORS} boosting iterations • seven leaves • shaded voltages were never used for training",
        "Authored noise-free calibration rule • actual VGI LightGBM v0.1.0 fit/predict\n"
        "Sampled grid: 0.05 V; the reference is a designed rule, not a measurement or uncertainty band.",
        name="calibration",
    )


def errors(metrics_frame, N_ESTIMATORS):
    """Present worker-returned predictions and their evaluation population."""
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    colors = ["#9BAAA2", "#327659", "#B5693B"]
    bars = ax.bar(range(len(metrics_frame)), metrics_frame.mae_mm, color=colors, width=0.60)
    ax.bar_label(bars, labels=[f"{v:.3f} mm" for v in metrics_frame.mae_mm], padding=6)
    ax.set_xticks(
        range(len(metrics_frame)),
        [f"{r.sample_role}\n{r.points:.0f} fixed grid points" for r in metrics_frame.itertuples()],
    )
    ax.set_ylabel("Mean absolute error (mm; lower is better)")
    ax.set_ylim(0, float(metrics_frame.mae_mm.max()) * 1.3)
    finish(
        fig,
        "Measure held-out error separately from training fit",
        f"{N_ESTIMATORS} boosting iterations • identical reference points and split across experiments",
        "SQL averages of absolute prediction errors against the authored reference\n"
        "VGI LightGBM v0.1.0; no sampling uncertainty is modeled; "
        "beyond-range errors describe a separate condition.",
        name="errors",
    )


def importances(frame, evaluation_split):
    """Show incomparable importance quantities on explicitly separate axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    y = np.arange(len(frame))
    colors = ["#A7B6AD" if v == "constant_zero" else "#327659" for v in frame.feature]
    bars = axes[0].barh(y, frame.split_gain / 1_000_000, color=colors)
    axes[0].bar_label(bars, labels=[f"{v / 1_000_000:.3f}" for v in frame.split_gain], padding=4, fontsize=9)
    axes[0].set_yticks(y, frame.feature)
    axes[0].invert_yaxis()
    axes[0].set_title("Inside fitted trees", fontsize=12)
    axes[0].set_xlabel("Total split gain (millions)\n(objective-dependent units)")
    axes[0].set_xlim(0, max(1, frame.split_gain.max() / 1_000_000) * 1.30)
    axes[1].barh(y, frame.mae_increase, color=colors)
    axes[1].errorbar(frame.mae_increase, y, xerr=frame.shuffle_std, fmt="none", color="#B16D44", capsize=4)
    for position, row in enumerate(frame.itertuples()):
        offset = 2.0 if row.mae_increase >= 0 else -2.0
        edge = row.mae_increase + (row.shuffle_std if offset > 0 else -row.shuffle_std)
        axes[1].text(
            edge + offset,
            position,
            f"{row.mae_increase:.3f}",
            va="center",
            ha="left" if offset > 0 else "right",
            fontsize=9,
        )
    axes[1].axvline(0, color="#AAB6B0", lw=1)
    axes[1].set_title("After evaluation-feature shuffling", fontsize=12)
    axes[1].set_xlabel("MAE increase (target units)\n(error bars: shuffle SD, not confidence intervals)")
    axes[1].margins(x=0.18)
    finish(
        fig,
        "Split gain and shuffle loss measure different things",
        f"Evaluation: {evaluation_split} · all four features retained · five shuffles per feature, seed 7",
        "Generated sample · actual VGI LightGBM v0.1.0 · fixed training model\n"
        "Separate x scales/units; constant-zero feature is a control, "
        "and model importance is not a causal effect.",
        name="importance-comparison",
    )


def partial_response(frame, evaluation_split):
    """Plot averaged model responses at returned feature grid values."""
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(frame.feature_value, frame.partial_dependence, marker="o", lw=2.2, color="#327659")
    ax.set(
        xlabel="Replaced feature_0 value (generated feature units)",
        ylabel="Average model prediction\n(generated target units)",
    )
    finish(
        fig,
        "Partial dependence averages a substituted-feature query",
        f"Background: {evaluation_split} · {len(frame)} returned grid values · other feature values retained",
        "Generated sample · actual VGI LightGBM v0.1.0 partial_dependence\n"
        "One grid value is reconstructed with SQL prediction calls; "
        "the curve is neither observed data nor a causal intervention.",
        name="partial-response",
    )
