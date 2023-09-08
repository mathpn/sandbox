"""
This code is adapted from the MAPIE documentation for learning purposes.

https://mapie.readthedocs.io/en/latest/examples_regression/4-tutorials/plot_cqr_tutorial.html#sphx-glr-download-examples-regression-4-tutorials-plot-cqr-tutorial-py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble, tree

from mapie.metrics import regression_coverage_score, regression_mean_width_score
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample

ROUND_TO = 3
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)
ALPHA = 0.2

# %%
data = load_diabetes()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = data["target"]

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, random_state=RANDOM_STATE)

# %%

estimator = ensemble.RandomForestRegressor()
# estimator = tree.DecisionTreeRegressor()
# %%


def sort_y_values(y_test, y_pred, y_pis):
    """
    Sorting the dataset in order to make plots using the fill_between function.
    """
    indices = np.argsort(y_test)
    y_test_sorted = np.array(y_test)[indices]
    y_pred_sorted = y_pred[indices]
    y_lower_bound = y_pis[:, 0, 0][indices]
    y_upper_bound = y_pis[:, 1, 0][indices]
    return y_test_sorted, y_pred_sorted, y_lower_bound, y_upper_bound


def plot_prediction_intervals(
    title,
    axs,
    y_test_sorted,
    y_pred_sorted,
    lower_bound,
    upper_bound,
    coverage,
    width,
    num_plots_idx,
    rmse,
):
    """
    Plot of the prediction intervals for each different conformal
    method.
    """
    axs.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axs.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    lower_bound_ = np.take(lower_bound, num_plots_idx)
    y_pred_sorted_ = np.take(y_pred_sorted, num_plots_idx)
    y_test_sorted_ = np.take(y_test_sorted, num_plots_idx)

    error = y_pred_sorted_ - lower_bound_

    warning1 = y_test_sorted_ > y_pred_sorted_ + error
    warning2 = y_test_sorted_ < y_pred_sorted_ - error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=np.abs(error[~warnings]),
        capsize=5,
        marker="o",
        elinewidth=2,
        linewidth=0,
        label="Inside prediction interval",
    )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=np.abs(error[warnings]),
        capsize=5,
        marker="o",
        elinewidth=2,
        linewidth=0,
        color="red",
        label="Outside prediction interval",
    )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        marker="*",
        color="green",
        label="True value",
    )
    axs.set_xlabel("True disease progression score")
    axs.set_ylabel("Prediction of disease progression score")
    ab = AnnotationBbox(
        TextArea(
            f"Coverage: {np.round(coverage, ROUND_TO)}\n"
            + f"Interval width: {np.round(width, ROUND_TO)}\n"
            + f"RMSE: {np.round(rmse, ROUND_TO)}"
        ),
        xy=(np.min(y_test_sorted_) * 3, np.max(y_pred_sorted_ + error) * 0.95),
    )
    lims = [
        np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
        np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes
    ]
    axs.plot(lims, lims, "--", alpha=0.75, color="black", label="x=y")
    axs.add_artist(ab)
    axs.set_title(title, fontweight="bold")


# %%
STRATEGIES = {
    "naive": {"method": "naive"},
    "cv_plus": {"method": "plus", "cv": 10},
    "cv_base": {"method": "base", "cv": 10},
    "jackknife_plus_ab": {"method": "plus", "cv": Subsample(n_resamplings=50)},
    # "cqr": {"method": "quantile", "cv": "split", "alpha": ALPHA},
}
y_pred, y_pis = {}, {}
y_test_sorted, y_pred_sorted, lower_bound, upper_bound = {}, {}, {}, {}
coverage, width, rmse = {}, {}, {}
for strategy, params in STRATEGIES.items():
    mapie = MapieRegressor(estimator, **params, random_state=RANDOM_STATE)
    mapie.fit(X_train, y_train)
    y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=ALPHA)
    rmse[strategy] = np.sqrt(mean_squared_error(y_test, y_pred[strategy]))
    (
        y_test_sorted[strategy],
        y_pred_sorted[strategy],
        lower_bound[strategy],
        upper_bound[strategy],
    ) = sort_y_values(y_test, y_pred[strategy], y_pis[strategy])
    coverage[strategy] = regression_coverage_score(
        y_test, y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
    )
    width[strategy] = regression_mean_width_score(
        y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
    )
# %%

perc_obs_plot = 0.5
num_plots = RNG.choice(len(y_test), int(perc_obs_plot * len(y_test)), replace=False)
fig, axs = plt.subplots(2, 2, figsize=(15, 13))
coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
for strategy, coord in zip(STRATEGIES.keys(), coords):
    plot_prediction_intervals(
        strategy,
        coord,
        y_test_sorted[strategy],
        y_pred_sorted[strategy],
        lower_bound[strategy],
        upper_bound[strategy],
        coverage[strategy],
        width[strategy],
        num_plots,
        rmse[strategy],
    )
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
plt.legend(
    lines[:4],
    labels[:4],
    loc="upper center",
    bbox_to_anchor=(0, -0.15),
    fancybox=True,
    shadow=True,
    ncol=2,
)
plt.show()
# %%
