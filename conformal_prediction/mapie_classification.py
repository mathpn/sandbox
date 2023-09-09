"""
This code is adapted from the MAPIE documentation for learning purposes.

https://mapie.readthedocs.io/en/latest/examples_classification/4-tutorials/plot_main-tutorial-classification.html#sphx-glr-examples-classification-4-tutorials-plot-main-tutorial-classification-py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import linear_model
from mapie.classification import MapieClassifier

ROUND_TO = 3
RANDOM_STATE = 42
RNG = np.random.default_rng(RANDOM_STATE)
ALPHA = 0.1

# %%
data = load_iris()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = data["target"]

# %%

df = pd.concat([X, pd.Series(y, name="class")], axis=1)
pear_corr = df.corr(method="pearson")
top_feats = pear_corr["class"].abs().sort_values()[:-1].index[:2].values.tolist()
X = X[top_feats]

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)
X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, random_state=RANDOM_STATE)

# %%
clf = linear_model.LogisticRegression()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).ravel()
mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
mapie_score.fit(X_cal, y_cal)

# %%


def plot_results(alphas, X_true, y_true, X, y_pred, y_ps):
    tab10 = plt.cm.get_cmap("Purples", 4)
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
    y_pred_col = list(map(colors.get, y_pred))
    y_true_col = list(map(colors.get, y_true))
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 10))
    axs = {0: ax1, 1: ax2, 2: ax3, 3: ax4}
    axs[0].scatter(X[:, 0], X[:, 1], color=y_pred_col, marker=".", s=10, alpha=0.05)
    axs[0].scatter(X_true[:, 0], X_true[:, 1], color=y_true_col, marker=".", s=30, alpha=1)
    axs[0].set_title("Predicted labels")
    for i, alpha in enumerate(alphas):
        y_pi_sums = y_ps[:, :, i].sum(axis=1)
        num_labels = axs[i + 1].scatter(
            X[:, 0], X[:, 1], c=y_pi_sums, marker=".", s=10, alpha=1, cmap=tab10, vmin=0, vmax=3
        )
        plt.colorbar(num_labels, ax=axs[i + 1])
        axs[i + 1].set_title(f"Number of labels for alpha={alpha}")
    plt.show()


# %%

step = 0.01
xx, yy = np.meshgrid(
    np.arange(X_test.iloc[:, 0].min(), X_test.iloc[:, 0].max(), step),
    np.arange(X_test.iloc[:, 1].min(), X_test.iloc[:, 1].max(), step),
)
X_test_mesh = np.stack([xx.ravel(), yy.ravel()], axis=1)

alpha = [0.2, 0.1, 0.05]
y_pred_score, y_ps_score = mapie_score.predict(X_test_mesh, alpha=alpha)

y_pred_score = y_pred_score.ravel()

plot_results(alpha, X_test.values, y_test, X_test_mesh, y_pred_score, y_ps_score)
# %%
