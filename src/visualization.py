from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_decision_regions(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    resolution: float = 0.02,
) -> None:
    """
    Plota fronteiras de decisão para datasets 2D.

    Parameters
    ----------
    X : ndarray, shape = [n_amostras, 2]
        Atributos (apenas duas features já selecionadas/treinadas).
    y : ndarray
        Rótulos em {-1, 1} ou {0, 1}.
    classifier : objeto com método predict(X)
        Modelo já treinado.
    resolution : float
        Passo da malha para o grid de decisão.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_decision_regions requer X com exatamente 2 features.")

    markers = ("s", "x", "o", "^", "v")
    colors = ("lightblue", "lightcoral", "lightgreen")
    cmap = plt.cm.RdBu

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    Z = classifier.predict(grid_points)
    Z = np.where(Z <= 0, 0, 1).reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    y_plot = np.where(y <= 0, 0, 1)
    for idx, cl in enumerate(np.unique(y_plot)):
        plt.scatter(
            x=X[y_plot == cl, 0],
            y=X[y_plot == cl, 1],
            alpha=0.8,
            c=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            label=f"Classe {cl}",
            edgecolor="k",
        )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()


def plot_cost_history(models_list: Iterable[Tuple[str, object]]) -> None:
    """
    Plota a curva de custo por época para múltiplos modelos.

    Parameters
    ----------
    models_list : iterable de tuplas (label, modelo)
        Cada modelo deve expor o atributo cost_ contendo a sequência de custos.
    """
    for label, model in models_list:
        cost = getattr(model, "cost_", None)
        if cost is None:
            continue
        epochs = np.arange(1, len(cost) + 1)
        plt.plot(epochs, cost, label=label)

    plt.xlabel("Épocas")
    plt.ylabel("Custo (SSE)")
    plt.title("Convergência do custo por época")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plota um Heatmap da Matriz de Confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()


def plot_feature_importance(model, feature_names):
    """
    Plota os pesos do ADALINE como importância de atributos.
    """
    # Pega os pesos (ignorando o bias w[0])
    weights = model.w_[1:]

    plt.figure(figsize=(10, 6))
    colors = ["red" if w < 0 else "green" for w in weights]
    plt.barh(feature_names, weights, color=colors)
    plt.title("Importância das Características (Pesos do ADALINE)")
    plt.xlabel("Peso (Influência na Qualidade)")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()
