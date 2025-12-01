import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

import data_prep
from models.adaline import AdalineGD
from models.perceptron import MultilayerPerceptron
from visualization import (
    plot_confusion_matrix,
    plot_cost_history,
    plot_decision_regions,
    plot_feature_importance,
)


def main():
    parser = argparse.ArgumentParser(description="Sommelier Artificial")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Taxa de aprendizado geral"
    )
    parser.add_argument(
        "--epochs_ada", type=int, default=500, help="Épocas para ADALINE"
    )
    parser.add_argument("--epochs_mlp", type=int, default=30000, help="Épocas para MLP")

    args = parser.parse_args()

    print("--- Carregando dados ---")
    X_train, X_test, y_train, y_test = data_prep.carregar_dados()

    if X_train is None:
        return

    print(f"\n--- Treinando ADALINE ({args.epochs_ada} épocas) ---")
    ada = AdalineGD(eta=0.0001, n_iter=args.epochs_ada, random_state=1)
    ada.fit(X_train, y_train)

    y_pred_ada = ada.predict(X_test)
    acc_ada = accuracy_score(y_test, y_pred_ada)
    print(f"Acurácia ADALINE: {acc_ada:.2%}")
    print("Matriz de Confusão ADALINE:\n", confusion_matrix(y_test, y_pred_ada))

    print(f"\n--- Treinando MLP ({args.epochs_mlp} épocas) ---")
    mlp = MultilayerPerceptron(
        hidden_layers=(32, 16, 8),
        lr=args.lr,
        n_iter=args.epochs_mlp,
        random_state=42,
    )

    # Converter y_test para {0, 1} para calcular acurácia do MLP
    y_test_mlp = np.where(y_test == -1, 0, 1)

    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)

    acc_mlp = accuracy_score(y_test_mlp, y_pred_mlp)
    print(f"Acurácia MLP: {acc_mlp:.2%}")
    print("Matriz de Confusão MLP:\n", confusion_matrix(y_test_mlp, y_pred_mlp))

    print("\n--- Gerando Gráficos ---")

    dir_figs = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(dir_figs, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plot_cost_history([("Adaline", ada), ("MLP", mlp)])
    plt.savefig(os.path.join(dir_figs, "comparacao_custo.png"))
    print(f"Gráfico salvo em: {dir_figs}/comparacao_custo.png")

    print("\n--- Gerando Heatmaps ---")
    plot_confusion_matrix(y_test, y_pred_ada, "ADALINE")
    plt.savefig(os.path.join(dir_figs, "confusion_matrix_adaline.png"))

    y_test_mlp = np.where(y_test == -1, 0, 1)
    plot_confusion_matrix(y_test_mlp, y_pred_mlp, "MLP")
    plt.savefig(os.path.join(dir_figs, "confusion_matrix_mlp.png"))

    colunas = [
        "acidez fixa",
        "acidez volátil",
        "ácido cítrico",
        "açúcar residual",
        "cloretos",
        "dióxido de enxofre livre",
        "dióxido de enxofre total",
        "densidade",
        "pH",
        "sulfatos",
        "álcool",
    ]

    print("\n--- Gerando Importância de Atributos ---")
    plot_feature_importance(ada, colunas)
    plt.savefig(os.path.join(dir_figs, "feature_importance_adaline.png"))

    dir_models = os.path.join(os.path.dirname(__file__), "..", "results", "models")
    os.makedirs(dir_models, exist_ok=True)

    with open(os.path.join(dir_models, "adaline_model.pkl"), "wb") as f:
        pickle.dump(ada, f)
    print("Modelo ADALINE salvo em results/models/adaline_model.pkl")

    with open(os.path.join(dir_models, "mlp_model.pkl"), "wb") as f:
        pickle.dump(mlp, f)
    print("Modelo MLP salvo em results/models/mlp_model.pkl")

    print("\n--- FIM DO EXPERIMENTO ---")


if __name__ == "__main__":
    main()
