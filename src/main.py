import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Importações dos seus módulos
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
    # Configuração de Argumentos via Terminal
    parser = argparse.ArgumentParser(description="Sommelier Artificial")

    # Parâmetros Gerais
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Taxa de aprendizado geral"
    )

    # Parâmetros Específicos do ADALINE (Convexo, convergiu rápido)
    parser.add_argument(
        "--epochs_ada", type=int, default=500, help="Épocas para ADALINE"
    )

    # Parâmetros Específicos do MLP (Não-convexo, precisa de tempo)
    parser.add_argument("--epochs_mlp", type=int, default=30000, help="Épocas para MLP")

    args = parser.parse_args()

    # 1. Carregar dados processados
    print("--- Carregando dados ---")
    X_train, X_test, y_train, y_test = data_prep.carregar_dados()

    if X_train is None:
        return

    # ==========================================================
    # 2. Treinamento do ADALINE
    # ==========================================================
    print(f"\n--- Treinando ADALINE ({args.epochs_ada} épocas) ---")
    # Usa o args.epochs_ada aqui
    ada = AdalineGD(eta=0.0001, n_iter=args.epochs_ada, random_state=1)
    ada.fit(X_train, y_train)

    y_pred_ada = ada.predict(X_test)
    acc_ada = accuracy_score(y_test, y_pred_ada)
    print(f"Acurácia ADALINE: {acc_ada:.2%}")
    print("Matriz de Confusão ADALINE:\n", confusion_matrix(y_test, y_pred_ada))

    # ==========================================================
    # 3. Treinamento do MLP
    # ==========================================================
    print(f"\n--- Treinando MLP ({args.epochs_mlp} épocas) ---")

    # Usa o args.epochs_mlp aqui
    mlp = MultilayerPerceptron(
        hidden_layers=(32, 16),
        lr=args.lr,
        n_iter=args.epochs_mlp,  # <--- AQUI ESTÁ A MUDANÇA
        random_state=42,
    )

    # MLP usa target 0/1 internamente, mas sua classe já trata a conversão no fit()
    # Porem, o y_test original é -1/1. Precisamos converter o y_test para 0/1
    # APENAS para calcular a métrica de acurácia do MLP corretamente
    y_test_mlp = np.where(y_test == -1, 0, 1)

    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)  # Retorna 0 ou 1

    acc_mlp = accuracy_score(y_test_mlp, y_pred_mlp)
    print(f"Acurácia MLP: {acc_mlp:.2%}")
    print("Matriz de Confusão MLP:\n", confusion_matrix(y_test_mlp, y_pred_mlp))

    # ==========================================================
    # 4. Geração de Gráficos (Essencial para o Relatório)
    # ==========================================================
    print("\n--- Gerando Gráficos ---")

    # Criar pasta de resultados se não existir
    dir_figs = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(dir_figs, exist_ok=True)

    # Gráfico 1: Comparação de Convergência (Custo)
    plt.figure(figsize=(10, 6))
    plot_cost_history([("Adaline", ada), ("MLP", mlp)])
    plt.savefig(os.path.join(dir_figs, "comparacao_custo.png"))
    print(f"Gráfico salvo em: {dir_figs}/comparacao_custo.png")

    # 6. Heatmaps de Confusão
    print("\n--- Gerando Heatmaps ---")
    plot_confusion_matrix(y_test, y_pred_ada, "ADALINE")
    plt.savefig(os.path.join(dir_figs, "confusion_matrix_adaline.png"))

    # MLP precisa converter y_test para 0/1 se ainda não estiver
    y_test_mlp = np.where(y_test == -1, 0, 1)
    plot_confusion_matrix(y_test_mlp, y_pred_mlp, "MLP")
    plt.savefig(os.path.join(dir_figs, "confusion_matrix_mlp.png"))

    # 7. Importância de Atributos (Química do Vinho)
    # Nomes das colunas na ordem do dataset
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

    # ==========================================================
    # SALVAMENTO DOS MODELOS (Persistência)
    # ==========================================================
    dir_models = os.path.join(os.path.dirname(__file__), "..", "results", "models")
    os.makedirs(dir_models, exist_ok=True)

    # Salvar ADALINE
    with open(os.path.join(dir_models, "adaline_model.pkl"), "wb") as f:
        pickle.dump(ada, f)
    print("Modelo ADALINE salvo em results/models/adaline_model.pkl")

    # Salvar MLP
    with open(os.path.join(dir_models, "mlp_model.pkl"), "wb") as f:
        pickle.dump(mlp, f)
    print("Modelo MLP salvo em results/models/mlp_model.pkl")

    print("\n--- FIM DO EXPERIMENTO ---")


if __name__ == "__main__":
    main()
