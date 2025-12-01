import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
    # 1. Carregar dados processados
    print("--- Carregando dados ---")
    X_train, X_test, y_train, y_test = data_prep.carregar_dados()

    if X_train is None:
        return

    # ==========================================================
    # 2. Treinamento do ADALINE (Baseline Linear)
    # ==========================================================
    print("\n--- Treinando ADALINE ---")
    # Nota: Adaline é sensível ao Learning Rate (eta). Se o custo subir, diminua o eta.
    ada = AdalineGD(eta=0.0001, n_iter=1000, random_state=1)
    ada.fit(X_train, y_train)

    y_pred_ada = ada.predict(X_test)
    acc_ada = accuracy_score(y_test, y_pred_ada)
    print(f"Acurácia ADALINE: {acc_ada:.2%}")
    print("Matriz de Confusão ADALINE:\n", confusion_matrix(y_test, y_pred_ada))

    # ==========================================================
    # 3. Treinamento do MLP (Deep Learning)
    # ==========================================================
    print("\n--- Treinando Perceptron Multicamadas (MLP) ---")
    # Configuração: 2 camadas ocultas com 16 e 8 neurônios
    mlp = MultilayerPerceptron(
        hidden_layers=(32, 16, 8),  # Três camadas, mais neurônios (antes era 16)
        lr=0.001,  # Taxa de aprendizado conservadora
        n_iter=30000,  # Muito mais épocas (Backprop é lento)
        random_state=42,  # Mude a semente para tentar um ponto de partida melhor
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
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]

    print("\n--- Gerando Importância de Atributos ---")
    plot_feature_importance(ada, colunas)
    plt.savefig(os.path.join(dir_figs, "feature_importance_adaline.png"))

    print("\n--- FIM DO EXPERIMENTO ---")


if __name__ == "__main__":
    main()
