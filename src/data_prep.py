import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preparar_e_salvar_dados():
    """
    Versão Kaggle: Define 'Bom' apenas como notas > 6.5 (7 e 8).
    Isso gera uma acurácia base mais alta, mas desbalanceia as classes.
    """
    dir_atual = os.path.dirname(os.path.abspath(__file__))
    path_raw = os.path.join(dir_atual, "..", "data", "raw", "winequality-red.csv")
    dir_processed = os.path.join(dir_atual, "..", "data", "processed")

    os.makedirs(dir_processed, exist_ok=True)
    print(f"--- PROCESSAMENTO DE DADOS (ESTILO KAGGLE) ---")

    try:
        df = pd.read_csv(path_raw)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {path_raw}")
        return

    X = df.drop("quality", axis=1).values
    y_raw = df["quality"].values

    # --- MUDANÇA PRINCIPAL AQUI ---
    # Notebook Kaggle: bins = (2, 6.5, 8)
    # Notas > 6.5 (7, 8) são Bons (1). O resto (3,4,5,6) é Ruim (-1).
    y = np.where(y_raw > 6.5, 1, -1)

    print(f"Distribuição das classes (Imbalance Warning!):")
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    # Você verá algo como {-1: 1382, 1: 217}

    # Kaggle usa test_size=0.2
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    # Mantemos o fit apenas no treino (o Kaggle errou nisso, mas corrigimos aqui)
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Salvando em: {dir_processed}")
    np.save(os.path.join(dir_processed, "X_train.npy"), X_train)
    np.save(os.path.join(dir_processed, "X_test.npy"), X_test)
    np.save(os.path.join(dir_processed, "y_train.npy"), y_train)
    np.save(os.path.join(dir_processed, "y_test.npy"), y_test)

    print("--- CONCLUÍDO ---")


def carregar_dados():
    dir_atual = os.path.dirname(os.path.abspath(__file__))
    dir_processed = os.path.join(dir_atual, "..", "data", "processed")
    try:
        X_train = np.load(os.path.join(dir_processed, "X_train.npy"))
        X_test = np.load(os.path.join(dir_processed, "X_test.npy"))
        y_train = np.load(os.path.join(dir_processed, "y_train.npy"))
        y_test = np.load(os.path.join(dir_processed, "y_test.npy"))
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        return None, None, None, None


if __name__ == "__main__":
    preparar_e_salvar_dados()
