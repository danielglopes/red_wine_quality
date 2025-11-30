import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preparar_e_salvar_dados():
    """
    Função principal que:
    1. Lê o CSV bruto.
    2. Realiza o pré-processamento (Binarização + Normalização).
    3. Salva os arrays numpy (.npy) prontos em data/processed.
    """
    # --- 1. Definição de Caminhos ---
    dir_atual = os.path.dirname(os.path.abspath(__file__))
    path_raw = os.path.join(dir_atual, "..", "data", "raw", "winequality-red.csv")
    dir_processed = os.path.join(dir_atual, "..", "data", "processed")

    # Cria a pasta de destino se não existir
    os.makedirs(dir_processed, exist_ok=True)

    print(f"--- PROCESSAMENTO DE DADOS ---")

    # --- 2. Carregamento ---
    try:
        print(f"Lendo arquivo: {path_raw}")
        df = pd.read_csv(path_raw)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo não encontrado em {path_raw}")
        print(
            "Verifique se o nome do arquivo é 'winequality-red.csv' e se está na pasta data/raw"
        )
        return

    # --- 3. Engenharia de Features e Target ---
    X = df.drop("quality", axis=1).values
    y_raw = df["quality"].values

    # Target Binarizado: Nota >= 6 é Bom (1), senão Ruim (-1)
    # Ideal para ADALINE e Tanh
    y = np.where(y_raw >= 6, 1, -1)

    print(f"Distribuição das classes: {np.unique(y, return_counts=True)}")

    # --- 4. Divisão Treino/Teste ---
    # Random State 42 garante que o sorteio seja sempre igual
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- 5. Normalização (StandardScaler) ---
    scaler = StandardScaler()

    # Fit apenas no treino para evitar vazamento de dados
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- 6. Salvamento (Checkpoint) ---
    print(f"Salvando arrays processados em: {dir_processed}")

    np.save(os.path.join(dir_processed, "X_train.npy"), X_train)
    np.save(os.path.join(dir_processed, "X_test.npy"), X_test)
    np.save(os.path.join(dir_processed, "y_train.npy"), y_train)
    np.save(os.path.join(dir_processed, "y_test.npy"), y_test)

    print("--- CONCLUÍDO COM SUCESSO ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")


def carregar_dados():
    """
    Função utilitária para ser usada pelos modelos.
    Carrega os dados já processados da pasta data/processed.
    """
    dir_atual = os.path.dirname(os.path.abspath(__file__))
    dir_processed = os.path.join(dir_atual, "..", "data", "processed")

    try:
        X_train = np.load(os.path.join(dir_processed, "X_train.npy"))
        X_test = np.load(os.path.join(dir_processed, "X_test.npy"))
        y_train = np.load(os.path.join(dir_processed, "y_train.npy"))
        y_test = np.load(os.path.join(dir_processed, "y_test.npy"))
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("Erro: Arquivos processados não encontrados.")
        print("Execute 'python src/data_prep.py' primeiro.")
        return None, None, None, None


# Executa o processamento apenas se rodar o arquivo diretamente
if __name__ == "__main__":
    preparar_e_salvar_dados()
