# 🍷 O Sommelier Artificial: Redes Neurais vs. Vinho

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/Numpy-Implementation-013243?style=for-the-badge&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)

> **Um estudo comparativo entre modelos Lineares (ADALINE) e Não-Lineares (Deep Learning) na predição da qualidade de vinhos tintos, implementados matematicamente do zero.**

---

## 📖 Sobre o Projeto

A indústria vinícola depende tradicionalmente de especialistas humanos (sommeliers) para avaliar a qualidade de safras, um processo subjetivo e não escalável. Este projeto propõe um **sistema de suporte à decisão** capaz de classificar vinhos baseando-se exclusivamente em suas propriedades físico-químicas (como pH, teor alcoólico, acidez, etc.).

O objetivo acadêmico central é confrontar duas abordagens de Aprendizado de Máquina implementadas com **pura matemática matricial (NumPy)**, sem o uso de frameworks de alto nível para a lógica das redes:

1.  **ADALINE (Adaptive Linear Neuron):** Representando a abordagem linear clássica (Regra de Widrow-Hoff).
2.  **MLP (Multilayer Perceptron):** Representando a abordagem moderna de Deep Learning (Backpropagation).

---

## 🧪 O Problema & A Química

Utilizamos a base de dados pública **[Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)**.

* **Entrada:** 11 variáveis físico-químicas (Ex: Acidez fixa, Açúcar residual, Cloretos, Densidade, Álcool...).
* **Saída:** Qualidade (Nota de 0 a 10).
* **O Desafio:** Detectar vinhos **"Premium"** (Notas 7 e 8) em meio a uma maioria de vinhos comuns.

### A Hipótese
> *"Será que a relação entre a química e a qualidade é linear (mais álcool = melhor), ou existem interações complexas e sutis que apenas uma Rede Neural Profunda consegue capturar?"*

---

## ⚙️ Arquitetura do Projeto

A estrutura foi organizada seguindo boas práticas de Ciência de Dados para garantir reprodutibilidade.

```text
red_wine_quality/
├── data/
│   ├── raw/                  # Dataset original (winequality-red.csv)
│   └── processed/            # Arrays NumPy normalizados (.npy)
│
├── results/
│   ├── figures/              # Gráficos gerados (Matrizes, Custos, etc.)
│   └── models/               # Modelos treinados salvos (.pkl)
│
├── src/
│   ├── models/
│   │   ├── adaline.py        # Implementação manual do ADALINE
│   │   └── perceptron.py     # Implementação manual do MLP (Backprop)
│   ├── data_prep.py          # Pipeline de limpeza e normalização
│   ├── visualization.py      # Geração de gráficos
│   └── main.py               # Script principal de execução
│
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação
````

-----

## 🚀 Como Executar

### 1\. Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado.

```bash
# Clone o repositório
git clone [https://github.com/seu-usuario/sommelier-artificial.git](https://github.com/seu-usuario/sommelier-artificial.git)
cd sommelier-artificial

# Crie um ambiente virtual (Recomendado)
python -m venv .venv

# Ative o ambiente
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 2\. Preparação dos Dados

Este script realiza a limpeza, separa as classes (Bons vs Ruins com corte em 6.5) e normaliza os dados (Z-Score).

```bash
python src/data_prep.py
```

### 3\. Treinamento e Teste

Execute o `main.py`. Você pode ajustar os hiperparâmetros via linha de comando:

```bash
# Execução padrão (500 épocas pro ADALINE, 30k pro MLP)
python src/main.py

# Teste personalizado (Ex: Ajuste fino no Deep Learning)
python src/main.py --epochs_mlp 50000 --lr 0.002
```

**Argumentos disponíveis:**

  * `--lr`: Taxa de aprendizado global (Default: 0.001)
  * `--epochs_ada`: Épocas para o ADALINE (Default: 500)
  * `--epochs_mlp`: Épocas para o MLP (Default: 30000)
  * `--seed`: Semente de aleatoriedade para reprodutibilidade (Default: 42)

-----

## 📊 Resultados e Análise

Os resultados demonstram a superioridade de modelos não-lineares em cenários de dados desbalanceados.

| Modelo                  | Acurácia Global | Capacidade de Detecção (Sensibilidade)                                                                                 |
| :---------------------- | :-------------: | :--------------------------------------------------------------------------------------------------------------------- |
| **ADALINE**             |     \~86.8%     | **Baixa.** Tende a classificar quase tudo como "Ruim" para minimizar o erro médio. Falha em encontrar os vinhos raros. |
| **MLP (Deep Learning)** |   **\~93.4%**   | **Alta.** Consegue desenhar fronteiras complexas para isolar e identificar corretamente os vinhos Premium.             |

### Visualizações Geradas (`results/figures/`)

1.  **Comparação de Custo:** Mostra a convergência rápida do modelo linear (convexo) vs. a descida lenta e complexa do modelo profundo (não-convexo).
2.  **Matriz de Confusão:** O "mapa da verdade" que revela onde cada modelo errou.
3.  **Importância de Atributos:** Revela quais químicos o modelo considerou cruciais (Ex: Álcool positivo, Acidez Volátil negativa).
4.  **Fronteira de Decisão:** Uma prova visual de que o problema não é linearmente separável.

-----

## 🛠️ Tecnologias Utilizadas

  * **Python 3:** Linguagem base.
  * **NumPy:** Todo o cálculo matricial, gradientes e funções de ativação.
  * **Pandas:** Manipulação e leitura do dataset.
  * **Matplotlib & Seaborn:** Visualização de dados e gráficos estatísticos.
  * **Scikit-Learn:** Apenas para métricas de avaliação e normalização (não usado para os modelos).

-----
Este projeto foi desenvolvido com fins educacionais. Beba com moderação. 🍷