import numpy as np


class MultilayerPerceptron:
    """
    MLP totalmente conectado treinado via Backpropagation (Rumelhart, Hinton & Williams).

    Parameters
    ----------
    hidden_layers : tuple[int, ...]
        Tamanho de cada camada oculta (ex.: (10, 5) cria duas camadas).
    lr : float
        Taxa de aprendizado.
    n_iter : int
        Número de épocas.
    random_state : int | None
        Semente para reprodutibilidade da inicialização.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (10,),
        lr: float = 0.01,
        n_iter: int = 100,
        random_state: int | None = None,
    ):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights_: list[np.ndarray] = []
        self.cost_: list[float] = []

    def _initialize_weights(self, n_features: int) -> None:
        """Inicializa pesos com distribuição normal pequena."""
        layer_sizes = (n_features,) + self.hidden_layers + (1,)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # +1 para o bias na dimensão de entrada
            w = rng.normal(loc=0.0, scale=0.1, size=(n_in + 1, n_out))
            self.weights_.append(w)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Aplicação da função sigmoide."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(z: np.ndarray) -> np.ndarray:
        """Derivada da sigmoide em função de z pré-ativação."""
        sig = 1.0 / (1.0 + np.exp(-z))
        return sig * (1.0 - sig)

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Executa forward pass.

        Returns
        -------
        activations : list
            Ativações por camada (inclui entrada em activations[0]).
        zs : list
            Somatórios ponderados (sem bias) por camada.
        """
        activations = [X]
        zs: list[np.ndarray] = []

        a = X
        for w in self.weights_:
            a_bias = np.c_[np.ones((a.shape[0], 1)), a]
            z = a_bias.dot(w)
            a = self._sigmoid(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultilayerPerceptron":
        """
        Treina a rede com Backpropagation.
        """
        # Conversão de rótulos para {0, 1}
        y_bin = np.where(y <= 0, 0.0, 1.0).reshape(-1, 1)
        n_samples, n_features = X.shape

        self._initialize_weights(n_features)
        self.cost_ = []

        for _ in range(self.n_iter):
            print(f"Época {_ + 1}/{self.n_iter}", end="\r")
            activations, zs = self._forward(X)
            output = activations[-1]

            # Erro
            error = output - y_bin

            # Custo (apenas para monitoramento)
            cost = (error**2).sum() / 2.0
            self.cost_.append(cost)

            # --- BACKPROPAGATION ---

            # Delta da camada de saída
            delta = error * self._sigmoid_deriv(zs[-1])

            grads: list[np.ndarray] = []

            for layer_idx in reversed(range(len(self.weights_))):
                a_prev = activations[layer_idx]

                # Adiciona bias na ativação anterior para bater com a dimensão dos pesos
                a_prev_bias = np.c_[np.ones((n_samples, 1)), a_prev]

                # --- CORREÇÃO AQUI ---
                # Removemos a divisão por n_samples.
                # Queremos a SOMA dos gradientes, não a média, para acelerar o treino.
                grad_w = a_prev_bias.T.dot(delta)

                grads.insert(0, grad_w)

                # Calcula o delta para a próxima iteração (camada anterior)
                if layer_idx != 0:
                    # Remove o peso do bias para propagar o erro
                    w_no_bias = self.weights_[layer_idx][1:, :]
                    delta = (delta.dot(w_no_bias.T)) * self._sigmoid_deriv(
                        zs[layer_idx - 1]
                    )

            # Atualização de pesos
            for i, grad in enumerate(grads):
                self.weights_[i] -= self.lr * grad

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades (saída da sigmoide)."""
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna classe prevista em {0, 1} usando limiar 0.5.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
