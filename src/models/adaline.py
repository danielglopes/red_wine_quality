import numpy as np


class AdalineGD:
    """
    ADALINE (Adaptive Linear Neuron) treinado com Gradiente Descendente em batch.

    Parameters
    ----------
    eta : float
        Taxa de aprendizado.
    n_iter : int
        Número de épocas de treinamento.
    random_state : int | None
        Semente para reprodutibilidade da inicialização dos pesos.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int | None = None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_: np.ndarray | None = None
        self.cost_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdalineGD":
        """
        Ajusta os pesos usando Gradiente Descendente (regra de Widrow-Hoff).

        Parameters
        ----------
        X : ndarray, shape = [n_amostras, n_features]
            Atributos de entrada.
        y : ndarray, shape = [n_amostras]
            Rótulos alvo em {-1, 1}.

        Returns
        -------
        self : AdalineGD
            Instância treinada.
        """
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features + 1)
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self._net_input(X)
            output = self._activation(net_input)
            errors = y - output
            # Atualização em batch
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """Calcula a soma ponderada."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _activation(self, X: np.ndarray) -> np.ndarray:
        """Função de ativação linear (identidade)."""
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna classe prevista {-1, 1} aplicando o limiar zero sobre a saída linear.
        """
        net_input = self._net_input(X)
        return np.where(net_input >= 0.0, 1, -1)
