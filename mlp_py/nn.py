import numpy as np
from tensor import Tensor


class Module:
    def __init__(self):
        self._parameters = []

    def parameters(self):
        return self._parameters

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        k = np.sqrt(1 / in_features)
        self.weight = Tensor(
            np.random.uniform(-k, k, (in_features, out_features)), requires_grad=True
        )
        self._parameters.append(self.weight)

        if bias:
            self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
            self._parameters.append(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for layer in layers:
            if isinstance(layer, Module):
                self._parameters.extend(layer.parameters())

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, Module):
                x = layer(x)
            elif callable(layer):
                x = layer(x)
            else:
                raise ValueError(f"Layer must be Module or callable, got {type(layer)}")
        return x


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
        out = Tensor(
            x.data * mask, requires_grad=x.requires_grad, _children=(x,), _op="dropout"
        )

        def _backward():
            g = out.grad
            if g is None:
                return
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += g * mask

        out._backward = _backward
        return out


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones((1, num_features)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, num_features)), requires_grad=True)
        self._parameters.extend([self.gamma, self.beta])

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            diff = x - mean
            var = (diff * diff).mean(axis=0, keepdims=True)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.data
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.data
        else:
            mean = Tensor(self.running_mean, requires_grad=False)
            var = Tensor(self.running_var, requires_grad=False)

        x_norm = (x - mean) / (var + self.eps) ** 0.5
        out = x_norm * self.gamma + self.beta

        return out


def cross_entropy_loss(predictions: Tensor, targets: np.ndarray) -> Tensor:
    batch_size = predictions.shape[0]
    log_probs = log_softmax(predictions, axis=1)
    targets_one_hot = np.zeros_like(predictions.data)
    targets_one_hot[np.arange(batch_size), targets.astype(int).flatten()] = 1
    loss = -(Tensor(targets_one_hot) * log_probs).sum() / batch_size
    return loss


def log_softmax(x: Tensor, axis=-1) -> Tensor:
    x_max = x.max(axis=axis, keepdims=True)
    x_shifted = x - x_max
    logsumexp = x_shifted.exp().sum(axis=axis, keepdims=True).log()
    return x_shifted - logsumexp


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    diff = predictions - targets
    return (diff * diff).mean()
