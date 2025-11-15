import numpy as np

from tensor import Tensor


def relu(x: Tensor) -> Tensor:
    out = Tensor(
        np.maximum(0, x.data), requires_grad=x.requires_grad, _children=(x,), _op="ReLU"
    )

    def _backward():
        g = out.grad
        if g is None:
            return
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += (x.data > 0) * g

    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    data = x.data
    t = (np.exp(2 * data) - 1) / (np.exp(2 * data) + 1)
    out = Tensor(t, requires_grad=x.requires_grad, _children=(x,), _op="tanh")

    def _backward():
        g = out.grad
        if g is None:
            return
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += (1 - t**2) * g

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    sig = 1 / (1 + np.exp(-x.data))
    out = Tensor(sig, requires_grad=x.requires_grad, _children=(x,), _op="sigmoid")

    def _backward():
        g = out.grad
        if g is None:
            return
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += sig * (1 - sig) * g

    out._backward = _backward
    return out


def softmax(x: Tensor, axis=-1) -> Tensor:
    exp_x = (x - Tensor(x.data.max(axis=axis, keepdims=True))).exp()
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    out = Tensor(
        np.where(x.data > 0, x.data, alpha * x.data),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="LeakyReLU",
    )

    def _backward():
        g = out.grad
        if g is None:
            return
        if x.requires_grad:
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += np.where(x.data > 0, 1, alpha) * g

    out._backward = _backward
    return out
