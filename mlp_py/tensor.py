import numpy as np
from typing import Union, Tuple, Optional


class Tensor:
    def __init__(
        self,
        data: Union[np.ndarray, list, float],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        self.data = (
            np.array(data, dtype=np.float32)
            if not isinstance(data, np.ndarray)
            else data.astype(np.float32)
        )
        self.requires_grad = requires_grad
        self.grad = None

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self, gradient: Optional[np.ndarray] = None):
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError("Gradient must be specified for non-scalar tensors")

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += gradient

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            g = out.grad
            if g is None:
                return

            if self.requires_grad:
                grad = g
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = g
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            g = out.grad
            if g is None:
                return
            if self.requires_grad:
                grad = g * other.data
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad

            if other.requires_grad:
                grad = g * self.data
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@",
        )

        def _backward():
            g = out.grad
            if g is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += g @ other.data.T

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ g

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(
            self.data**other,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{other}",
        )

        def _backward():
            g = out.grad
            if g is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += other * (self.data ** (other - 1)) * g

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward():
            g = out.grad
            if g is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                grad = g
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)

                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def reshape(self, *shape):
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward():
            g = out.grad
            if g is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += g.reshape(self.data.shape)

        out._backward = _backward
        return out

    def log(self):
        safe_data = np.maximum(self.data, 1e-8)
        out = Tensor(
            np.log(safe_data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward():
            g = out.grad
            if g is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += g / safe_data

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward():
            g = out.grad
            if g is None:
                return

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.data * g

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data, requires_grad=self.requires_grad, _children=(self,), _op="max"
        )

        def _backward():
            if not self.requires_grad or out.grad is None:
                return

            max_val_expanded = self.data.max(axis=axis, keepdims=True)
            mask = self.data == max_val_expanded

            grad_expand = out.grad
            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    grad_expand = np.expand_dims(grad_expand, ax)

            grad = mask * np.broadcast_to(grad_expand, self.data.shape)

            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += grad

        out._backward = _backward
        return out
