import numpy as np
np.random.seed(42)


class Affine():
    def __init__(self, in_dim: int, out_dim: int):
        """
        in_dim : explanatory_variable_dim_
        out_dim : target_variable_dim_
        w : weights
        b : bias
        dw : parameter_gradient
        db : bias_gradient
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.w = np.random.randn(self.in_dim, self.out_dim)
        self.b = np.zeros(self.out_dim, dtype=float)

        self.dx , self.dw, self.db = None, None, None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ forward_propagation
        x : input_data_ (batch_size, in_dim)
        out : target_data_ (batch_size, out_dim)
        """
        self.x = x
        self.out = np.dot(self.x, self.w) + self.b
        self.param = {'w' : self.w, 'b' : self.b}
        return  self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """ back_propagation
        grad : previous_gradient_ (batch_size, out_dim)
        dx : gradient_ (batch_size, in_dim)
        """
        self.dx = np.dot(grad, self.w.T)
        self.dw = np.dot(self.x.T, grad)
        self.db = np.sum(self.b)
        self.grad_param = {'x' : self.dx, 'w' : self.dw, 'b' : self.db}
        return self.dx