import numpy as np
np.random.seed(42)


import numpy as np
np.random.seed(42)


class Affine():
    def __init__(self, in_dim: int, out_dim: int):
        """
        in_dim : explanatory_variable_dim_
        out_dim : target_variable_dim_
        dw : parameter_gradient_
        db : bias_gradient_
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = np.random.randn(in_dim, out_dim)
        self.bias = np.zeros(out_dim, dtype=float)

        self.dx , self.dw, self.db = None, None, None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ forward_propagation
        x : input_data_ (batch_size, in_dim)
        output : output_data_ (batch_size, out_dim)
        """
        self.x = x
        output = np.dot(self.x, self.weight) + self.bias
        self.param = {'w' : self.weight, 'b' : self.bias}
        return  output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """ back_propagation
        grad : previous_gradient_ (batch_size, out_dim)
        dx : gradient_ (batch_size, in_dim)
        """
        dx = np.dot(grad, self.weight.T)
        dw = np.dot(self.x.T, grad)
        db = np.sum(self.bias)
        self.grad_param = {'w' : dw, 'b' : db}
        return dx