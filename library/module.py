import numpy as np
np.random.seed(42)

##################################################
#             Linear Transformation              #
##################################################
class Linear():
    def __init__(self, in_dim: int, out_dim: int):
        """
        in_dim : explanatory_variable_dim_
        out_dim : target_variable_dim_
        dw : parameter_gradient_
        db : bias_gradient_
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = np.random.randn(out_dim, in_dim)
        self.bias = np.zeros(out_dim, dtype=float)

        self.dx , self.dw, self.db = None, None, None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ forward_propagation
        x : input_data_ (batch_size, in_dim)
        output : output_data_ (batch_size, out_dim)
        """
        self.x = x
        # Affine
        output = np.dot(self.x, self.weight.T) + self.bias
        self.param = {'w' : self.weight, 'b' : self.bias}
        return  output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """ back_propagation
        grad : previous_gradient_ (batch_size, out_dim)
        dx : gradient_ (batch_size, in_dim)
        """
        # transpose x_shape
        if self.x.ndim == 2:
            x_T = self.x.T
        if self.x.ndim == 3:
            x_T = np.transpose(self.x, (0, 2, 1))
        if self.x.ndim == 4:
            x_T = np.transpose(self.x, (0, 1, 3, 2))
        # calculate gradient
        dx = np.dot(grad, self.weight)
        dw = np.dot(x_T, grad)
        db = np.sum(grad, axis=0)
        self.grad_param = {'w' : dw, 'b' : db}
        return dx

##################################################
#              Activation Function               #
##################################################

class ReLu():
    def __init__(self):
        self.mask = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ forward
        self.mask : boolean_mask_ where(x>0, True)
        """
        self.mask  = [x > 0]
        dims = list(x.shape)
        out = np.zeros(dims)
        out[x > 0] = x[x > 0]
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        mask = self.mask[0]
        dx = grad * mask
        return dx