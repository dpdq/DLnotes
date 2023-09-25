"""
Notation:
--------------
N: number of neurons per layer
L -1: number of hidden layers
ell: index in [0, ... , L -1]
f_input: N array, this is layer 0 of the MLP (MultiLayer Percetron)
x: Nx(L+1) array, x[ : , ell] are the inputs for the layer ell+1, x[ : , L] are the outputs
b: NxL array, b[ : , ell] are the additive weights for the layer ell+1
W: NxNxL array, W[ : , : , ell] are the ''matrix-weights" for the layer ell+1
phi: activation function (component wise, i.e. "universal" for numpy)
phi_prime: derivative of phi (component wise, i.e. "universal" for numpy)
transition: transition function of the MLP, returns x: NxL array (see above)
"""

import numpy as np


def phi (ell, x, debug=False):
    """
    Activation function for a neuron on layer ell.
    phi( ell , x ) = 0 when ell=0,
    phi( ell , x ) = max(0,x) when ell \ne 0.
    """
    if ell == 0:
        if debug:
            print("ell==0")
        return x
    else:
        if debug:
            print("ell != 0")
        return np.heaviside(x, 0) * x
phi = np.frompyfunc(phi, 2, 1)  # make the function phi "universal" (ufunc in numpy)


def phi_prime(ell, x):
    """ Derivative of the activation function for a neuron on layer ell """
    if ell == 0:
        return 1
    else:
        return np.heaviside(x, 0)
phi_prime = np.frompyfunc(phi_prime, 2, 1)  # make the function phi "universal" (ufunc in numpy)


def transition(b, W, f_input):
    """ Transition function of the MLP: output = F(b,W,f_input) """
    # I'm trusting the sizes match and are well defined..
    N = b[:, 0].size
    L = b[0, :].size

    # Compute the "partial inputs", i.e. the Nx(L+1) array x
    x = np.empty(shape=(N, L+1), dtype=float)  # initialization
    x[:, 0] = f_input
    for ell in range(L):
        x[:, ell+1] = b[:, ell] + W[:, :, ell] @ phi(ell,  x[:, ell])
    return x


def create_random_weights(N, L, debug=False):
    """ Returns [b,W] random weights, where b is a NxL-dim array and W NxNxL-matrix"""
    W = np.random.rand(N, N, L)
    b = np.random.rand(N, L)
    if debug:
        print("W = ", W)
        print("b = ", b)
    return [b, W]


# TESTS
def main():
    # Number of neurons in each layer
    N = 2
    # Number of layers
    # (actually, excluding the input and the output the number of (hidden) layers is L-1,
    # that is, the 0-th layer is the input and the L-th layer is the output):
    L = 3

    f_input = np.random.rand(N)  # input data

    [b, W] = create_random_weights(N, L)  # weights
    x = transition(b, W, f_input)
    print("MLP output = ", x[:, L])


if __name__ == "__main__":
    main()
