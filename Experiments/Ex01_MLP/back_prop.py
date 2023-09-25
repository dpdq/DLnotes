"""
Implementation of backprop in numpy.
"""
import mlp

import numpy as np
import matplotlib.pyplot as plt


def compute_gradS(b, W, x, f_training, debug=False):
    """
    Computes the gradients of the cost function S
    via back propagation algorithm.
    """
    # I'm trusting the sizes of b, W, x match and are well defined..
    N = b[:, 0].size
    L = b[0, :].size

    # Initialize empty multi-arrays
    #    gradS_b = np.empty(shape=(N, L+1), dtype=float)  # note the L+1
    gradS_b = np.empty(shape=(N, L), dtype=float)
    gradS_W = np.empty(shape=(N, N, L), dtype=float)

    # Back-prop algorithm
    gradS_b[:, L-1] = 2*(x[:, L] - f_training)
    for ell in reversed(range(L)):
        TPhi = np.tensordot(np.identity(N), mlp.phi(ell, x[:, ell]),  0)  # Phi(x^ell)^{H <-- W}
        DPhi = np.diag(mlp.phi_prime(ell, x[:, ell]),  0)  # DPhi( x^ell )
        if debug:
            print(f"back_prop.compute_gradS: ell = {ell}")
            print(DPhi.shape, W.shape,   gradS_b[:, ell].shape)
            print("DPhi = ", DPhi)
            print("W[:,:,ell] = ", W[:, :, ell])
            print("x[:,ell] = ", x[:, ell])
            print("TPhi = ", TPhi)
        if ell > 0:
            gradS_b[:, ell-1] = np.transpose(DPhi) @ np.transpose(W[:, :, ell]) @ gradS_b[:, ell]  # back-prop , (1)
        gradS_W[:, :, ell] = np.transpose(TPhi, [1, 2, 0]) @ gradS_b[:, ell]  # (2)

    return [gradS_b, gradS_W]  # note we cut the extra (0th) el. of gradS_b

    # Remarks:
    # (1) the transpose of DPhi is redundant since DPhi is diagonal, it's there only for readability;
    # (2) instead of transposing TPhi we could have defined it already in that shape.


# TESTS
def simpletest_1():
    # Number of neurons in each layer, N >= 1
    N = 1
    # Number of ''layers'', L >= 1
    # actually, excluding the input and the output the number of (hidden) layers is L-1,
    # that is, the 0-th layer is the input and the L-th layer is the output:
    L = 4

    steps = 10000
    linear_vector = np.linspace(-1, 1, steps)

    # f_input = np.random.rand(N)  # input data
    f_input = 1

    # [b,W] = mlp.create_random_weights(N,L)  # weights
    linear_b = np.empty(shape=(N, L, steps), dtype=float)
    linear_b[0, 0, :] = linear_vector
    linear_W = np.ones(shape=(N, N, L, steps), dtype=float)
    linear_W[0, 0, 0, :] = linear_vector
    # print( "b = ", b )
    # print( "W = ", W )

    response_x = np.empty(shape=(steps))
    print(response_x.shape)
    for i in range(steps):
        x = mlp.transition(linear_b[:, :, i],   linear_W[:, :, :, i],   f_input)
        response_x[i] = x[0, L]
        print("linear_W = ", linear_W[0, 0, 0, i])
        print("response_x = ", response_x[i])

    plt.plot(linear_W[0, 0, 0, :],  response_x[:])
    plt.show()

    # print( "mlp.phi( x[ : , L ] ) = ", mlp.phi( L , x[:,L] ) )

    f_training = 0  # np.random.rand(N)  # training data (only one point for the moment)
    [gradS_b, gradS_W] = compute_gradS(linear_b[:, :, 0],  linear_W[:, :, :, 0],  x,  f_training)
    print("gradS_b, gradS_W = ", gradS_b, gradS_W)


def main():
    simpletest_1()


if __name__ == "__main__":
    main()
