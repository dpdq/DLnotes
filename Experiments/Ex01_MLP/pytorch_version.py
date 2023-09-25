"""
Literal reimplementation of mlp.py and back_prop.py using pytorch.
This allows us to check our formulas against the implementation
of the back propagation in pytorch.
"""
import torch
import torch.nn.functional as FF

import numpy as np
import back_prop
import mlp


def transition(b, W, f_input):
    """ Transition function of the MLP: output = F(b,W,f_input) """
    # I'm trusting the sizes match and are well defined..
    N = b.size(dim=0)
    L = b.size(dim=1)

    # Compute the "partial inputs", i.e. the Nx(L+1) array x
    #    x = torch.empty(N, L+1, requires_grad=True)  # initialization
    a = []  # we use a list instead of a vector to make autograd work 
    a.append(f_input)
    # x[:, 0] = f_input
    for ell in range(L):
        if ell == 0:
            #x[:, ell+1] = b[:, ell] + torch.matmul(W[:, :, ell],   x[:, ell])
            a.append(b[:, ell] + torch.matmul(W[:, :, ell],   a[-1]))
        else:
            #x[:, ell+1] = b[:, ell] + torch.matmul(W[:, :, ell],   FF.relu(x[:, ell]))
            a.append(b[:, ell] + torch.matmul(W[:, :, ell],   FF.relu(a[-1])))
    return a


def create_random_weights(N, L, debug=False):
    """ Returns [b,W] random weights, where b is a NxL-dim array and W NxNxL-matrix"""
    W = torch.rand(N, N, L, requires_grad=True)   # np.random.rand(N,N,L)
    b = torch.rand(N, L, requires_grad=True)  # np.random.rand(N,L)
    if debug is True:
        print("W = ", W)
        print("b = ", b)
    return [b, W]


def compute_S_gradS(b, W, a, f_training):
    N = b.size(dim=0)
    L = b.size(dim=1)
    #S = torch.matmul(x[:, L] - f_training, torch.ones(N))
    S = torch.sum(torch.square(a[-1]))
    S.backward()
    gradS_b = b.grad
    gradS_W = W.grad
    return [S, gradS_b, gradS_W]


# TESTS
def main(debug = False):
    # Number of neurons in each layer
    N = 3
    # Number of layers
    # (actually, excluding the input and the output the number of (hidden) layers is L-1,
    # that is, the 0-th layer is the input and the L-th layer is the output):
    L = 5

    # Pytorch version
    f_input = torch.rand(N, requires_grad=False)  # input data
    [b, W] = create_random_weights(N, L)  # weights
    a = transition(b, W, f_input)
    f_training = torch.zeros(N, requires_grad=False)
    [S, gradS_b, gradS_W] = compute_S_gradS(b, W, a, f_training)
    if debug:
        print("a = ", a)
        print(f"S = {S}")
        print(f"gradS_b = {gradS_b}")
        print(f"gradS_W = {gradS_W}")

    # Numpy version
    b_numpy = b.detach().numpy()
    W_numpy = W.detach().numpy()
    f_input_numpy = f_input.numpy()
    f_training_numpy = f_training.numpy()
    x_numpy = mlp.transition(b_numpy, W_numpy, f_input_numpy)
    [gradS_b_mumpy, gradS_W_numpy] = back_prop.compute_gradS(b_numpy,  W_numpy,  x_numpy,  f_training_numpy)
    S_numpy = sum(np.square(x_numpy[:, L]))
    if debug:
        print(f"S_numpy = {S_numpy}")
        print(f"gradS_b_mumpy = {gradS_b_mumpy}")
        print(f"gradS_W_numpy = {gradS_W_numpy}")

    # Comparison between the two versions
    print("We compare our numpy implementation of the backprop with the pytorch one.")
    print("If our implementation is correct the following tensors should be zero:")
    difference_gradS_b = np.around(gradS_b.detach().numpy() - gradS_b_mumpy,  4)
    difference_gradS_W = np.around(gradS_W.detach().numpy() - gradS_W_numpy,  4)
    print(f"difference_gradS_b = \n{difference_gradS_b}")
    print(f"difference_gradS_W = \n{difference_gradS_W}")


if __name__ == "__main__":
    main()
