import mlp

import numpy as np

def compute_gradS( b , W , x , f_training ):
    """ Computes the gradients of the cost function S via back propagation algorithm."""
    # I'm trusting the sizes of b, W, x match and are well defined..
    N = b[ : , 0 ].size 
    L = b[ 0 , : ].size
    
    # Initialize empty multi-arrays
    gradS_b = np.empty( shape=(N,L+1), dtype=float )  # note the L+1
    gradS_W = np.empty( shape=(N,N,L), dtype=float )

    # Back-prop algorithm
    gradS_b[ : , L ] = x[ : , L ]  - f_training 
    for ell in reversed(range(L)):
        TPhi = np.tensordot( np.identity(N) , mlp.phi( ell , x[: ,ell] )  ,  0 )  # Phi(x^ell)^{H <-- W}
        DPhi = np.diag(  mlp.phi_prime( ell , x[ : ,ell] )   ,  0)  # DPhi( x^ell )
        print(DPhi.shape, W.shape, gradS_b[ : , ell + 1].shape )
        gradS_b[ : , ell  ] = np.transpose( DPhi ) @ np.transpose( W[ : , : , ell] ) @ gradS_b[ : , ell + 1 ]  # back-prop , (1)
        gradS_W[ : , : , ell ] = np.transpose(  TPhi , [1,2,0] ) @ gradS_b[ : , ell ]  # (2)

    return [gradS_b, gradS_W]

    # Remarks:
    # (1) the transpose of DPhi is redundant since DPhi is diagonal, it's there only for readability;
    # (2) instead of transposing TPhi we could have defined it already in that shape.


# TESTS
def main():
    # Number of neurons in each layer
    N = 2  
    # Number of layers
    # (actually, excluding the input and the output the number of (hidden) layers is L-1,
    # that is, the 0-th layer is the input and the L-th layer is the output):
    L = 3

    f_input = np.random.rand(N)  # input data

    
    [b,W] = mlp.create_random_weights(N,L)  # weights
    x = mlp.transition( b , W , f_input )
    print( "MLP output = " , x[ : , L] )

    print( "mlp.phi( x[ : , L ] ) = ", mlp.phi( L , x[:,L] ) )

    f_training = np.random.rand(N)  # training data (only one point for the moment)
    [gradS_b, gradS_W] = compute_gradS(b, W, x, f_training)
    print( gradS_b, gradS_W )

    
if __name__ == "__main__":
    main()


