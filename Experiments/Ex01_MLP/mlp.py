import numpy as np





#N = 3 # number of neurons per layer
#L = 2   # number of layers

def phi ( ell , x ):
    """ activation function for a neuron on layer ell """
    if ell == 0:
        return x
    else:
        return np.heaviside( x, 0 )
phi = np.frompyfunc( phi , 2 , 1 ) # make the function phi "universal" (ufunc in numpy)


def create_random_weights( N , L ):
    """ creates random weights b is a N-dim vector and w NxN-matrix"""
    w = np.random.rand(N,N,L)
    b = np.random.rand(N,L)
    #print("w = ", w)
    #print("b = ", b)
    return [b, w]


def transition ( b , w , f_input ):
    """ Transition function of the MLP: output = F(b,w,f_input) """
    # I'm trusting the sizes match and are well defined..
    N = b[ : , 0 ].size 
    L = b[ 0 , : ].size

    # Create the "layer-functions"
    X = np.empty(shape=(N, L+1), dtype=float)

    X[:, 0] = f_input
    for ell in range(L):
        #print( "ell = ", ell )
        #print( "w[ ell = ", ell, "] = ", w[ :, :, ell] )
        #print( "b[ ell = ", ell, "] = ", b[:, ell] )
        X[ : , ell+1 ] = b[ : , ell ]  +  w[ : , : , ell ] @ phi ( ell ,  X[ : , ell ]  )    
        #print( "phi = ", phi( ell , X[ : , ell ] ) )
        #print( "X[ ell+1 = ", ell+1,  " ] = ", X[ : , ell+1 ] )

    return X



# TESTS
def main():
    # Number of neurons in each layer
    N = 2  
    # Number of layers
    # (actually, excluding the input and the output the number of (hidden) layers is L-1,
    # that is, the 0-th layer is the input and the L-th layer is the output):
    L = 3

    f_input = np.random.rand(N)  # input data

    [b,w] = create_random_weights(N,L)  # weights
    X = transition( b , w , f_input )
    print( "MLP output = " , X[ : , L] )
    
if __name__ == "__main__":
    main()



