import numpy as np

from get_frontal_faces import get_frontal_faces
from t_prod import t_prod


def t_SVD(A, frontal_faces = None):
    """Tensor singular value decomposition 
    
    Decomposition of a tensor A = U * S * V (* denoting the tensor product from [1]) 
    where U and V are both orthogonal tensors and S is 'diagonal', as derived in [1] 
    and reproduced in Theorom 2.23 of [2].
    
    Parameters
    ----------
    A : np-dimensional array
        n1 x n2 x n3 x ... x np tensor
       
    Returns
    ----------       
    U : np-dimensional array
        n1 x n1 x n3 x ... x np tensor
    
    S : np-dimensional array
        n1 x n2 x n3 x ... x np tensor
            
    V : np-dimensional array (usually written transposed V^{T}, similar to the SVD)
        n2 x n2 x n3 x ... x np tensor
    
    References
    ----------
    [1] C.D. Martin, R. Shafer, and B. Larue, "An Order-$p$ Tensor Factorization with Applications in Imaging"
    SIAM Journal on Scientific Computing, vol. 35, n. 1,  pp. 474-490, 2013. DOI: 10.1137/110841229
    
    [2] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    """
    
    if frontal_faces is None:
        frontal_faces = get_frontal_faces(A)
    
    dim_A = A.shape
    n_1 = dim_A[0]
    n_2 = dim_A[1]
    
    dim_U = list(dim_A)
    dim_U[1] = n_1
    
    dim_S = dim_A
    
    dim_V = list(dim_A)
    dim_V[0] = n_2

    U = np.zeros(tuple(dim_U),dtype=complex)
    S = np.zeros(tuple(dim_S),dtype=complex)
    V = np.zeros(tuple(dim_V),dtype=complex)
    
    dims = len(dim_A)
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
        
    for index in frontal_faces:
        
        i_A = tuple([slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index])
        
        i_U = tuple([slice(0, dim_U[0]),slice(0, dim_U[1])] + [i for i in index])
        i_S = tuple([slice(0, dim_S[0]),slice(0, dim_S[1])] + [i for i in index])
        i_V = tuple([slice(0, dim_V[0]),slice(0, dim_V[1])] + [i for i in index])
        
        u, s, v = np.linalg.svd(A[i_A])
        
        U[i_U] = u
        V[i_V] = v
        
        S_empty = np.zeros((n_1,n_2), dtype=complex)
        np.fill_diagonal(S_empty, s)
        S[i_S] =  S_empty               
        
    for i in range(dims - 2):
        U = np.fft.ifft(U, axis = i + 2)
        S = np.fft.ifft(S, axis = i + 2)
        V = np.fft.ifft(V, axis = i + 2)
        
    return(U,S,V)