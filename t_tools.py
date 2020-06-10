import numpy as np

def soft_threshold(X, v):
    """Soft Threshold
    
        For each element of the array x, if that element is:
            greater than v, change it to itself minus v
            less than minus v, change it to itself plus v
            between minus v and v, change it to 0
           
    Parameters
    ----------
    X : np-dimensional array
        n1 x ... x np tensor
    
    v : float
        a positive real number

    Returns
    -------
     X : np-dimensional array
        n1 x ... x np tensor
    
    """
    
    X = np.where(((X < v) & (X > - v)), 0, X)
    X = np.where(X > v, X - v, X)
    X = np.where(X < - v, X + v, X)
    
    return(X)

def get_frontal_faces(A):
    """Indices to fix for tensor multiplication

       Provides a list of the indices that need to be fixed in each
       dimension (other than the first two) for the t_product

    Parameters
    ----------
    A : np-dimensional array
        n1 x j x n3 x ... x np tensor

    Returns
    -------
    frontal_faces : list
    the list of frontal faces


    Notes
    -----
    To implement this tensor product, which is recursively defined with
    a base case of matrix-matrix multiplication, this helper function
    provides a list of the indices that need to be fixed in each
    dimension (other than the first two) to produce the base-case matrices
    to be multiplied.
    
    This was previously within the t_prod function, but repeated evaluation 
    had a huge computational cost and was not neccery.
    
    Can probably be replaced with some simple permutations?
    def get_frontal_faces_2(A):
        return(set(itertools.product(*[list(range(dim)) for dim in A.shape[2:]])))
    """
    
    dims = A.shape
    a_frontal_slice_index = tuple([0,0] + [slice(0, dims[i + 2]) for i in range(len(dims) - 2)])
    a_frontal_slice = A[a_frontal_slice_index]

    frontal_faces = set([i for i, b in np.ndenumerate(a_frontal_slice)])
    
    return(frontal_faces)

def t_prod(A, B, frontal_faces=None):
    """Tensor-tensor product 
    
    Product of two p-dimensional tensors, C = A*B, as defined in [1] and reproduced
    in Definition 2.4 in [2].

    Parameters
    ----------
    A : np-dimensional array
        n1 x j x n3 x ... x np tensor
    B : np-dimensional array
        j x n2 x n3 x ... x np tensor
       
    Returns
    ----------       
    C : np-dimensional array
        n1 x n2 x n3 x ... x np tensor
          
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
    dim_B = B.shape
    dim_C = list((dim_A[0], dim_B[1])) +  list(dim_A[2:])
    dims = len(dim_A)
    
    C = np.zeros(tuple(dim_C), dtype = complex)
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
        B = np.fft.fft(B, axis = i + 2)
           
    for index in frontal_faces:         
        i_A = tuple([slice(0, dim_A[0]), slice(0, dim_A[1])] + [i for i in index])
        i_B = tuple([slice(0, dim_B[0]), slice(0, dim_B[1])] + [i for i in index])
        i_C = tuple([slice(0, dim_C[0]), slice(0, dim_C[1])] + [i for i in index])
        
        C[i_C] = A[i_A] @ B[i_B]
    
    for i in range(dims - 2):
        C = np.fft.ifft(C, axis = i + 2)

    return(C)

def t_SVD(A, frontal_faces=None):
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
    [1] C.D. Martin, R. Shafer, and B. Larue, "An Order-$p$ Tensor Factorization 
    with Applications in Imaging" SIAM Journal on Scientific Computing, 
    vol. 35, n. 1,  pp. 474-490, 2013. DOI: 10.1137/110841229
    
    [2] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    """
    
    if frontal_faces is None:
        frontal_faces = get_frontal_faces(A)
    
    dim_A = A.shape
    n_1 = dim_A[0]
    n_2 = dim_A[1]
    
    dim_U = list((n_1,n_1)) + list(dim_A[2:])
    dim_S = dim_A
    dim_V = list((n_2,n_2)) + list(dim_A[2:])
    dims = len(dim_A)

    U = np.zeros(tuple(dim_U), dtype = complex)
    S = np.zeros(tuple(dim_S), dtype = complex)
    V = np.zeros(tuple(dim_V), dtype = complex)
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
        
    for index in frontal_faces:
        
        i_A = tuple([slice(0, dim_A[0]), slice(0, dim_A[1])] + [i for i in index])
        i_U = tuple([slice(0, dim_U[0]), slice(0, dim_U[1])] + [i for i in index])
        i_S = tuple([slice(0, dim_S[0]), slice(0, dim_S[1])] + [i for i in index])
        i_V = tuple([slice(0, dim_V[0]), slice(0, dim_V[1])] + [i for i in index])
        
        u, s, v = np.linalg.svd(A[i_A])
        
        U[i_U] = u
        V[i_V] = v
        
        S_empty = np.zeros((n_1, n_2), dtype=complex)
        np.fill_diagonal(S_empty, s)
        S[i_S] = S_empty               
        
    for i in range(dims - 2):
        U = np.fft.ifft(U, axis = i + 2)
        S = np.fft.ifft(S, axis = i + 2)
        V = np.fft.ifft(V, axis = i + 2)
        
    return(U, S, V)

def t_transpose(A, frontal_faces = None):
    """Tensor Transpose
    
    Transposes the tensor A according to defition 3.9 of [1] and Definition 2.11 of [2]
    
    Parameters
    ----------
    A : np-dimensional array
        n1 x n2 x n3 x ... x np tensor
       
    Returns
    ----------       
    A : np-dimensional array
        n1 x n2 x n3 x ... x np tensor
    
    References
    ----------
    [1] C.D. Martin, R. Shafer, and B. Larue, "An Order-$p$ Tensor Factorization 
    with Applications in Imaging" SIAM Journal on Scientific Computing, 
    vol. 35, n. 1,  pp. 474-490, 2013. DOI: 10.1137/110841229
    
    [2] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    """
  
    if frontal_faces is None:
        frontal_faces = get_frontal_faces(A)
    
    dim_A = A.shape
    dim_A_T = list((dim_A[1], dim_A[0])) + list(dim_A[2:])
    dims = len(dim_A)
    A_T = np.zeros(tuple(dim_A_T), dtype = complex)
              
    for index in frontal_faces:
        
        i_A = tuple([slice(0, dim_A[0]), slice(0, dim_A[1])] + [index[i] for i in range(len(index))])
        i_A_T = tuple([slice(0, dim_A_T[0]), slice(0, dim_A_T[1])] + [(-index[i] % (dim_A[i + 2])) for i in range(len(index))])
        
        A_T[i_A_T] = A[i_A].T

    return(A_T)

def identity(dim):
    """Tensor Identity
    
    Returns a tensor of dimension dim that is the identity under the t_prod,
    as defined in defition 3.4 of [1] and Definition 2.8 of [2]
    
    Parameters
    ----------
    dim : 1-dimensional array of length p
       
    Returns
    ----------       
    A : np-dimensional array
        dim_1 x  ... x dim_p tensor
    
    References
    ----------
    [1] C.D. Martin, R. Shafer, and B. Larue, "An Order-$p$ Tensor Factorization 
    with Applications in Imaging" SIAM Journal on Scientific Computing, 
    vol. 35, n. 1,  pp. 474-490, 2013. DOI: 10.1137/110841229
    
    [2] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    """
    
    if(dim[0] != dim[1]):
        print('error, non-square inverse')
        return('error, non-square inverse')
    
    I = np.zeros(dim)
    zeros = list(np.zeros(len(dim) - 2, dtype = int)) # frontal face indices
   
    for i in range(dim[0]):
        index = [i,i] + zeros

        I[tuple(index)] = 1
        
    return(I)    

def t_invert(A, frontal_faces):
    """Tensor Invertor
    
    Returns the inverse tensor of under the t_prod,
    as defined in Definition 3.5 of [1] and Definition 2.9 of [2]
    
    Parameters
    ----------
    A : np-dimensional array
        n_1 x n_1 x ... x n_p tensor
       
    Returns
    ----------       
    A_inverse : np-dimensional array
        n_1 x n_1 x ... x n_p tensor that is the inverse of A
    
    References
    ----------
    [1] C.D. Martin, R. Shafer, and B. Larue, "An Order-$p$ Tensor Factorization 
    with Applications in Imaging" SIAM Journal on Scientific Computing, 
    vol. 35, n. 1,  pp. 474-490, 2013. DOI: 10.1137/110841229
    
    [2] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    """
    
    A_f = tdiag_list(A, frontal_faces)
    storage = []
    for face, index in A_f:
        inverse = np.linalg.inv(face)
        storage = storage + [tuple([inverse, index])]
    
    return(tdiag_list_inverse(storage, A.shape))


def tdiag_list(A, frontal_faces=None):
    """Returns a list of the frontal faces of 'diagonalised' (by fft) A and 
       their indices
        
    Parameters
    ----------
    A : np-dimensional array
        n_1 x n_1 x ... x n_p tensor
       
    Returns
    ----------       
    storage : list
        list of tuples containing all frontal faces (in fourier domain) and that
        face's index
    """

    if frontal_faces is None:
        print('waiting faces tdiag_list')
        frontal_faces = get_frontal_faces(A)
        
    dim_A = A.shape
    dims = len(dim_A)
    storage = []
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
    
    for index in frontal_faces:
        i_A = [slice(0, dim_A[0]), slice(0, dim_A[1])] + [i for i in index]
        storage = storage + [tuple([A[tuple(i_A)], index])]

    return(storage)

def tdiag_list_inverse(A_f, dim):
    """Inverts tdiag_list
        
    Parameters
    ----------
    A_f : list
        list of tuples containing all frontal faces (in fourier domain) and that 
        face's index of a tensor A

    Returns
    ----------       
    A : np-dimensional array
        n_1 x n_1 x ... x n_p tensor
    """

    A = np.zeros(dim, dtype = complex)
    dims = len(dim)
    
    for face, index in A_f:
        i_A = [slice(0, dim[0]), slice(0, dim[1])] + [i for i in index]
        A[tuple(i_A)] = face
        
    for i in range(dims - 2):
        A = np.fft.ifft(A, axis = i + 2)
        
    return(A)
        