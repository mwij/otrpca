import numpy as np

from get_frontal_faces import get_frontal_faces


def t_prod(A, B, frontal_faces = None):
    """Tensor-tensor product 
    
    Product of two p-dimensional tensors, C = A*B, as defined in [1] and reproduced
    in definition 2.4 in [2].

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
    
    dim_C = np.array(dim_A)
    dim_C[1] = dim_B[1]
    
    C = np.zeros(tuple(dim_C),dtype=complex)
    
    dims = len(dim_A)
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
        B = np.fft.fft(B, axis = i + 2)
           
    for index in frontal_faces:         
        i_A = tuple([slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index])
        i_B = tuple([slice(0, dim_B[0]),slice(0, dim_B[1])] + [i for i in index])
        i_C = tuple([slice(0, dim_C[0]),slice(0, dim_C[1])] + [i for i in index])
        
        C[i_C] = A[i_A] @ B[i_B]
    
    for i in range(dims - 2):
        C = np.fft.ifft(C, axis = i + 2)

    return(C)