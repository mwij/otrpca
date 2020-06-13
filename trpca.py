import numpy as np
from t_tools import *

def trpca(M, reg=None, frontal_faces=None, 
          rho=1.1, mu=1e-3, mu_max=1e10, eps=1e-8):

    """Tensor robust principal component analysis.
    
    Decomposition of a tensor M = L + C where L has low tensor tubal 
    rank and C is a sparse noise tnesor. Note this is algorithm 4 in [1] 
    and matlab implementation can be found here: https://github.com/
    canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA.
    
    Parameters
    ----------
    M : np-dimensional array, observed tensor
        n1 x n2 x n3 x ... x np tensor
        
    reg : float, regularisation parameter on the 1-norm that is 
          applied to the noise tensor (lamda in text) calculated below
          defaults to 1 / sqrt(max(n1,n2) * n3 * ... np) if none

    frontal_faces : indices of frontal faces to iterate over
                    list of tuples
           
    mu : float, calibrates the penalty for the sum of the low-rank 
         and the sparse components differeing from the observation
         defaults to 1e-3 
    
    mu_max : float, upper bound for mu
             defaults to 1e10
             
    rho : float, grow mu by this each iteration (until upper bound)
          defaults to 1.1
             
    eps : float, converged when maximum difference between succesecive 
          tensors (using infinity norm) is less than this
          defaults to 1e-8
       
    Returns
    ----------
    L : np-dimensional array, uncorrupted tensor
        n1 x n1 x n3 x ... x np tensor
    
    C : np-dimensional array, noise tensor
        n1 x n2 x n3 x ... x np tensor

    References
    ----------
    [1] C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin, and S. Yan, 
    "Tensor Robust Principal Component Analysis with A New Tensor 
    Nuclear Norm", 2018, arXiv:1804.03728.
    """
    
    if frontal_faces is None:
        frontal_faces = get_frontal_faces(M)

    dim = M.shape
    
    if reg is None:
        n_min = min(dim[0], dim[1])
        reg = np.sqrt(n_min / np.prod(dim))
    
    #Initialise values.
    L = np.zeros(dim)
    C = np.zeros(dim)
    Y = np.zeros(dim)
    L_old = L
    C_old = C
    
    error = 1
    count = 0
    
    while(error > eps):
        
        #Update values.
        L = t_SVT(M - C - Y / mu, 1 / mu, frontal_faces)
        C = soft_threshold(M - L - Y / mu, reg / mu)
        Y = Y + mu * (L + C - M)     
        mu = min(rho * mu, mu_max)
        
        #Update error.
        L_error = np.max(np.abs(L - L_old))
        C_error = np.max(np.abs(C - C_old))
        M_error = np.max(np.abs(L + C - M))
        error = max(L_error, C_error, M_error)
        
        L_old = L
        C_old = C

    return(L, C)

def t_SVT(A, tau, frontal_faces = None):

    """Tensor singluar value threshold. This is algorithm 3 in [1].
    
    Parameters
    ----------
    A : np-dimensional array, tensor being singular values thresholded
        n1 x n2 x n3 x ... x np tensor

    tau : float, threshold for singular values
          reg / mu, where reg and mu are defined in trpca algorithm

    frontal_faces : indices of frontal faces to iterate over
                    list of tuples
       
    Returns
    ----------
    W : np-dimensional array, tensor with singular values thresholded
        n1 x n1 x n3 x ... x np tensor

    References
    ----------
    [1] C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin, and S. Yan, 
    "Tensor Robust Principal Component Analysis with A New Tensor 
    Nuclear Norm", 2018, arXiv:1804.03728.
    """
    
    if frontal_faces is None:
        frontal_faces = set([i[2:] for i, b in np.ndenumerate(A)])
    
    dim_A = A.shape
    
    n_1 = dim_A[0]
    n_2 = dim_A[1]
     
    W = np.zeros(tuple(dim_A), dtype=complex)

    dims = len(dim_A)
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
        
    for index in frontal_faces:
        
        #Frontal face having its singular values thresholded.
        i = tuple( [slice(0, dim_A[0]), slice(0, dim_A[1])] + [i for i in index])
        u, s, v = np.linalg.svd(A[i])
        
        #Old singular values.
        s_new = np.zeros((n_1,n_2), dtype=complex)
        np.fill_diagonal(s_new, s)
        
        #Calculate new singular values then multiply.
        w = u @ np.where(s_new - tau < 0, 0, s_new - tau) @ v
        
        W[i] = w
        
    for i in range(dims - 2):
        W = np.fft.ifft(W, axis = i + 2)

    return(W)