import numpy as np

from get_frontal_faces import get_frontal_faces
from soft_threshold import soft_threshold


def trpca(M, reg = None, frontal_faces = None):
    """Tensor robust principal component analysis 
    
    Decomposition of a tensor M = L + C where L has low tensor tubal rank 
    and C is a sparse noise tnesor. Note this is algorithm 4 in [1] and a
    matlab implementation can be found here: 
    https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA.
    
    Parameters
    ----------
    M : np-dimensional array, observed tensor
        n1 x n2 x n3 x ... x np tensor
       
    Returns
    ----------       
    L : np-dimensional array, uncorrupted tensor
        n1 x n1 x n3 x ... x np tensor
    
    C : np-dimensional array, noise tensor
        n1 x n2 x n3 x ... x np tensor

    References
    ----------
    [1] C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin, and S. Yan, 
    "Tensor Robust Principal Component Analysis with A New Tensor Nuclear Norm",
    2018, arXiv:1804.03728.
    """
    
    if frontal_faces is None:
        frontal_faces = get_frontal_faces(M)
    
    if reg is None:
        n_min = min(dim[0], dim[1])
        reg = np.sqrt( n_min / np.prod(dim))
    
    dim = M.shape
    
    L = np.zeros(dim)
    C = np.zeros(dim)
    Y = np.zeros(dim)
    
    L_old = L
    C_old = C
    
    rho    = 1.1
    mu     = 1e-3 
    mu_max = 1e10
    eps    = 1e-8
    
    error = 1
    count = 0
    
    while(error > eps):
        
        L = min_L(M, Y, C, mu, frontal_faces)
        C = min_C(M, Y, L, mu, reg)
        Y = Y + mu*(L + C - M)
        
        mu = min(rho*mu, mu_max)
        
        L_error = np.max(np.abs(L - L_old))
        C_error = np.max(np.abs(C - C_old))
        M_error = np.max(np.abs(L + C - M))
        
        error = max(L_error, C_error, M_error)
        
        L_old = L
        C_old = C
                            
    return(L, C)



def min_C(M, Y, L, mu, reg):
    
    A = M - L - Y / mu
    v = reg / mu
    
    return(soft_threshold(A,v))


def min_L(M, Y, C, mu, frontal_faces):
    
    A = M - C - Y / mu
    L = t_SVT(A, 1/mu, frontal_faces)
    
    return(L)

def t_SVT(A, tau, frontal_faces = None):
    """Tensor singluar value threshold. This is algorithm 3 in [1] 

    References
    ----------
    [1] C. Lu, J. Feng, Y. Chen, W. Liu, Z. Lin, and S. Yan, 
    "Tensor Robust Principal Component Analysis with A New Tensor Nuclear Norm",
    2018, arXiv:1804.03728.
    """
    if frontal_faces is None:
        print('waiting faces SVD')
        frontal_faces = set([i[2:] for i, b in np.ndenumerate(A)])
    
    dim_A = A.shape
    
    n_1 = dim_A[0]
    n_2 = dim_A[1]
     
    W = np.zeros(tuple(dim_A),dtype=complex)

    dims = len(dim_A)
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
        
    for index in frontal_faces:
        
        i = tuple( [slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index])
                
        u, s, v = np.linalg.svd(A[i])
        
        s_new = np.zeros((n_1,n_2), dtype=complex)
        np.fill_diagonal(s_new, s)
        
        w = u @ np.where(s_new - tau < 0, 0, s_new - tau) @ v
        
        W[i] = w
        
    for i in range(dims - 2):
        W = np.fft.ifft(W, axis = i + 2)

    return(W)