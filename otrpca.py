import numpy as np
from t_tools import *

def otrpca(M, rank=1, initial_A='t_SVD', iterations=1, 
           reg_l_1=None, reg_nuc=1, reg_c=100, 
           bc_conv=5, bc_error=1e-8, frontal_faces=None):

    """Online Tensor Robust Principal Component Analysis.
    
    Decomposition of a tensor M = L + C where L has low tensor tubal rank 
    and C is a sparse noise tnesor. Note this is algorithm 1 in [1] and a
    two-dimensional version is presented in [2]. In most cases, L is a 
    sequence of n1 x n3 x ... x np tensors that are stacked along the
    second (index 1) axis n2 times.
    
    Parameters
    ----------
    M : np-dimensional array, observed tensor
        n1 x n2 x n3 x ... x np tensor

    rank : integer greater than 0 and less than M[1] (r in Algorithm 1 
    of [1]) 
    defaults to 1

    intial_A: np-dimensional array, starting value for A that is either 
              supplied or calculated below (this A_0 in Algorithm 1 of [1])
              n1 x rank x n3 x ... x np tensor

    iterations: the number of times that the minimum of each slice will 
                be calculated
                defaults to 1 

    reg_l_1 : float, regularisation parameter on the 1-norm that is 
              applied to the noise tensor (lamda_1 in Algorithm 1 of [1]) calculated below if None
              defaults to 1 / sqrt(max(n1,n2) * n3 * ... np)

    reg_nuc : float, regularisation parameter on the frobenius norm that
              is applied to the low rank tensor (lamda_2 in Algorithm 1 of [1])
              defaults to 1

    reg_c : float, regularistion on the frobenius norm that is minimised 
            so that M = L + C. So the larger it is, the larger the penalty 
            is for L + C differeing from M. In [1] this apears above 
            equation 4.01 [1] as lambda_3. In the text, this value is 
            divided out (so lambda_i -> lambda_i / lambda_3). This has 
            not been done in this implementation to improve compuational stability.
            defaults to 100

    bc_conv : integer, the number of times that the error in min_b_c 
              must be below bc_error for min_b_c to have converged
              defaults to 5

    bc_error : float, the target for the error between successive 
               iterations in min_b_c to be below
               defaults to 1e-8

    frontal_faces : indices of frontal faces to iterate over
                    list of tuples
       
    Returns
    ---------- 
    L : np-dimensional array, uncorrupted tensor
        n1 x n1 x n3 x ... x np tensor
    
    C : np-dimensional array, noise tensor
        n1 x n2 x n3 x ... x np tensor

    References
    ----------
    [1] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    
    [2] Z. Zhang, D. Liu, S. Aeron, and A. Vetro, "An online tensor 
    robust PCA algorithm for sequential 2D data" 2016 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016.
    DOI: 10.1109/icassp.2016.7472114
    """
    
    dim = list(M.shape)

    #Make sure rank is not larger than its max possible value.
    if rank > dim[1]:
        rank = dim[1]

    #Initialise values.
    if initial_A == 't_SVD':
        U, S, V = t_SVD(M)
        A = U[:, 0:rank, :, :]

    elif initial_A == 'random_normal':
        dim_A = list((dim[0], rank)) + dim[2:]
        A = np.random.normal(0, 1, dim_A) 

    else:
        A = initial_A
    
    if reg_l_1 is None:
        n_min = min(dim[0], dim[1])
        reg_l_1 = np.sqrt( n_min / np.prod(dim))
        
    if frontal_faces is None:
        frontal_faces = get_frontal_faces(M)
        
    b = np.zeros(list((rank, 1)) + dim[2:], dtype=complex)  
    Y = np.zeros(list((rank, rank)) + dim[2:], dtype=complex)
    X = np.zeros(list((dim[0], rank)) + dim[2:], dtype=complex)
    B = np.zeros([rank] + dim[1:], dtype=complex)
    C = np.zeros(dim, dtype=complex)
    
    M_slice = [slice(0, dim[i]) for i in range(len(dim))]
    B_slice = [slice(0, rank)] + M_slice[1:]
    
    while iterations > 0:
        
        for i in range(dim[1]):
            
            #Keep track of progress.
            print('iterations remaining: ', iterations - 1,
                  'frames remaining: ', dim[1] - i - 1, 
                  end="\r")
            
            #Select slice to work on.
            M_slice[1] = slice(i, i + 1)
            m = M[tuple(M_slice)]

            #Minimise with respect to b and c.
            b, c = min_b_c(m, A, b, reg_c, reg_l_1, reg_nuc, 
                frontal_faces, bc_conv, bc_error)
            
            #Store b and c.
            C[tuple(M_slice)] = c
            B_slice[1] = slice(i, i + 1)
            B[tuple(B_slice)] = b

            #Update Xs and Ys.
            Y = Y + reg_c * t_prod(b, 
                                   t_transpose(b, 
                                   frontal_faces), frontal_faces)
            X = X + reg_c * t_prod(m - c, 
                                   t_transpose(b, 
                                   frontal_faces), frontal_faces)
               
            #Minimise with respect to A.
            A = min_A(A, X, Y + reg_nuc * identity(Y.shape), frontal_faces)

        iterations = iterations - 1 
            
    #Compute final estimate of L.
    L = t_prod(A, B)
        
    return(L, C)

def min_b_c(m, A, b, reg_c, reg_nuc, reg_l_1, frontal_faces, 
            bc_conv=5, bc_error=1e-8):

    """Finds the b and c that minimise equation 4.2.6 in [1] (this is 
    algorithm 2 in [1]).
    
    Parameters
    ----------
    m : np-dimensional array, slice t of M (fixing second axis, indexed 1) 
        (m_t in algorithm 1 of [1])
        n1 x 1 x n3 x ... x np tensor

    A : np-dimensional array, A_{t - 1} in algorithm 1 of [1] 
        n1 x rank x n3 x ... x np tensor

    b : np-dimensional array, prior b (b_{t - 1} in algorithm 1 of [1])
          rank x n2 x n3 x ... x np tensor

    reg_c, reg_nuc, reg_l_1, frontal_faces, bc_conv, bc_error : As in otrpca algorithm above

    Returns
    ----------
    b : np-dimensional array, updated b (b in algorithm 1 of [1])
          rank x n2 x n3 x ... x np tensor

    c : np-dimensional array, noise on m (c_t in algorithm 1 of [1])
        n1 x 1 x n3 x ... x np tensor

    References
    ----------
    [1] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    
    [2] Z. Zhang, D. Liu, S. Aeron, and A. Vetro, "An online tensor 
    robust PCA algorithm for sequential 2D data" 2016 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016.
    DOI: 10.1109/icassp.2016.7472114
    """
    
    dim = m.shape
     
    #Calculate A tilde.
    A_t_A_shape = [A.shape[1]] + list(A.shape[1:])
    A_1 = reg_c * t_prod(t_transpose(A, frontal_faces), A, frontal_faces) 
    + reg_nuc * identity(A_t_A_shape)
    A_1_inv = t_invert(A_1, frontal_faces)
    A_2 = reg_c * t_transpose(A, frontal_faces)
    A_tilde = t_prod(A_1_inv, A_2, frontal_faces)

    #Initialise values.
    c = np.zeros(tuple(dim), dtype=complex)
    c_minus_one = c
    b_minus_one = b
    error = 1
    error_minus_one = 1
    small_errors = 0
    
    #Converged once this concludes.
    while(small_errors < bc_conv):
        
        #Update c and b.
        c_mult = m - t_prod(A, b, frontal_faces)
        c = soft_threshold(np.real(c_mult), reg_l_1 / reg_c)
        b = t_prod(A_tilde, m - c, frontal_faces)
        
        #Update errors.
        error_minus_one = error
        error = max(np.linalg.norm(c - c_minus_one), 
                    np.linalg.norm(b - b_minus_one))
        
        #While loop concludes when the error is below the given
        #value bc_conv times.
        if(error < bc_error):
            small_errors = small_errors + 1

        else:
            small_errors = 0
        
        #Store the old c and b.
        c_minus_one = c
        b_minus_one = b
        
    return(b, c)

def min_A(A, X, Y, frontal_faces):

    """Finds the A that minimises equation 4.2.6 in [1] (this is 
    algorithm 3 in [1]).
    
    Parameters
    ----------
    A : np-dimensional array, prior A (A_{t - 1} in algorithm 1 of [1])
        n1 x rank x n3 x ... x np tensor

    X : np-dimensional array, X_t in algorithm 1 of [1]
        n1 x rank x n3 x ... x np tensor
 
    Y : np-dimensional array, Y_t in algorithm 1 of [1] 
        n2 x n2 x n3 x ... x np tensor

    frontal_faces : indices of frontal faces to iterate over
                    list of tuples

    Returns
    ----------
    A : np-dimensional array, updated A (A_{t} in algorithm 1 of [1])
        n1 x n2 x n3 x ... x np tensor

    References
    ----------
    [1] M. Wijnen, "Online Tensor Robust Principle Component Analysis"
    ANU Open Access Theses, 2018. DOI: 10.25911/5d889f8814c25
    
    [2] Z. Zhang, D. Liu, S. Aeron, and A. Vetro, "An online tensor 
    robust PCA algorithm for sequential 2D data" 2016 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016
    DOI: 10.1109/icassp.2016.7472114
    """
    dim = X.shape

    #Get tensors in list diagonlised (in fourier domain) with indices.
    Y_f = tdiag_list(Y, frontal_faces)
    X_f = tdiag_list(X, frontal_faces)
    A_f = tdiag_list(A, frontal_faces)

    for j in range(dim[1]):
        
        for k in range(len(A_f)):
            
            X_f_i, index = X_f[k][0], X_f[k][1]
            Y_f_i, index = Y_f[k][0], Y_f[k][1]
            A_f_i, index = A_f[k][0], A_f[k][1]
            
            #Calculate new value.
            new_val = (X_f_i[:,j] - A_f_i @ Y_f_i[:,j]) / Y_f_i[j,j]
            A_f_i[:,j] = new_val + A_f_i[:,j]
            
            #Store updated value.
            A_f[k] = tuple([A_f_i, index])

    #Back from list in fourier domain to tensor.
    A = tdiag_list_inverse(A_f, dim)

    return(A)
