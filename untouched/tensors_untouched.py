#test

import copy
import numpy as np
import psutil
import math
import psutil

def get_frontal_faces(A):
    
    dims = A.shape
    a_frontal_slice_index = tuple( [0,0] + [slice(0, dims[i + 2]) for i in range(len(dims) - 2)]  )
    a_frontal_slice = A[a_frontal_slice_index]

    frontal_faces = set([i for i, b in np.ndenumerate(a_frontal_slice)])
    
    return(frontal_faces)
    

def t_prod(A,B, frontal_faces = None):
    
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
            

    for index in frontal_faces: #paralelise this bit?
        
        
        i_A = tuple( [slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index])
        i_B = tuple( [slice(0, dim_B[0]),slice(0, dim_B[1])] + [i for i in index])
        i_C = tuple( [slice(0, dim_C[0]),slice(0, dim_C[1])] + [i for i in index])

        
        C[i_C] = A[i_A] @ B[i_B]

    
    for i in range(dims - 2):
        
        C = np.fft.ifft(C, axis = i + 2)

    return(C)


def tdiag_list(A, frontal_faces = None):
    
    if frontal_faces is None:
        print('waiting faces tdiag_list')
        frontal_faces = get_frontal_faces(A)
        
    dim_A = A.shape
    dims = len(dim_A)
    
    storage = []
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
    
    for index in frontal_faces:
        i_A = [slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index]
        storage = storage + [tuple([A[tuple(i_A)],index])]

    return(storage)
        


def tdiag_list_inverse(A_f, dim):
    
    A = np.zeros(dim, dtype = complex)
    
    dims = len(dim)
    
    for face, index in A_f:
        i_A = [slice(0, dim[0]),slice(0, dim[1])] + [i for i in index]
        A[tuple(i_A)] = face
        
    for i in range(dims - 2):
        A = np.fft.ifft(A, axis = i + 2)
        
    return(A)
        
    


def t_SVD(A, frontal_faces = None):
    
    if frontal_faces is None:
        #print('waiting faces SVD')
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
        
        i_A = tuple( [slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index])
        
        i_U = tuple( [slice(0, dim_U[0]),slice(0, dim_U[1])] + [i for i in index])
        i_S = tuple( [slice(0, dim_S[0]),slice(0, dim_S[1])] + [i for i in index])
        i_V = tuple( [slice(0, dim_V[0]),slice(0, dim_V[1])] + [i for i in index])
        
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



def t_transpose(A, frontal_faces = None):
    
    if frontal_faces is None:
        #print('waiting faces')
        frontal_faces = get_frontal_faces(A)
    
    dim_A = A.shape
    dim_A_T = list(dim_A)
    dim_A_T[0] = dim_A[1]
    dim_A_T[1] = dim_A[0]
    
    A_T = np.zeros(tuple(dim_A_T),dtype=complex)
    
    dims = len(dim_A)
            
    for index in frontal_faces:
        
        
        i_A = tuple( [slice(0, dim_A[0]),slice(0, dim_A[1])] + [index[i] for i in range(len(index))])
        i_A_T = tuple( [slice(0, dim_A_T[0]),slice(0, dim_A_T[1])] + [(-index[i]%(dim_A[i + 2] )) for i in range(len(index))])

        A_T[i_A_T] = A[i_A].T

    return(A_T)




def identity(dim):
    if(dim[0] != dim[1]):
        print('error')
        return('error')
    
    I = np.zeros(dim)
    
    zeros = list(np.zeros(len(dim) - 2, dtype = int))
    

    
    for i in range(dim[0]):
        index = [i,i] + zeros

        I[tuple(index)] = 1
        
    return(I)
        





def min_b(A_1, A_2, frontal_faces):
    
    
    dim = A_2.shape
    
    A_1_f = tdiag_list(A_1, frontal_faces)
    A_2_f = tdiag_list(A_2, frontal_faces)
    storage = []
    
    for i in range(len(A_1_f)):
        A_1_f_face, index  = A_1_f[i][0], A_1_f[i][1]
        A_2_f_face, index  = A_2_f[i][0], A_2_f[i][1]
        
        temp = np.zeros(A_2_f_face.shape, dtype = complex)
        
        for col in range(temp.shape[1]):
            temp[:,col] = np.linalg.solve(A_1_f_face, A_2_f_face[:,col])
        
        storage = storage + [tuple([temp, index])]
        
    storage = tdiag_list_inverse(storage, dim)
    
    return(storage)
            
    





def t_solve(A_1, A_2, frontal_faces ):
    #solves for the tensor B st  A_1 * B = A_2
    
    dim = A_2.shape
    
    A_1_f = tdiag_list(A_1, frontal_faces)
    A_2_f = tdiag_list(A_2, frontal_faces)
    storage = []
    
    for i in range(len(A_1_f)):
        A_1_f_face, index  = A_1_f[i][0], A_1_f[i][1]
        A_2_f_face, index  = A_2_f[i][0], A_2_f[i][1]
        
        temp = np.zeros(A_2_f_face.shape, dtype = complex)
        
        for col in range(temp.shape[1]):
            temp[:,col] = np.linalg.solve(A_1_f_face, A_2_f_face[:,col])
        
        storage = storage + [tuple([temp, index])]
        
    storage = tdiag_list_inverse(storage, dim)
    
    return(storage)





def min_A(A, X,Y,l, frontal_faces):
    
    
    dim = X.shape
    
    Y = Y + l * identity(Y.shape)
    Y_f = tdiag_list(Y, frontal_faces)
    X_f = tdiag_list(X, frontal_faces)
    
    A = np.zeros(dim, dtype = complex)
    A_f = tdiag_list(A, frontal_faces)
    
    for i in range(len(A_f)):
        
        X_f_i, index = X_f[i][0], X_f[i][1]
        Y_f_i, index = Y_f[i][0], Y_f[i][1]
        
        new_A = np.zeros((dim[0],dim[1]), dtype = complex)

        for A_row in range(new_A.shape[0]):
            new_A[A_row,:] = (np.linalg.solve(Y_f_i.T, (X_f_i[A_row,:]).T)).T
            
        A_f[i] = tuple([new_A, index])
    
    A = tdiag_list_inverse(A_f, dim)
    
    return(A)






def min_A_BD(A,X,Y, frontal_faces):
    
    
    dim = X.shape
    
    Y_f = tdiag_list(Y, frontal_faces)
    X_f = tdiag_list(X, frontal_faces)
    A_f = tdiag_list(A, frontal_faces)
    
    for j in range(dim[1]):  #r
        
        for k in range(len(A_f)): #n_1
            
            X_f_i, index = X_f[k][0], X_f[k][1]
            Y_f_i, index = Y_f[k][0], Y_f[k][1]
            A_f_i, index = A_f[k][0], A_f[k][1]
            
            Q = (X_f_i[:,j] - A_f_i @ Y_f_i[:,j]) / Y_f_i[j,j] + A_f_i[:,j]
            
            A_f_i[:,j] = Q#/max(np.linalg.norm(Q), 1)
                        
            A_f[k] = tuple([A_f_i, index])
    
    A = tdiag_list_inverse(A_f, dim)
    
    return(A)





def soft_threshold(x, reg_l_1):
    
    x = np.where(((x < reg_l_1) & (x > - reg_l_1)), 0, x)
    x = np.where(x > reg_l_1, x - reg_l_1, x)
    x = np.where(x < - reg_l_1, x + reg_l_1, x)
    
    return(x)






def min_b(A_1, A_2, frontal_faces):
    
    
    dim = A_2.shape
    
    A_1_f = tdiag_list(A_1, frontal_faces)
    A_2_f = tdiag_list(A_2, frontal_faces)
    storage = []
    
    for i in range(len(A_1_f)):
        A_1_f_face, index  = A_1_f[i][0], A_1_f[i][1]
        A_2_f_face, index  = A_2_f[i][0], A_2_f[i][1]
        
        temp = np.zeros(A_2_f_face.shape, dtype = complex)
        
        for col in range(temp.shape[1]):
            temp[:,col] = np.linalg.solve(A_1_f_face, A_2_f_face[:,col])
        
        storage = storage + [tuple([temp, index])]
        
    storage = tdiag_list_inverse(storage, dim)
    
    return(storage)
            
    





def t_invert(A, frontal_faces):
    A_f = tdiag_list(A, frontal_faces)
    storage = []
    for face, index in A_f:
        inverse = np.linalg.inv(face)
        storage = storage + [tuple([inverse, index])]
    
    return(tdiag_list_inverse(storage, A.shape))






def min_b_c(m, A, reg_c, reg_nuc, reg_l_1, frontal_faces, iteration, total_iterations, feed_forward, b_0 = None):
    
    dim = m.shape
    
    if b_0 is None:
        dim_b_t = list(dim)
        dim_b_t[0] = A.shape[1]
        b_0 = np.zeros(tuple(dim_b_t), dtype = complex)
        
    if (feed_forward == False):
        dim_b_t = list(dim)
        dim_b_t[0] = A.shape[1]
        b_0 = np.zeros(tuple(dim_b_t), dtype = complex)

    A_shape = list(A.shape)
    A_t_A_shape = A_shape
    A_t_A_shape[0] = A_t_A_shape[1]
    A_1 = reg_c * t_prod(t_transpose(A, frontal_faces), A, frontal_faces) + reg_nuc * identity(A_t_A_shape)
    A_1_inv = t_invert(A_1, frontal_faces)
    A_2 = reg_c*t_transpose(A, frontal_faces)
    
    A_tilde = t_prod(A_1_inv, A_2, frontal_faces)
    
    dim_A = A.shape
    dims = len(dim_A)
    
    c = np.zeros(tuple( dim), dtype = complex)
    b_t = b_0
    
    
    c_minus_one = c
    b_t_minus_one = b_t
    
    error = 8888
    error_minus_one = 0
    small_errors = 0
    count = 0
    

    while(small_errors < 5):
        
        #print('itrtion: ', iteration,'of',total_iterations, '| n: ', count,
         #     '| trip: ', small_errors,  '| error: ', error, '| D error: ', error_minus_one - error,
         #    'Ram', psutil.virtual_memory()[2],)
        
        c_mult = m - t_prod(A, b_t, frontal_faces)
        c =  soft_threshold(np.real(c_mult), reg_l_1/reg_c)
        b_t = t_prod(A_tilde, m - c, frontal_faces)
        #b_t = min_b(A_1, t_prod(A_2, m - c, frontal_faces), frontal_faces)
        
        error_minus_one = error
        
        error = max(np.linalg.norm(c - c_minus_one), np.linalg.norm(b_t - b_t_minus_one))
        
        if(error < 1e-8):
            small_errors = small_errors + 1
            
        else:
            small_errors = 0
            
        count = count + 1
        
        c_minus_one = c
        b_t_minus_one = b_t
        
    return(b_t, c)
        
        
    








def otrpca(M, reg_c = None, reg_nuc  = None, reg_l_1 = None, feed_forward = None, rank = None, A = None):
    
    dim = M.shape
 
    if reg_c is None:
        reg_c = 100
        
    if reg_nuc is None:
        reg_nuc = 1
    
    if reg_l_1 is None:
        n_min = min(dim[0], dim[1])
        reg_l_1 = np.sqrt( n_min / np.prod(dim))
    
    n_3_tilde = np.product(dim[2:])
    
    frontal_faces = get_frontal_faces(M)
    
    if feed_forward is None:
        feed_forward = False
        
    if rank is None:
        rank = 1
    
    if A is None:
        dim_A = list(dim)
        dim_A[1] = rank
        A = np.random.normal(0,1,(dim_A)) 
    
    dim_b = list(dim)
    dim_b[0] = rank
    dim_b[1] = 1
    b_t = np.zeros(dim_b, dtype = complex)  

    
    dim_Y = list(dim)
    dim_Y[0] = rank
    dim_Y[1] = rank
    
    Y = np.zeros(dim_Y, dtype = complex)
    
    
    dim_X = list(dim)
    dim_X[1] = rank
    
    X = np.zeros(dim_X, dtype = complex)
    
    dim_B = list(dim)
    dim_B[0] = rank
    B = np.zeros(dim_B, dtype = complex)
    
    C = np.zeros(M.shape, dtype = complex)
    
    all_slice = [slice(0, dim[i]) for i in range(len(dim))]
    M_slice  = copy.deepcopy(all_slice)
    B_slice  = copy.deepcopy(all_slice)
    B_slice[0] = slice(0,rank)
    
    
    for i in range(dim[1]):
        
        M_slice[1] = slice(i, i + 1)
        m = M[tuple( M_slice)]
        
        b_t, c = min_b_c(m, A, reg_c, reg_l_1, reg_nuc, frontal_faces, i, dim[1], feed_forward, b_t)
        
        C[tuple( M_slice)] = c
        B_slice[1] = slice(i, i + 1)
        B[tuple( B_slice)] = b_t
        
        
        
        Y = Y + reg_c * t_prod(b_t, t_transpose(b_t, frontal_faces), frontal_faces)
        X = X + reg_c * t_prod((m - c), t_transpose(b_t, frontal_faces), frontal_faces)
           
        
        print('minimising A')
        
        
        A = min_A_BD(A,X,Y + reg_nuc *identity(Y.shape) , frontal_faces) 
        
    return(A, B, C)
        
        
    

    



def min_C(M, Y, L, mu, reg):
    
    A = M - L - Y / mu
    v = reg / mu
    
    return(soft_threshold(A,v))





def t_SVT(A, tau, frontal_faces = None):
    
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





def trpca(M, reg):
    dim = M.shape
    
    if reg is None:
        n_min = min(dim[0], dim[1])
        reg = np.sqrt( n_min / np.prod(dim))
    
    frontal_faces = get_frontal_faces(M)
    
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
    
    print('start trpca')
    
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
        
        
        if(count % 10 == 0):
            print('cpu', psutil.cpu_percent(), 'ram', psutil.virtual_memory()[2], count, error)
        
        count = count +1
        
        
        

    
    return(L, C)



def min_L(M, Y, C, mu, frontal_faces):
    
    A = M - C - Y / mu
    

    
    L = t_SVT(A, 1/mu, frontal_faces)
    
    return(L)






def approx(A, rank, U, S, V, frontal_faces):
    
    dim_A = A.shape
    
    storage = np.zeros(dim_A)
    
    #U, S, V = t_SVD(A, frontal_faces)
    
    all_slices = [slice(0, dim_A[i]) for i in range(len(dim_A))]
    
    for i in range(rank):
        U_slice = all_slices[:]
        U_slice[1] = slice(i, i + 1)
        U_sliced = U[tuple(U_slice)]
        
        S_slice = all_slices[:]
        S_slice[0] = slice(i, i + 1)
        S_slice[1] = slice(i, i + 1)
        S_sliced = S[tuple(S_slice)]
        
        V_slice = all_slices[:]
        V_slice[0] = slice(i, i + 1)
        V_sliced = V[tuple(V_slice)]
        
        storage = storage + t_prod(t_prod(U_sliced, S_sliced, frontal_faces), V_sliced, frontal_faces)
        
    return(storage)

def all_ranks(A, U, S, V, frontal_faces):
    
    dim_A = A.shape
    
    storage = np.zeros(dim_A)
    
    all_slices = [slice(0, dim_A[i]) for i in range(len(dim_A))]
    
    to_return = []
    
    for i in range(min(dim_A[0], dim_A[1])):
        U_slice = all_slices[:]
        U_slice[1] = slice(i, i + 1)
        U_sliced = U[tuple(U_slice)]
        
        S_slice = all_slices[:]
        S_slice[0] = slice(i, i + 1)
        S_slice[1] = slice(i, i + 1)
        S_sliced = S[tuple(S_slice)]
        
        V_slice = all_slices[:]
        V_slice[0] = slice(i, i + 1)
        V_sliced = V[tuple(V_slice)]
        
        storage = storage + t_prod(t_prod(U_sliced, S_sliced, frontal_faces), V_sliced, frontal_faces)
        
        to_return = to_return + [storage]
        
    return(to_return)



def t_extract_diag(A,i):
    dim_A = A.shape
    
    all_slices = [slice(0, dim_A[i]) for i in range(len(dim_A))]
    to_slice = all_slices
    to_slice[0] = slice(i, i + 1)
    to_slice[1] = slice(i, i + 1)
    
    return(A[tuple(to_slice)])


def tnn(A):
    dims = np.array(A.shape)
    u, s, v = t_SVD(A)
    r_max = min(dims[0], dims[1])
    zero = dims * 0
    storage = 0

    for i in(range(r_max)):
        index = zero
        index[0:2] = i
        

        storage = storage + s[tuple(index)]
    
    return(storage)


def rank(A):
    u,s,v = t_SVD(A)
    
    rank = 0

    y = [np.linalg.norm(t_extract_diag(s,i)) for i in range(min(A.shape[0], A.shape[1]))]
    
    for sv in y:
        if sv > 1e-10 :
            rank = rank + 1
    
    return(rank)



def error(estimate, acutual):
    return(np.linalg.norm(estimate - acutual)/np.linalg.norm(acutual))



def analyser(L_0, C_0):
    
    M_0 = L_0 + C_0
    
    dims = M_0.shape
    
    n_min = min(dims[0], dims[1])
    reg = np.sqrt( n_min / np.prod(dims))
    
    L , C = trpca(M_0, reg)
    
    L_error = np.linalg.norm(L - L_0)/np.linalg.norm(L_0)
    C_error = np.linalg.norm(C - C_0)/np.linalg.norm(C_0)
    
    #print(L_error)
    #print(C_error)
    
    return(L_error, C_error)



def analyser_online(L_0, C_0):
    
    M_0 = L_0 + C_0
    
    dims = M_0.shape
    
    n_min = min(dims[0], dims[1])
    reg = np.sqrt( n_min / np.prod(dims))
    
    A, B, C = otrpca(M_0)
    
    L = t_prod(A,B)
    
    L_error = np.linalg.norm(L - L_0)/np.linalg.norm(L_0)
    C_error = np.linalg.norm(C - C_0)/np.linalg.norm(C_0)
    
    #print(L_error)
    #print(C_error)
    
    return(L_error, C_error)


def t_avg_rank(A, frontal_faces = None):
    
    if frontal_faces is None:
        #print('waiting faces SVD')
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
        
    ranks = []
        
    for index in frontal_faces:
        
        i_A = tuple( [slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index])
        
        ranks = ranks + [np.linalg.matrix_rank(A[i_A])]
    
    
    return(np.average(ranks))





def t_SVD_skinny(A, frontal_faces = None):
    
    r = rank(A)
    
    if frontal_faces is None:
        #print('waiting faces SVD')
        frontal_faces = get_frontal_faces(A)
    
    dim_A = A.shape
    
    n_1 = dim_A[0]
    n_2 = dim_A[1]
    
    dim_U = list(dim_A)
    dim_U[1] = r
    
    dim_S = list(dim_A)
    dim_S[0] = r
    dim_S[1] = r
    
    dim_V = list(dim_A)
    dim_V[0] = r
    
    
    U = np.zeros(tuple(dim_U),dtype=complex)
    S = np.zeros(tuple(dim_S),dtype=complex)
    V = np.zeros(tuple(dim_V),dtype=complex)
    
    dims = len(dim_A)
    
    for i in range(dims - 2):
        A = np.fft.fft(A, axis = i + 2)
        
    for index in frontal_faces:
        
        i_A = tuple( [slice(0, dim_A[0]),slice(0, dim_A[1])] + [i for i in index])
        
        i_U = tuple( [slice(0, dim_U[0]),slice(0, dim_U[1])] + [i for i in index])
        i_S = tuple( [slice(0, dim_S[0]),slice(0, dim_S[1])] + [i for i in index])
        i_V = tuple( [slice(0, dim_V[0]),slice(0, dim_V[1])] + [i for i in index])
        
        u, s, v = np.linalg.svd(A[i_A])
        
        U[i_U] = u[:,0:r]
        V[i_V] = v[0:r,:]
        
        S_empty = np.zeros((r,r), dtype=complex)
        np.fill_diagonal(S_empty, s[0:r])
        
        S[i_S] =  S_empty               
        

    for i in range(dims - 2):
        U = np.fft.ifft(U, axis = i + 2)
        S = np.fft.ifft(S, axis = i + 2)
        V = np.fft.ifft(V, axis = i + 2)
        
    return(U,S,V)


def mu_TIC(A):
    
    dims = np.array(A.shape)
    
    n_1_tilde = np.product(dims)
    
    n_1 = dims[0]
    n_2 = dims[1]
    n_3_tilde = n_1_tilde/(n_1 * n_2)
    
    U, S, V = t_SVD_skinny(A)
    
    r = rank(A)
    
    dim_E_U = np.copy(dims)
    dim_E_V = np.copy(dims)

        
    dim_E_U[0] = n_1
    dim_E_U[1] = 1
   
    dim_E_V[0] = n_2
    dim_E_V[1] = 1
    
    E_U = np.zeros(dim_E_U)
    E_V = np.zeros(dim_E_V)
    
    
    mu = 0
    
    for i in range(n_1):
        
        e_i = E_U
        
        e_i_index = [i] + [0 for i in range(len(dims) -1)]
        
        e_i[tuple(e_i_index)] = 1

            
        mu_candidate = (np.sum(np.absolute((t_prod(t_transpose(U), e_i)))))*n_1*n_3_tilde/r
        
        
        mu = max(mu, mu_candidate)

        E_U = np.zeros(dim_E_U)
        
    for i in range(n_2):
        
        e_i = E_V
        
        e_i_index = [i] + [0 for i in range(len(dims) -1)]
                
        e_i[tuple(e_i_index)] = 1
                        
        mu_candidate = (np.sum(np.absolute((t_prod(V, e_i)))))*n_2*n_3_tilde/r
        
        mu = max(mu, mu_candidate)

        E_V = np.zeros(dim_E_V)
        
    mu_candidate = ((((np.max(np.absolute(t_prod(U,V))))))**2)*n_1_tilde*n_3_tilde/r
    
    
    mu = max(mu, mu_candidate)
    
    return(mu)
    

