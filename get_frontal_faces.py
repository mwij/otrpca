import numpy as np

def get_frontal_faces(A):
    """Indices to fix for multiplication

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