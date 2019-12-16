import numpy as np

def soft_threshold(x, reg_l_1):
    
    x = np.where(((x < reg_l_1) & (x > - reg_l_1)), 0, x)
    x = np.where(x > reg_l_1, x - reg_l_1, x)
    x = np.where(x < - reg_l_1, x + reg_l_1, x)
    
    return(x)
    