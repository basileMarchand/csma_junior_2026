import numpy as np 

def apply_dirichlet_boundary_conditions(K, f, fixed_dofs):
    n = K.shape[0]

    m = np.zeros((len(fixed_dofs), n))
    b = np.zeros((len(fixed_dofs),1))
    for i, dof in enumerate(fixed_dofs):
        m[i, dof] = 1.
 
    constrained_K = np.empty((n+m.shape[0], n+m.shape[0]))
    constrained_K[:n,:n] = K
    constrained_K[n:,:n] = m
    constrained_K[:n, n:] = m.T 
    constrained_K[n:,n:] = np.zeros((m.shape[0], m.shape[0]))
  
    constrained_F = np.empty((n+m.shape[0], 1))
    constrained_F[:n,0] = f 
    constrained_F[n:,:] = np.zeros((m.shape[0], 1))
    return constrained_K, constrained_F

def apply_periodic_conditions(K, f, fixed_dofs):
    n = K.shape[0]

    m = np.zeros((len(fixed_dofs), n))
    b = np.zeros((len(fixed_dofs),1))
    for i, (dof1, dof2) in enumerate(fixed_dofs):
        m[i, dof1] = 1.
        m[i, dof2] = -1.
 
    constrained_K = np.empty((n+m.shape[0], n+m.shape[0]))
    constrained_K[:n,:n] = K
    constrained_K[n:,:n] = m
    constrained_K[:n, n:] = m.T 
    constrained_K[n:,n:] = np.zeros((m.shape[0], m.shape[0]))
  
    constrained_F = np.empty((n+m.shape[0], 1))
    constrained_F[:n,0] = f 
    constrained_F[n:,:] = np.zeros((m.shape[0], 1))
    return constrained_K, constrained_F
