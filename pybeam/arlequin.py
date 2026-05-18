import numpy as np 
import pybeam.elements as elem

def get_dofs(mesh, elem_id):
    return [mesh["dofs"][node_id] for node_id in mesh["elements"][elem_id]]


def compute_coupling(mesh_multiplier, mesh_model, param_h1=0.):
    coupling= np.zeros((mesh_multiplier["nb_nodes"], mesh_model["nb_nodes"]))
    for i, coords in enumerate(mesh_multiplier["elements"]):
        dofs = get_dofs(mesh_multiplier, i)
        coords = mesh_multiplier["coords"][mesh_multiplier["elements"][i]]
        gauss_points, gauss_weights = elem.INTEG_RULES["beam_2"]
        dx = mesh_multiplier["dx"][i]
        for wi, xi in zip(gauss_weights, gauss_points):
            ## Lagrange multiplier shape and grad functions 
            dN_dx = elem.DSHAPE_FUNCTION["beam_linear"](xi)
            N = elem.SHAPE_FUNCTION["beam_linear"](xi)

            ## Similar for the other mesh 
            # 0. Compute ip in real space 
            xi_real = N@coords.reshape((-1,1))
            # 1. find element in the model mesh containing the current ip projection 
            elem_rk = None
            for i, elem_other in enumerate(mesh_model['elements']):
                coords_other = mesh_model["coords"][elem_other]
                if xi_real >= coords_other[0] and xi_real <= coords_other[1]:
                    elem_rk = i
                    break
            # 2. Compute ip position in ref element 
            if elem_rk is None:
                raise Exception("failed to identify element")
            coords_other = mesh_model["coords"][mesh_model["elements"][elem_rk]]
            dx_other = ( coords_other[1] - coords_other[0])
            ip_in_ref = 2.*(xi_real - coords_other[0])/dx_other - 1. 
            # 3. evaluation shape and grad for this ip 
            dN_dx_u = elem.DSHAPE_FUNCTION["beam_linear"](ip_in_ref)
            N_u = elem.SHAPE_FUNCTION["beam_linear"](ip_in_ref)

            # 4. find impacted dofs in model mesh 
            dofs_u = get_dofs(mesh_model, elem_rk)
            coupling[np.ix_(dofs, dofs_u)] += wi * (N.reshape((-1,1))*N_u.reshape((1,-1)) + param_h1 *(4/(dx*dx_other))*dN_dx.reshape((-1,1))@dN_dx_u.reshape((1,-1)))*dx/2
    return coupling


def build_full_system(lhs1, lhs2, rhs1, rhs2, coupling1, coupling2):

    n_dof_1 = coupling1.shape[1]
    n_dof_2 = coupling2.shape[1]
    nc_dof_1 = lhs1.shape[0]
    nc_dof_2 = lhs2.shape[0]

    full_lhs = np.zeros((lhs1.shape[0] + lhs2.shape[0] + coupling1.shape[0], lhs1.shape[1] + lhs2.shape[1] + coupling1.shape[0]))
    full_rhs = np.zeros((lhs1.shape[0] + lhs2.shape[0] + coupling1.shape[0], 1))

    full_lhs[0:nc_dof_1,0:nc_dof_1] = lhs1
    full_lhs[nc_dof_1:nc_dof_1+nc_dof_2,nc_dof_1:nc_dof_1+nc_dof_2] = lhs2
    start_row = nc_dof_1+nc_dof_2
    full_lhs[start_row:,0:n_dof_1] = coupling1
    full_lhs[start_row:,nc_dof_1:nc_dof_1+n_dof_2] = -coupling2
    
    full_lhs[0:n_dof_1,nc_dof_1+nc_dof_2:] = coupling1.T
    full_lhs[nc_dof_1:nc_dof_1+n_dof_2,nc_dof_1+nc_dof_2:] = -coupling2.T


    full_rhs[:nc_dof_1, :] = rhs1
    full_rhs[nc_dof_1:nc_dof_1+nc_dof_2, :] = rhs2
    return full_lhs, full_rhs

