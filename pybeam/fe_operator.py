

import numpy as np
import pybeam.elements as elem

def get_dofs(mesh, elem_id):
    return [mesh["dofs"][node_id] for node_id in mesh["elements"][elem_id]]


def integrate(mesh, array_data_at_ip):
    result = 0.
    for i in range(mesh["nb_elements"]):
        gauss_points, gauss_weights = elem.INTEG_RULES["beam_2"]
        dx = mesh["dx"][i]
        for p, (wi, xi) in enumerate(zip(gauss_weights, gauss_points)):
            result += array_data_at_ip[i, p] * dx/2 * wi
    return result

def volumetric_load_vector(mesh, q_f, alpha_f=None):
    if alpha_f is None:
        alpha_f = lambda x: 1. 
    f = np.zeros((mesh["nb_nodes"]))
    for i in range(mesh["nb_elements"]):
        gauss_points, gauss_weights = elem.INTEG_RULES["beam_2"]
        dofs = get_dofs(mesh, i)
        load = np.zeros(len(dofs))
        coords = mesh["coords"][mesh["elements"][i]]
        dx = mesh["dx"][i]
        for wi, xi in zip(gauss_weights, gauss_points):
            # shape function evaluated at gauss point
            N = elem.SHAPE_FUNCTION["beam_linear"](xi)
            #volumetric load
            ip_coords = N@coords.reshape((-1,1))
            alpha = alpha_f( ip_coords )
            q = q_f( ip_coords )
            load += N * q * dx/2 * wi * alpha
            # Assemble the load vector
        f[dofs] += load   
    return f


def periodic_load_vector(mesh, E):
    f = np.zeros((mesh["nb_nodes"]))
    for i in range(mesh["nb_elements"]):
        gauss_points, gauss_weights = elem.INTEG_RULES["beam_2"]
        dofs = get_dofs(mesh, i)
        load = np.zeros(len(dofs))
        coords = mesh["coords"][mesh["elements"][i]]
        dx = mesh["dx"][i]
        for wi, xi in zip(gauss_weights, gauss_points):
            # shape function evaluated at gauss point
            N = elem.SHAPE_FUNCTION["beam_linear"](xi)
            B = elem.DSHAPE_FUNCTION["beam_linear"](xi)*2/dx
            #volumetric load
            ip_coords = N@coords.reshape((-1,1))

            load += B * E[i]*1. * dx/2 * wi
            # Assemble the load vector
        f[dofs] -= load   
    return f

def compute_stiffness_matrix(mesh, E, A=1., alpha_f=None):
    if alpha_f is None:
        alpha_f = lambda x: 1. 
    # Matrices globales
    K = np.zeros((mesh["nb_nodes"], mesh["nb_nodes"]))
    # Boucle sur les éléments
    for i in range(mesh["nb_elements"]):
        dofs = get_dofs(mesh, i)
        k = np.zeros((len(dofs), len(dofs)))  
        gauss_points, gauss_weights = elem.INTEG_RULES["beam_2"]
        dx = mesh["dx"][i]
        coords = mesh["coords"][mesh["elements"][i]]
        for wi, xi in zip(gauss_weights, gauss_points):
            # Shape function derivatives
            dN_dx = elem.DSHAPE_FUNCTION["beam_linear"](xi)
            N = elem.SHAPE_FUNCTION["beam_linear"](xi)
            alpha = alpha_f( N@coords.reshape((-1,1)) )
            # Element stiffness matrix
            k += wi * E[i] * alpha * A * ( dN_dx.reshape((-1,1))*dN_dx) * 2/dx
        K[np.ix_(dofs, dofs)] += k
    return K


def compute_l2_operator(mesh, alpha_f=None):
    if alpha_f is None:
        alpha_f = lambda x: 1. 
    # Matrices globales
    mat = np.zeros((mesh["nb_nodes"], mesh["nb_nodes"]))
    # Boucle sur les éléments
    for i in range(mesh["nb_elements"]):
        dofs = get_dofs(mesh, i)
        k = np.zeros((len(dofs), len(dofs)))  
        gauss_points, gauss_weights = elem.INTEG_RULES["beam_2"]
        dx = mesh["dx"][i]
        coords = mesh["coords"][mesh["elements"][i]]
        for wi, xi in zip(gauss_weights, gauss_points):
            # Shape function derivatives
            dN_dx = elem.DSHAPE_FUNCTION["beam_linear"](xi)
            N = elem.SHAPE_FUNCTION["beam_linear"](xi)
            alpha = alpha_f( N@coords.reshape((-1,1)) )
            # Element stiffness matrix
            k += wi * alpha * ( N.reshape((-1,1))*N) * dx / 2.
        mat[np.ix_(dofs, dofs)] += k
    return mat

def compute_strain_and_stress(mesh, young, solution):
    strain = []
    stress = []
    for i in range(mesh["nb_elements"]):
        gauss_points, gauss_weights = elem.INTEG_RULES["beam_2"]
        dofs = get_dofs(mesh, i)
        coords = mesh["coords"][mesh["elements"][i]]
        dx = mesh["dx"][i]
        strain_element = []
        stress_elem = []
        for wi, xi in zip(gauss_weights, gauss_points):
            # shape function evaluated at gauss point
            B = elem.DSHAPE_FUNCTION["beam_linear"](xi)*2/dx
            eto = B @ solution[dofs]
            strain_element.append(eto[0])
            stress_elem.append(young[i] * eto[0])
        strain.append(strain_element)
        stress.append(stress_elem)
    return np.array(strain), np.array(stress)


def find_elem_in_ver(x_elem, mesh_ver):
    rve_length = mesh_ver["coords"][-1] - mesh_ver["coords"][0]
    nb_rve = int(x_elem / rve_length)
    local_pos = x_elem - nb_rve*rve_length
    for i, elem in enumerate(mesh_ver["elements"]):
        coords = mesh_ver["coords"][elem]
        if coords[0] < local_pos < coords[1]:
            return i
    return None

def relocalize_strain_and_stress(mesh, mesh_ver, strain_h, tensor_A, tensor_B):
    strain = []
    stress = []
    for e in range(mesh["nb_elements"]):
        strain_element = []
        stress_elem = []
        x_elem = mesh["coords"][mesh["elements"][e]]
        center = (x_elem[0] + x_elem[1])/2.
        e_ver = find_elem_in_ver(center, mesh_ver)

        for i, eto in enumerate(strain_h[e]):
            strain_element.append(tensor_A[e_ver, i] * eto )
            stress_elem.append(tensor_B[e_ver, i] * eto )
        strain.append(strain_element)
        stress.append(stress_elem)
    return np.array(strain), np.array(stress)
