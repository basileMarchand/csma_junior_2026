import numpy as np 

def create_mesh_linear( x_min, x_max, nb_elements, progression="linear", offset_node_numbering=0):
    mesh = {}
    if progression=="linear":
        coords = np.linspace(x_min, x_max, nb_elements+1)
    elif progression=="left_to_right":
        coords = np.geomspace(1.+x_min, 1.+x_max, nb_elements+1) - 1.
    elif progression=="right_to_left":
        coords = np.geomspace(1.+x_max, 1.+x_min, nb_elements+1) - 1.
        coords = coords[::-1]
    mesh["nb_nodes"] = nb_elements+1
    mesh["nb_elements"] = nb_elements
    mesh["coords"] = coords
    mesh["dx"] = coords[1:] - coords[:-1] # [:-1] selectionne toute la liste sauf le dernier nombre; 
                                            # [1:] sélectionne toute la liste sauf le premier nombre
    mesh["nodes"] = np.arange(offset_node_numbering, nb_elements+1)
    mesh["dofs"] = mesh["nodes"]
    
    elements = [None]*nb_elements
    for i in range( nb_elements):
        elements[i] = [mesh["nodes"][i], mesh["nodes"][i+1]]
    mesh["elements"] = elements
    return mesh