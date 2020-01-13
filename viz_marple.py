import numpy as np
from mayavi import mlab

from test1 import read_symmetry, read_regions

MAX_INT = 2147483648


def pos(element_number):
    if element_number < 0:
        return element_number + MAX_INT
    return element_number

if __name__ == '__main__':
    fname = 'Cyl-Hex-10K.mrp'
    fp = open(fname)
    fp.readline()  #4
    fp.readline()  #SymTransforms 2
    fp.readline()  #spaceDividers 1
    fp.readline()  #    Meshes::Mesh3D 2
    fp.readline()  #elemGroup 1

    read_symmetry(fp)

    r = read_regions(fp)[0]

    obj_boundary = []
    for se in r.super_elements:
        if se.name == 'Wall':
            obj_boundary = se.elements_list

    pts = np.asarray(r.geometry.pts)
    tri = np.zeros((len(obj_boundary)*2), dtype=(np.uint16, 3))
    for idx, el in enumerate(obj_boundary):
        e1, e2, e3, e4 = r.geometry.elements_incident_to_dimension[2][el] # 4 numbers
        e1, e2, e3, e4 = map(pos, [e1, e2, e3, e4])
        p1, p2 = r.geometry.elements_incident_to_dimension[1][e1]
        p2_, p3 = r.geometry.elements_incident_to_dimension[1][e2]
        p3_, p4 = r.geometry.elements_incident_to_dimension[1][e3]
        p4_, p1_ = r.geometry.elements_incident_to_dimension[1][e4]
        p1, p2, p3, p4 = list(set((pos(p1), pos(p2), pos(p3), pos(p4), pos(p1_), pos(p2_), pos(p3_), pos(p4_))))

        tri[2 * idx] = (pos(p1), pos(p2), pos(p3))
        tri[2 * idx + 1] = (pos(p1), pos(p4), pos(p3))

    mlab.triangular_mesh(pts[:, 0], pts[:, 1], pts[:, 2], tri, color=(0.2, 0.3, 0.7))
    mlab.show()