import numpy as np
from mayavi import mlab

from test1 import read_symmetry, read_regions

MAX_INT = 2147483648


def pos(element_number):
    if element_number < 0:
        return element_number + MAX_INT
    return element_number


def draw_faces(pts, faces, edges_incident_to_faces, points_incident_to_edges, colour, opacity):

    tri = np.zeros((2 * len(faces)), dtype=(np.uint16, 3))
    for idx, el in enumerate(faces):

        edges = [(1, x) if x >= 0 else (-1, MAX_INT + x)
                 for x in edges_incident_to_faces[el]]

        points = []
        for e in edges:
            p1, p2 = [x if x >= 0 else MAX_INT + x
                      for x in points_incident_to_edges[e[1]]]
            if e[0] < 0:
                p1, p2 = p2, p1

            if not points:
                points = [p1, p2]
            else:
                if p1 not in points:
                    points.extend([p1, p2])
            # points[idx] = p1
            # if points[idx] < 0:
            #    points[idx] += MAX_INT

        #try:
        tri[2 * idx] = (points[0], points[1], points[2])
        #except OverflowError:
        #    print(points)
        #    print(idx)
        #    input()
        tri[2 * idx + 1] = (points[0], points[3], points[2])

    mlab.triangular_mesh(pts[:, 0], pts[:, 1], pts[:, 2], tri, color=colour, opacity=opacity)


def main():
    fname = 'Cyl-Hex-10K.mrp'
    fp = open(fname)
    fp.readline()  #4
    fp.readline()  #SymTransforms 2
    fp.readline()  #spaceDividers 1
    fp.readline()  #    Meshes::Mesh3D 2
    fp.readline()  #elemGroup 1

    read_symmetry(fp)

    r = read_regions(fp)[0]
    pts = np.asarray(r.geometry.pts)

    faces = []
    for idx, se in enumerate(r.super_elements):
        #if se.name == 'Wall':
        faces = se.elements_list
        draw_faces(pts, faces,
                   r.geometry.elements_incident_to_dimension[2], r.geometry.elements_incident_to_dimension[1],
                   (0.1, 0.1, 0.1*idx), 0.1*idx)

    mlab.show()

if __name__ == '__main__':
    main()