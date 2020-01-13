
import numpy as np
import vtk
from mayavi import mlab

from remp_to_marple import read_cel, read_grd, read_grd_dim


def get_x_obj_boundary_faces(space, x, y, z):
    nx, ny, nz = space.shape

    tri = np.zeros((nx * ny * nz), dtype=(np.float16, 4))

    idx_v = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if space[ix + 1, iy + 1, iz + 1]:
                    if not space[ix, iy + 1, iz + 1]:
                        pts[idx_v] = (ix, iy, iz)
                        idx_v += 1

    return pts[:idx_v, ], tri[:idx_v, ]

@mlab.show
def show():
    #mlab.test_plot3d()
    pts = np.loadtxt('points')
    tri = np.loadtxt('triangles')
    for t in tri:
        t[:-1] -= 1
    mlab.triangular_mesh(pts[:, 0], pts[:, 0], pts[:, 0], tri[:, :-1])


if __name__ == '__main__':

    cel_fn = r'DEFAULT.CEl'
    grd_fn = r'DEFAULT.GRD'

    nx, ny, nz = read_grd_dim(grd_fn)
    space_li = read_cel(cel_fn, nx * ny * nz)
    space = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.uint8)
    space[1:-1, 1:-1, 1:-1] = space_li.reshape((nx, ny, nz))
    space[0, ...] = space[1, ...]
    space[-1, ...] = space[-2, ...]
    space[:, 0, :] = space[:, 1, :]
    space[:, -1, :] = space[:, -2, :]
    space[..., 0] = space[..., 1]
    space[..., -1] = space[..., -2]

    space.resize((nx, ny, nz))

    #x_faces = get_x_obj_boundary_faces(space)

    xx, yy, zz = read_grd(grd_fn)



    #mlab.test_plot3d()
    show()


    #mlab.show()
