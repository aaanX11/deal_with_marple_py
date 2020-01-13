
import numpy as np
from mayavi import mlab

from remp_to_marple import read_grd_dim, read_cel, read_grd


def get_remp_geo(grd_fn, cel_fn):
    nx, ny, nz = read_grd_dim(grd_fn)
    space_li = read_cel(cel_fn, nx * ny * nz)
    space = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.uint8)
    space[1:-1, 1:-1, 1:-1] = space_li.reshape((nx, ny, nz))
    space[np.where(space == 1)] = 0
    space[0, ...] = space[1, ...]
    space[-1, ...] = space[-2, ...]
    space[:, 0, :] = space[:, 1, :]
    space[:, -1, :] = space[:, -2, :]
    space[..., 0] = space[..., 1]
    space[..., -1] = space[..., -2]
    #print(space[int(nx/2), int(ny/2), :])
    #mlab.imshow(space[int(nx/2), ...])
    #mlab.show()
    return space


def get_faces(space):

    nx, ny, nz = map(lambda x: x - 2, space.shape)

    def idx(ix, iy, iz):
        return ix*(ny + 1)*(nz + 1) + iy*(nz + 1) + iz

    tri = np.zeros((2 * (nx + 1) * ny * nz), dtype=(np.uint64, 3))
    tri_idx = 0
    for ix in range(nx + 1):
        for iy in range(ny):
            for iz in range(nz):

                if (not space[ix:ix + 2, iy + 1, iz + 1].all()) and space[ix:ix + 2, iy + 1, iz + 1].any():
                    tri[tri_idx] = (idx(ix + 1, iy, iz),
                                    idx(ix + 1, iy, iz + 1),
                                    idx(ix + 1, iy + 1, iz + 1))
                    tri[tri_idx + 1] = (idx(ix + 1, iy, iz),
                                        idx(ix + 1, iy + 1, iz),
                                        idx(ix + 1, iy + 1, iz + 1))
                    tri_idx += 2
    for ix in range(nx):
        for iy in range(ny + 1):
            for iz in range(nz):

                if (not space[ix + 1, iy:iy + 2, iz + 1].all()) and space[ix + 1, iy:iy + 2, iz + 1].any():
                    tri[tri_idx] = (idx(ix, iy + 1, iz),
                                    idx(ix, iy + 1, iz + 1),
                                    idx(ix + 1, iy + 1, iz + 1))
                    tri[tri_idx + 1] = (idx(ix, iy + 1, iz),
                                        idx(ix + 1, iy + 1, iz),
                                        idx(ix + 1, iy + 1, iz + 1))
                    tri_idx += 2

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz + 1):

                if (not space[ix + 1, iy + 1, iz:iz + 2].all()) and space[ix + 1, iy + 1, iz:iz + 2].any():
                    tri[tri_idx] = (idx(ix, iy, iz + 1),
                                    idx(ix, iy + 1, iz + 1),
                                    idx(ix + 1, iy + 1, iz + 1))
                    tri[tri_idx + 1] = (idx(ix, iy, iz + 1),
                                        idx(ix + 1, iy, iz + 1),
                                        idx(ix + 1, iy + 1, iz + 1))
                    tri_idx += 2
    print(tri_idx)
    return tri[:tri_idx]


def tri_obj_bou(grd_fn, cel_fn):
    space = get_remp_geo(grd_fn, cel_fn)

    faces = get_faces(space)
    return faces


def get_points_list(grd_fn):
    nx, ny, nz = read_grd_dim(grd_fn)
    x, y, z = read_grd(grd_fn)
    pts = np.zeros(((nx + 1) * (ny + 1) * (nz + 1)), dtype=(np.float, 3))
    p_idx = 0
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            for iz in range(nz + 1):
                pts[p_idx] = (x[ix], y[iy], z[iz])
                p_idx += 1
    return pts


def draw_cartesius(pts, tri):

    mlab.triangular_mesh(pts[:, 0], pts[:, 1], pts[:, 2], tri, opacity=0.5, color=(0.6, 0.3, 0.3))


def draw_tri(pts, tri):
    mlab.triangular_mesh(pts[:, 0], pts[:, 1], pts[:, 2], tri, color=(0.2, 0.3, 0.7))


def draw():

    faces = tri_obj_bou('ED1.GRD', 'ED1.CEL')

    pts = get_points_list('ED1.GRD')

    draw_cartesius(pts, faces)

    tri = np.loadtxt('triangles')
    faces = np.zeros((tri.shape[0]), dtype=(np.float, 3))
    idx = 0
    for t in tri:
        if t[3] == 2:
            faces[idx] = t[:3]
            idx += 1
    pts = np.loadtxt('points')
    draw_tri(pts, faces[:idx])
    mlab.show()


if __name__ == '__main__':
    draw()
