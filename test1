import numpy as np
from collections import OrderedDict

from remp_to_marple import write_remp_region_geo, write_object_boundary

MAX_INT = 2147483648
MAX_DIM = 3


def read_marple_list(fp, cnt):
    list1 = []
    list2 = []
    list3 = []

    for _ in range(cnt + 1):
        list1.append(int(fp.readline().strip()))

    if cnt == 0:
        fp.readline()
        return []

    print(list1)
    idx = 0
    while idx < list1[-1]:
        for i in fp.readline().split():
            if i.lstrip('-').isdigit():
                list2.append(int(i))
                idx += 1

    print(list2)
    for idx in range(cnt):
        list3.append(
            list2[list1[idx]: list1[idx+1]]
        )

    fp.readline()
    return list3


def read_symmetry(fp):
    t = fp.readline().strip().split()
    n = int(t[0])
    eps = float(t[1])

    fp.readline()
    fp.readline()
    line = fp.readline()
    line = fp.readline()
    line = fp.readline()


def read_regions(fp):
    line = fp.readline().strip()
    nReg = int(line)

    assert nReg == 1
    regions = []
    for _ in range(nReg):
        r = Region()
        r.geometry.read_region_geom(fp)
        r.read_region_info(fp)
        regions.append(r)

    return regions


class Marking:
    def __init__(self):
        self.name = ""
        self.labels = []
        self.elements_of_dim = dict()

    def read_marking_info(self, fp):
        self.name, labels_count = fp.readline().strip().split()

        self.labels = [l for l in range(int(labels_count))]

        has_elems_of_dim = []
        for _ in range(MAX_DIM+ 1):
            has_elems_of_dim.append(int(fp.readline().strip()) == 1)

        for dimension, has in enumerate(has_elems_of_dim):
            if has:
                self.elements_of_dim[dimension] = fp.readline().strip()
                fp.readline()  # ????????????????????


class SuperElement:
    def __init__(self, dim, name="", elements=None):
        self.dim = dim
        self.name = name
        if elements is None:
            elements = []
        self.elements_list = elements


class RegionGeo:
    def __init__(self):
        self.elem_counts = []
        self.pts = []
        self.elements_incident_to_dimension = OrderedDict()

    def read_region_geom(self, fp):
        maxdim = int(fp.readline().strip())
        assert maxdim == 3

        self.elem_counts = list(map(int, fp.readline().strip().split()))

        nAttr = int(fp.readline().strip())

        self.read_incidence(fp)

        self.read_points(fp)

    def read_points(self, fp):
        for _ in range(self.elem_counts[0]):
            p = map(float, fp.readline().strip().strip('( )').split(', '))
            self.pts.append(tuple(p))

    @staticmethod  # or not; assert elems_counts elemsTo_count
    def read_elem_incidence(fp):
        print(fp.readline())
        line = fp.readline().strip().split()
        print(line)
        elemsTo_count, elemsFr_count = map(int, line)
        print('elemsTo_count = %d elemsFro_count = %d' % (elemsTo_count, elemsFr_count))

        elemnt_incident_to_elemnts = read_marple_list(fp, elemsTo_count)

        return elemnt_incident_to_elemnts

    def read_incidence(self, fp):
        incidence_matrix = np.zeros((MAX_DIM + 1, MAX_DIM + 1), int)
        for i in range(MAX_DIM + 1):
            incidence_matrix[i, :] = [int(x) for x in fp.readline().strip().split()]

        for iTo, dimensions in enumerate(incidence_matrix):
            for iFro, exists in enumerate(dimensions):
                if exists:
                    print(iTo, iFro)
                    assert iFro == iTo - 1
                    self.elements_incident_to_dimension[iTo] = RegionGeo.read_elem_incidence(fp)
                    #fp.readline()
                    break

                assert iTo > 0 or iFro > iTo - 1


class Region:
    def __init__(self):
        self.geometry = RegionGeo()

        self.label = ""
        self.name = ""
        self.markings = []

        self.super_elements = []

    def read_region_info(self, fp):
        fp.readline()
        # 7 numbers on this line:
        #
        # size = 0 rank = 1 margin_width = 0
        #
        # special_elems_count[dim=0] = 0
        # special_elems_count[dim=1] = 0
        # special_elems_count[dim=2] = 0
        # special_elems_count[dim=3] = 0

        fp.readline()
        # special_elem_type1_description = -1

        fp.readline()
        # special_elem_type2_description = -1

        self.label, self.name = fp.readline().strip().split()

        reg_markings_count = int(fp.readline().strip())

        for _ in range(reg_markings_count):
            m = Marking()
            m.read_marking_info(fp)
            self.markings.append(m)

        line = fp.readline()
        for dim in range(MAX_DIM + 1):
            line = fp.readline().strip()
            element_count = int(line)
            super_elemnts_elemnts_list = read_marple_list(fp, element_count)
            for elements in super_elemnts_elemnts_list:
                name = fp.readline().strip()
                se = SuperElement(dim, name, elements)
                self.super_elements.append(se)

    def write(self, fn):
        write_remp_region_geo(r'DEFAULT.GRD', r'DEFAULT.CEL', r'test.mrp')
        fp = open(fn, 'a')
        fp.write('0 1 0 0 0 0 0\n-1\n-1\n')
        fp.write('{} {}\n'.format(self.label, self.name))
        fp.write('{}\n'.format(len(self.markings)))
        for m in self.markings:
            fp.write('{} {}\n'.format(m.name, len(m.labels)))
        if len(self.markings) > 0:
            # TODO: what is going on here??
            fp.write('0\n0\n0\n1\n')
            fp.write('1\n')
            fp.write('1\n')

        # no boundary conditions on points
        dim = 0
        fp.write('0\n0\n\n')

        # no boundary conditions on edges
        dim = 1
        fp.write('0\n0\n\n')
        fp.close()

        # faces have boundary conditions
        dim = 2
        write_boundaries()

        fp.open(fn, 'a')
        # no boundary conditions on volumes
        fp.write('0\n0\n\n')

        fp.close()


def read_mrp_file(fp):
    read_symmetry(fp)

    return read_regions(fp)


def write_symmetry(fn):
    pass


def write_mrp_file(fn, region):
    fp = open(fn)
    write_symmetry()
    region.write(fn)


if __name__ == '__main__':
    f = open('mesh.mrp')
    regions_ = read_mrp_file(f)

    write_mrp_file('test.mrp', regions_[0])
