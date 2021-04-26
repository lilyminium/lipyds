
cimport cython
import numpy
cimport numpy

from libc.math cimport fabs, copysign, sqrt

def unwrap_around(numpy.ndarray coords, numpy.ndarray center,
                  numpy.ndarray box):
    cdef int i
    cdef double half_x, half_y, half_z, dx, dy, dz
    coords = numpy.ascontiguousarray(coords)
    cdef int numCoords = coords.shape[0]

    # Unwrap coordinates to around the proposed center
    half_x = box[0] * 0.5;
    half_y = box[1] * 0.5;
    half_z = box[2] * 0.5;

    for i in range(numCoords):
        dx = coords[i][0] - center[0]
        dy = coords[i][1] - center[1]
        dz = coords[i][2] - center[2]

        # subtract (sign of difference) * (magnitude of box) if 
        # difference is greater than half the box
        coords[i][0] -= (fabs(dx) > half_x) * copysign(box[0], dx)
        coords[i][1] -= (fabs(dy) > half_y) * copysign(box[1], dy)
        coords[i][2] -= (fabs(dz) > half_z) * copysign(box[2], dz)

    return coords

def mean_unwrap_around(numpy.ndarray coords, numpy.ndarray center,
                       numpy.ndarray box):
    cdef int i
    cdef double half_x, half_y, half_z, dx, dy, dz, ux, uy, uz
    cdef numpy.ndarray[numpy.float64_t, ndim=1] mean = numpy.zeros(3, dtype=numpy.float64)
    coords = numpy.ascontiguousarray(coords)
    cdef int numCoords = coords.shape[0]

    # Unwrap coordinates to around the proposed center
    half_x = box[0] * 0.5;
    half_y = box[1] * 0.5;
    half_z = box[2] * 0.5;

    for i in range(numCoords):
        dx = coords[i][0] - center[0];
        dy = coords[i][1] - center[1];
        dz = coords[i][2] - center[2];

        # subtract (sign of difference) * (magnitude of box) if 
        # difference is greater than half the box
        ux = (fabs(dx) > half_x) * copysign(box[0], dx);
        uy = (fabs(dy) > half_y) * copysign(box[1], dy);
        uz = (fabs(dz) > half_z) * copysign(box[2], dz);

        mean[0] += coords[i][0] - ux;
        mean[1] += coords[i][1] - uy;
        mean[2] += coords[i][2] - uz;
    

    mean[0] /= numCoords;
    mean[1] /= numCoords;
    mean[2] /= numCoords;
    return mean


cdef _calc_norm_vec3(double[:] vec):
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])


cdef _calc_dot_vec3(double[:] vec1, double[:] vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]


def calc_cosine_similarity(numpy.ndarray vectors1, numpy.ndarray vectors2):
    vectors1 = numpy.ascontiguousarray(vectors1)
    vectors2 = numpy.ascontiguousarray(vectors2)
    cdef double[:, :] vec1 = vectors1
    cdef double[:, :] vec2 = vectors2
    cdef int i, j
    cdef int numVec1 = vectors1.shape[0]
    cdef int numVec2 = vectors2.shape[0]
    cosines = numpy.empty((numVec1, numVec2))
    cdef double[:, :] _cosines = cosines
    norm2 = numpy.empty(numVec2)
    cdef double[:] _norm2 = norm2
    cdef double sim;

    for j in range(numVec2):
        _norm2[j] = _calc_norm_vec3(vec2[j]);
    
    for i in range(numVec1):
        norm_i = _calc_norm_vec3(vec1[i]);
        for j in range(numVec2):
            norm_ij = norm_i * _norm2[j]
            if norm_ij > 0:
                sim =  _calc_dot_vec3(vec1[i], vec2[j]) / norm_ij;
            else:
                sim = 1
            _cosines[i, j] = sim;

    return cosines