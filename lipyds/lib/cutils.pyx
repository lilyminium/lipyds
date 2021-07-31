
cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport fabs, copysign, sqrt, round
from libc.float cimport FLT_EPSILON, FLT_MAX
from MDAnalysis.lib.c_distances import ortho_pbc, triclinic_pbc
from MDAnalysis.lib.mdamath import triclinic_vectors

@cython.boundscheck(False)
@cython.wraparound(False)
cdef minimum_image_ortho(float[:] coordinates, float[:] box, float[:] inverse_box):
    # this is copied from MDAnalysis.lib.include.calc_distances.minimum_image
    cdef int i
    cdef double disp

    for i in range(3):
        if box[i] > FLT_EPSILON:
            disp = inverse_box[i] * coordinates[i]
            coordinates[i] = box[i] * (disp - round(disp))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef minimum_image_triclinic(float[:] coordinates, float[:, :] box):
    # this is copied from MDAnalysis.lib.include.calc_distances.minimum_image_triclinic
    # which is in turn adapted from LAMMPS
    cdef float[3] image = [0, 0, 0]
    cdef float min_dist_squared = FLT_MAX
    cdef float dist_squared, rx
    cdef float[2] ry
    cdef float[3] rz
    cdef int ix, iy, iz, ii

    for ix in range(-1, 2):
        rx = coordinates[0] + box[0][0] * ix
        for iy in range(-1, 2):
            ry[0] = rx + box[1][0] * iy
            ry[1] = coordinates[1] + box[1][1] * iy
            for iz in range(-1, 2):
                rz[0] = ry[0] + box[2][0] * iz
                rz[1] = ry[1] + box[2][1] * iz
                rz[2] = coordinates[2] + box[2][2] * iz

                dist_squared = rz[0]*rz[0] + rz[1]*rz[1] + rz[2]*rz[2]
                
                if dist_squared < min_dist_squared:
                    dist_squared = min_dist_squared
                    for ii in range(3):
                        image[ii] = rz[ii]
    for ii in range(3):
        coordinates[ii] = image[ii]


@cython.boundscheck(False)
@cython.wraparound(False)
def _mean_unwrap_around(float[:, :] coordinates, float[:] center, float[:] box):
    cdef int i
    cdef float[:, ::1] triclinic_box = np.empty((3, 3), dtype=np.single)
    cdef float[:] inverse_box = np.empty(3, dtype=np.single)
    #cdef np.ndarray[float, ndim=1] unit_image = np.copy(center)
    cdef float[:] unit_image = np.empty(3, dtype=np.single)
    
    for i in range(3):
        unit_image[i] = center[i]

    for i in range(3, 6):
        if box[i] != 90.0:
            triclinic_box = triclinic_vectors(box)
            minimum_image_triclinic(unit_image, triclinic_box)
            return _mean_unwrap_around_triclinic(coordinates, center, unit_image, box, triclinic_box)

    for i in range(3):
        inverse_box[i] = 1 / box[i]
    minimum_image_ortho(unit_image, box, inverse_box)
    return _mean_unwrap_around_ortho(coordinates, center, unit_image, box, inverse_box)


@cython.boundscheck(False)
@cython.wraparound(False)
def mean_unwrap_around(float[:, :] coordinates, float[:] center, float[:] box):
    return np.array(_mean_unwrap_around(coordinates, center, box), dtype=np.single)


def unwrap_around(float[:, :] coordinates, center, box):
    cdef int i
    cdef float[:, ::1] triclinic_box = np.empty((3, 3), dtype=np.single)
    cdef float[:] inverse_box = np.empty(3, dtype=np.single)
    cdef np.ndarray[float, ndim=1] unit_image = np.array(center, dtype=np.single)

    for i in range(3, 6):
        if box[i] != 90.0:
            triclinic_box = triclinic_vectors(box)
            triclinic_pbc(unit_image, box)
            return unwrap_around_triclinic(coordinates, center, unit_image, box, triclinic_box)
    for i in range(3):
        inverse_box[i] = 1 / box[i]
    ortho_pbc(unit_image, box)
    return unwrap_around_ortho(coordinates, center, unit_image, box, inverse_box)

@cython.boundscheck(False)
@cython.wraparound(False)
def unwrap_around_ortho(float[:, :] coordinates, float[:] center,
                        float[:] unit_image, float[:] box, float[:] inverse_box):
    cdef int i, i_coord
    cdef int n_coords = coordinates.shape[0]
    cdef float[:, ::1] unwrapped = np.empty((n_coords, 3), dtype=np.single)
    cdef float displacement[3]
    cdef float vector[3]
        
    for i in range(3):
        displacement[i] = center[i] - unit_image[i]
    
    for i_coord in range(n_coords):
        for i in range(3):
            vector[i] = coordinates[i_coord][i] - unit_image[i]
        minimum_image_ortho(vector, box, inverse_box)
        for i in range(3):
            unwrapped[i_coord][i] = unit_image[i] + vector[i] + displacement[i]
    
    return unwrapped



            
@cython.boundscheck(False)
@cython.wraparound(False)
def _mean_unwrap_around_ortho(float[:, :] coordinates, float[:] center,
                             float[:] unit_image, float[:] box, float[:] inverse_box):
    cdef int i, i_coord
    cdef float mean[3]
    cdef float displacement[3]
    cdef float vector[3]
    cdef int n_coords = coordinates.shape[0]

    
    for i in range(3):
        displacement[i] = center[i] - unit_image[i]
    
    for i_coord in range(n_coords):
        for i in range(3):
            vector[i] = coordinates[i_coord][i] - unit_image[i]
        minimum_image_ortho(vector, box, inverse_box)
        for i in range(3):
            mean[i] += vector[i]
    
    for i in range(3):
        mean[i] /= n_coords
        mean[i] += unit_image[i]
        mean[i] += displacement[i]
    return mean

@cython.boundscheck(False)
@cython.wraparound(False)
def _mean_unwrap_around_triclinic(float[:, :] coordinates, float[:] center,
                                 float[:] unit_image, float[:] box, float[:, :] triclinic_box):
    cdef int i, i_coord
    cdef float[3] mean = [0, 0, 0]
    cdef float displacement[3]
    cdef float vector[3]
    cdef int n_coords = coordinates.shape[0]
    
    
    for i in range(3):
        displacement[i] = center[i] - unit_image[i]
    
    for i_coord in range(n_coords):
        for i in range(3):
            vector[i] = coordinates[i_coord][i] - unit_image[i]
        minimum_image_triclinic(vector, triclinic_box)
        for i in range(3):
            mean[i] += vector[i]
    
    for i in range(3):
        mean[i] /= n_coords
        mean[i] += unit_image[i]
        mean[i] += displacement[i]
    return mean

@cython.boundscheck(False)
@cython.wraparound(False)
def unwrap_around_triclinic(float[:, :] coordinates, float[:] center,
                            float[:] unit_image, float[:] box, float[:, :] triclinic_box):
    cdef int i, i_coord
    cdef int n_coords = coordinates.shape[0]
    cdef float[:, ::1] unwrapped = np.empty((n_coords, 3), dtype=np.single)
    cdef float displacement[3]
    cdef float vector[3]
    
    
    for i in range(3):
        displacement[i] = center[i] - unit_image[i]
    
    for i_coord in range(n_coords):
        for i in range(3):
            vector[i] = coordinates[i_coord][i] - unit_image[i]
        minimum_image_triclinic(vector, triclinic_box)
        for i in range(3):
            unwrapped[i_coord][i] = unit_image[i] + vector[i] + displacement[i]
    
    return unwrapped

cdef _calc_norm_vec3(float[:] vec):
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])


cdef _calc_dot_vec3(float[:] vec1, float[:] vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]


def calc_cosine_similarity(vectors1, vectors2):
    vectors1 = np.ascontiguousarray(vectors1)
    vectors2 = np.ascontiguousarray(vectors2)
    cdef float[:, :] vec1 = vectors1
    cdef float[:, :] vec2 = vectors2
    cdef int i, j
    cdef int numVec1 = vectors1.shape[0]
    cdef int numVec2 = vectors2.shape[0]
    cdef float[:, ::1] cosines = np.empty((numVec1, numVec2), dtype=np.single)
    cdef float[:] _norm2 = np.empty(numVec2, dtype=np.single)
    cdef float sim;

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
            cosines[i, j] = sim;

    return cosines

def get_centers_by_resindices(positions, centers,  resindices, box):
    cdef int n_values = centers.shape[0]
    cdef int n_positions = positions.shape[0]
    cdef float[:, ::1] mean = np.empty((n_values, 3), dtype=np.single)
    cdef int i = 0
    cdef int j = 0
    cdef int k = 1
    cdef int rix

    if n_positions == 0:
        return np.array(positions, dtype=np.single)

    if n_positions == n_values:
        return np.array(positions, dtype=np.single)

    rix = resindices[0]
    for k in range(1, n_positions):
        if resindices[k] != rix:
            i_mean = mean_unwrap_around(positions[j:k], centers[i], box)
            mean[i][0] = i_mean[0]
            mean[i][1] = i_mean[1]
            mean[i][2] = i_mean[2]
            j = k
            i += 1
            rix = resindices[k]
        k += 1
    i_mean = mean_unwrap_around(positions[j:], centers[i], box)
    mean[i][0] = i_mean[0]
    mean[i][1] = i_mean[1]
    mean[i][2] = i_mean[2]
    return np.asarray(mean)


def get_centers_around_first_by_resindices(float[:, :] positions,
                                            long[:] resindices,
                                            float[:] box):
    cdef int n_positions = positions.shape[0]
    cdef float[:, ::1] mean = np.empty((n_positions, 3), dtype=np.single)
    cdef int i = 0
    cdef int j = 0
    cdef int k = 1
    cdef int rix
    cdef float i_mean[3]

    if n_positions == 0:
        return positions

    rix = resindices[0]
    for k in range(1, n_positions):
        if resindices[k] != rix:
            i_mean = mean_unwrap_around(positions[j:k], positions[j], box)
            mean[i][0] = i_mean[0]
            mean[i][1] = i_mean[1]
            mean[i][2] = i_mean[2]
            j = k
            i += 1
            rix = resindices[k]
        k += 1
    i_mean = mean_unwrap_around(positions[j:], positions[j], box)
    mean[i][0] = i_mean[0]
    mean[i][1] = i_mean[1]
    mean[i][2] = i_mean[2]
    return np.array(mean[:i + 1])