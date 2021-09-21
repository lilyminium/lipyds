

cimport cython
import numpy as np
cimport numpy as np


cdef extern from "math.h":
    bint isnan(double x)

cdef extern from "pbcutils.h":
    ctypedef float coordinate[3]
    void _mean_unwrap_around_first_ortho(coordinate *coords, int numcoords, long *resindices, float *box, coordinate *output)
    void _mean_unwrap_around_centers_ortho(coordinate *coords, coordinate *centers, int numcoords, long *resindices, float *box, coordinate *output)
    void _mean_unwrap_around_first_triclinic(coordinate *coords, int numcoords, long *resindices, float *box, coordinate *output)
    void _mean_unwrap_around_centers_triclinic(coordinate *coords, coordinate *centers, int numcoords, long *resindices, float *box, coordinate *output)
    void _box_to_triclinic_vectors(float *box, float *triclinic_vectors)
    void _project_distances_nobox(coordinate *coordinates, coordinate *orientations, int *index_as, int *index_bs, double *distances, int n_pairs, float angle_factor)
    void _project_distances_triclinic(coordinate *coordinates, coordinate *orientations, int *index_as, int *index_bs, double *distances, int n_pairs, float *box, float angle_factor)
    void _project_distances_ortho(coordinate *coordinates, coordinate *orientations, int *index_as, int *index_bs, double *distances, int n_pairs, float *box, float angle_factor)
    void _unwrap_around_ortho(coordinate *coords, int numcoords, coordinate center, float *box, coordinate *output)
    void _unwrap_around_triclinic(coordinate *coords, int numcoords, coordinate center, float *box, coordinate *output)
    void _calc_cosine_similarity(coordinate a, coordinate *bs, int n_bs, double *cosines)

@cython.boundscheck(False)
@cython.wraparound(False)
def project_distances(np.ndarray coordinates, np.ndarray orientations,
                      np.ndarray pairs_1, np.ndarray pairs_2,
                      np.ndarray distances,
                      box, float angle_factor):
    
    cdef int n_pairs = pairs_1.shape[0]
    cdef int i
    cdef float[:] isbox
    if box is None:
        _project_distances_nobox(<coordinate*> coordinates.data,
                                 <coordinate*> orientations.data,
                                 <int*> pairs_1.data, <int*> pairs_2.data,
                                 <double*> distances.data,
                                 n_pairs, angle_factor)
            
    else:
        isbox = box
        for i in range(3, 6):
            if box[i] != 90:
                _project_distances_triclinic(<coordinate*> coordinates.data,
                                             <coordinate*> orientations.data,
                                             <int*> pairs_1.data,
                                             <int*> pairs_2.data,
                                             <double*> distances.data, 
                                             n_pairs, &isbox[0],
                                             angle_factor)
                break
        else:
            _project_distances_ortho(<coordinate*> coordinates.data,
                                     <coordinate*> orientations.data,
                                     <int*> pairs_1.data, <int*> pairs_2.data,
                                     <double*> distances.data,
                                     n_pairs, &isbox[0],
                                     angle_factor)
    return distances

@cython.boundscheck(False)
@cython.wraparound(False)
def mean_unwrap_around(np.ndarray coordinates, np.ndarray centers,
                      np.ndarray resindices, np.ndarray box):
    if centers is None:
        output = np.full_like(coordinates, np.nan, dtype=np.single)
        return mean_unwrap_around_first(coordinates, resindices, box, output)
    output = np.full_like(centers, np.nan, dtype=np.single)
    return mean_unwrap_around_centers(coordinates, centers, resindices, box, output)

@cython.boundscheck(False)
@cython.wraparound(False)
def mean_unwrap_around_centers(np.ndarray coordinates, np.ndarray centers,
                               np.ndarray resindices, np.ndarray box,
                               np.ndarray output):
    cdef int i
    cdef int n_centers = centers.shape[0]
    cdef int n_coordinates = coordinates.shape[0]
    
    for i in range(3, 6):
        if box[i] != 90:
            _mean_unwrap_around_centers_triclinic(<coordinate*> coordinates.data,
                                                  <coordinate*> centers.data,
                                                  n_coordinates,
                                                  <long*> resindices.data,
                                                  <float*> box.data,
                                                  <coordinate*>output.data)
            break
    else:
        _mean_unwrap_around_centers_ortho(<coordinate*> coordinates.data,
                                          <coordinate*> centers.data,
                                          n_coordinates,
                                          <long*> resindices.data,
                                          <float*> box.data,
                                          <coordinate*>output.data)
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def mean_unwrap_around_first(np.ndarray coordinates, np.ndarray resindices,
                             np.ndarray box, np.ndarray output):
    cdef int i, j
    cdef int n_coordinates = coordinates.shape[0]

    for i in range(n_coordinates):
        for j in range(3):
            output[i][j] = np.nan

    for i in range(3, 6):
        if box[i] != 90:
            _mean_unwrap_around_first_triclinic(<coordinate*> coordinates.data,
                                                n_coordinates,
                                                <long*> resindices.data,
                                                <float*> box.data,
                                                <coordinate*>output.data)
            break
    else:
        _mean_unwrap_around_first_ortho(<coordinate*> coordinates.data,
                                        n_coordinates,
                                        <long*> resindices.data,
                                        <float*> box.data,
                                        <coordinate*>output.data)
            

    output = np.asarray(output)

    for i in range(n_coordinates):
        if np.isnan(output[i][0]):
            return output[:i]
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def unwrap_around(np.ndarray coordinates, np.ndarray center,
                  np.ndarray box, np.ndarray output):
    cdef int n_coordinates = coordinates.shape[0]

    for i in range(3, 6):
        if box[i] != 90.0:
            _unwrap_around_triclinic(<coordinate*> coordinates.data, n_coordinates,
                                     <coordinate> center.data,
                                     <float*> box.data, <coordinate*> output.data)
            return
    _unwrap_around_ortho(<coordinate*> coordinates.data, n_coordinates, <coordinate> center.data,
                         <float*> box.data, <coordinate*> output.data)
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_angle(float[:] a, float[:] b):
    cdef double norm_a, norm_b, norm_ab, ab

    norm_a = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
    norm_b = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
    norm_ab = norm_a * norm_b
    ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if norm_ab > 0:
        return ab / norm_ab
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_cosine_similarity(np.ndarray vector_a, np.ndarray vectors):
    cdef int n_bs = vectors.shape[0]
    cdef np.ndarray output = np.empty(n_bs)
    _calc_cosine_similarity(<coordinate> vector_a.data,
                            <coordinate*> vectors.data,
                            n_bs, <double*> output.data)
    return np.asarray(output)