

cimport cython
import numpy as np
cimport numpy as np

cdef extern from "pbcutils.h":
    ctypedef float coordinate[3]
    void _mean_unwrap_around_first_ortho(coordinate *coords,
                                            int numcoords,
                                            long *resindices,
                                            float *box,
                                            coordinate *output)
    void _mean_unwrap_around_centers_ortho(coordinate *coords,
                                              coordinate *centers,
                                              int numcoords,
                                              long *resindices,
                                              float *box,
                                              coordinate *output)
    void _mean_unwrap_around_first_triclinic(coordinate *coords,
                                            int numcoords,
                                            long *resindices,
                                            float *box,
                                            coordinate *output)
    void _mean_unwrap_around_centers_triclinic(coordinate *coords,
                                              coordinate *centers,
                                              int numcoords,
                                              long *resindices,
                                              float *box,
                                              coordinate *output)
    void box_to_triclinic_vectors(float *box, float *triclinic_vectors)
    void project_distances_nobox(coordinate *coordinates,
                                     coordinate *orientations,
                                     int *index_as,
                                     int *index_bs,
                                     double *distances,
                                     int n_pairs,
                                     float angle_factor)
    void project_distances_triclinic(coordinate *coordinates,
                                         coordinate *orientations,
                                         int *index_as,
                                         int *index_bs,
                                         double *distances,
                                         int n_pairs,
                                         float *box,
                                         float angle_factor)
    void project_distances_ortho(coordinate *coordinates,
                                     coordinate *orientations,
                                     int *index_as,
                                     int *index_bs,
                                     double *distances,
                                     int n_pairs,
                                     float *box,
                                     float angle_factor)





def project_distances(np.ndarray coordinates, np.ndarray orientations,
                      np.ndarray pairs_1, np.ndarray pairs_2, np.ndarray distances,
                      box, float angle_factor):
    
    cdef int n_pairs = pairs_1.shape[0]
    cdef int i
    cdef float[:] isbox
    if box is None:
        project_distances_nobox(<coordinate*> coordinates.data, <coordinate*> orientations.data,
                                <int*> pairs_1.data, <int*> pairs_2.data, <double*> distances.data,
                                n_pairs,
                                angle_factor)
            
    else:
        isbox = box
        for i in range(3, 6):
            if box[i] != 90:
                project_distances_triclinic(<coordinate*> coordinates.data, <coordinate*> orientations.data,
                                <int*> pairs_1.data, <int*> pairs_2.data, <double*> distances.data, 
                                n_pairs, &isbox[0],
                                angle_factor)
                break
        else:
            project_distances_ortho(<coordinate*> coordinates.data, <coordinate*> orientations.data,
                                <int*> pairs_1.data, <int*> pairs_2.data, <double*> distances.data,
                                n_pairs, &isbox[0],
                                angle_factor)

def mean_unwrap_around(np.ndarray coordinates, np.ndarray centers, np.ndarray resindices, np.ndarray box):
    if centers is None:
        output = np.full_like(coordinates, np.nan, dtype=np.single)
        return mean_unwrap_around_first(coordinates, resindices, box, output)
    output = np.full_like(centers, np.nan, dtype=np.single)
    return mean_unwrap_around_centers(coordinates, centers, resindices, box, output)


def mean_unwrap_around_centers(np.ndarray coordinates, np.ndarray centers, np.ndarray resindices, np.ndarray box, np.ndarray output):
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



def mean_unwrap_around_first(np.ndarray coordinates, np.ndarray resindices, np.ndarray box, np.ndarray output):
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
        for j in range(3):
            if output[i][j] == np.nan:
                return output[:i]
    return output