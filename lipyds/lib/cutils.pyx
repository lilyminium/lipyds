

cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport fabs


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
    void minimum_image_ortho(double* x, float* box, float* inverse_box)
    void minimum_image_triclinic(double* dx, float* box)

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



@cython.boundscheck(False)
@cython.wraparound(False)
def unwrap_coordinates_around_center(
    np.ndarray coordinates,
    np.ndarray center,
    np.ndarray box,
):
    """Move all atoms in a single molecule so that bonds don't split over
    images.

    This function is most useful when atoms have been packed into the primary
    unit cell, causing breaks mid molecule, with the molecule then appearing
    on either side of the unit cell. This is problematic for operations
    such as calculating the center of mass of the molecule. ::

       +-----------+     +-----------+
       |           |     |           |
       | 6       3 |     |         3 | 6
       | !       ! |     |         ! | !
       |-5-8   1-2-| ->  |       1-2-|-5-8
       | !       ! |     |         ! | !
       | 7       4 |     |         4 | 7
       |           |     |           |
       +-----------+     +-----------+


    Parameters
    ----------
    coordinates: :class:`numpy.ndarray`
        This array should be of shape (n_atoms, 3)
    center: :class:`numpy.ndarray`
        This array should be of shape(3,)
    box: :class:`numpy.ndarray`
        This should be of shape (6,)

    Returns
    -------
    unwrapped: numpy.ndarray
        The unwrapped atom coordinates.



    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.unwrap`


    """
    cdef np.intp_t i, j, natoms, atom_i
    cdef float[:, :] unwrapped
    cdef bint ortho
    cdef float[:, :] tri_box
    cdef float half_box[3]
    cdef float inverse_box[3]
    cdef double vec[3]
    cdef float box_[6]
    cdef bint is_unwrapped


    natoms = coordinates.shape[0]
    # Nothing to do for less than 2 atoms
    if natoms < 2:
        return np.array(coordinates)

    for i in range(3):
        half_box[i] = 0.5 * box[i]
        box_[i] = box[i]

    ortho = True
    for i in range(3, 6):
        box_[i] = box[i]
        if box[i] != 90.0:
            ortho = False

    if ortho:
        # If atomgroup is already unwrapped, bail out
        is_unwrapped = True
        for i in range(1, natoms):
            for j in range(3):
                if fabs(coordinates[i, j] - center[j]) >= half_box[j]:
                    is_unwrapped = False
                    break
            if not is_unwrapped:
                break
        if is_unwrapped:
            return np.array(coordinates)
        for i in range(3):
            inverse_box[i] = 1.0 / box[i]
    else:
        from .mdamath import triclinic_vectors
        tri_box = triclinic_vectors(box)

    unwrapped = np.zeros((natoms, 3), dtype=np.float32)
    for atom_i in range(natoms):
        for i in range(3):
            # Draw vector from center to atom
            vec[i] = coordinates[atom_i, i] - center[i]
            # Apply periodic boundary conditions to this vector
            if ortho:
                minimum_image_ortho(&vec[0], &box_[0], &inverse_box[0])
            else:
                minimum_image_triclinic(&vec[0], &tri_box[0, 0])
            # Then define position of atom based on this vector
            for i in range(3):
                unwrapped[atom_i, i] = center[i] + vec[i]
    return np.array(unwrapped)