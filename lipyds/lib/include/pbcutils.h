#ifndef __PBCUTILS_H
#define __PBCUTILS_H

#include <math.h>
#include <float.h>

typedef float coordinate[3];

#ifdef PARALLEL
#include <omp.h>
#define USED_OPENMP 1
#else
#define USED_OPENMP 0
#endif

inline double norm(float *a)
{
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

inline double dot(float *a, float *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double cosine(float degrees)
{
    double DEG2RAD = M_PI / 180;
    double radians;
    if (degrees == 90)
    {
        return 0;
    }
    else
    {
        radians = degrees * DEG2RAD;
        return cos(radians);
    }
}

void minimum_image_ortho(double *x, float *box, float *inverse_box)
{ // COPIED FROM MDANALYSIS.LIB.INCLUDE.CALC_DISTANCES
    int i;
    double s;
    for (i = 0; i < 3; i++)
    {
        if (box[i] > FLT_EPSILON)
        {
            s = inverse_box[i] * x[i];
            x[i] = box[i] * (s - round(s));
        }
    }
}

void minimum_image_triclinic(double *dx, float *box)
{ // COPIED FROM MDANALYSIS.LIB.INCLUDE.CALC_DISTANCES
    /*
    * Minimum image convention for triclinic systems, modelled after domain.cpp
    * in LAMMPS.
    * Assumes that there is a maximum separation of 1 box length (enforced in
    * dist functions by moving all particles to inside the box before
    * calculating separations).
    * Assumes box having zero values for box[1], box[2] and box[5]:
    *   /  a_x   0    0   \                 /  0    1    2  \
    *   |  b_x  b_y   0   |       indices:  |  3    4    5  |
    *   \  c_x  c_y  c_z  /                 \  6    7    8  /
    */
    double dx_min[3] = {0.0, 0.0, 0.0};
    double dsq_min = FLT_MAX;
    double dsq;
    double rx;
    double ry[2];
    double rz[3];
    int ix, iy, iz;
    for (ix = -1; ix < 2; ++ix)
    {
        rx = dx[0] + box[0] * ix;
        for (iy = -1; iy < 2; ++iy)
        {
            ry[0] = rx + box[3] * iy;
            ry[1] = dx[1] + box[4] * iy;
            for (iz = -1; iz < 2; ++iz)
            {
                rz[0] = ry[0] + box[6] * iz;
                rz[1] = ry[1] + box[7] * iz;
                rz[2] = dx[2] + box[8] * iz;
                dsq = rz[0] * rz[0] + rz[1] * rz[1] + rz[2] * rz[2];
                if (dsq < dsq_min)
                {
                    dsq_min = dsq;
                    dx_min[0] = rz[0];
                    dx_min[1] = rz[1];
                    dx_min[2] = rz[2];
                }
            }
        }
    }
    dx[0] = dx_min[0];
    dx[1] = dx_min[1];
    dx[2] = dx_min[2];
}

static void _box_to_triclinic_vectors(float *box, float *triclinic_vectors)
{
    /*
    *   /  a_x   0    0   \                 /  0    1    2  \
    *   |  b_x  b_y   0   |       indices:  |  3    4    5  |
    *   \  c_x  c_y  c_z  /                 \  6    7    8  /
    */
    int i;
    float lx, ly, lz, a, b, g;
    double DEG2RAD = M_PI / 180;
    double cos_a, cos_b, cos_g, sin_g, cx, cy, cz;
    lx = box[0];
    ly = box[1];
    lz = box[2];
    a = box[3];
    b = box[4];
    g = box[5];

    for (i = 0; i < 9; i++)
    {
        triclinic_vectors[i] = 0;
    }

    for (i = 0; i < 3; i++)
    {
        // invalid box
        if (box[i] <= 0 || box[i + 3] > 180)
        {
            return;
        }
    }

    if (a == 90 && b == 90 && g == 90)
    {
        // diagonal
        triclinic_vectors[0] = lx;
        triclinic_vectors[4] = ly;
        triclinic_vectors[8] = lz;
        return;
    }

    cos_a = cosine(a);
    cos_b = cosine(b);
    cos_g = cosine(g);
    if (g == 90)
    {
        sin_g = 1;
    }
    else
    {
        sin_g = sin(DEG2RAD * g);
    }

    cx = lz * cos_b;
    cy = lz * (cos_a - cos_b * cos_g) / sin_g;
    cz = sqrt(lz * lz - cx * cx - cy * cy);
    if (cz > 0)
    {
        // if triangle inequality does not hold, discriminant cz is negative
        // or zero or nan
        triclinic_vectors[0] = lx;
        triclinic_vectors[3] = ly * cos_g;
        triclinic_vectors[4] = ly * sin_g;
        triclinic_vectors[6] = cx;
        triclinic_vectors[7] = cy;
        triclinic_vectors[8] = cz;
    }
    return;
}

static void _single_mean_unwrap_around_ortho(coordinate *coords, int numcoords,
                                             float *center,
                                             float *box, float *inverse_box,
                                             float *output)
{
    int i, j;
    double displacement[3];
    double vector[3];
    double unit_image[3];

    for (i = 0; i < 3; i++)
    {
        unit_image[i] = center[i];
        output[i] = 0;
    }
    // minimum_image_ortho(unit_image, box, inverse_box);

    for (i = 0; i < 3; i++)
    {
        displacement[i] = center[i] - unit_image[i];
    }

    for (j = 0; j < numcoords; j++)
    {
        for (i = 0; i < 3; i++)
        {
            vector[i] = coords[j][i] - unit_image[i];
        }
        minimum_image_ortho(vector, box, inverse_box);
        for (i = 0; i < 3; i++)
        {
            output[i] += vector[i];
        }
    }
    for (i = 0; i < 3; i++)
    {
        output[i] /= numcoords;
        output[i] += unit_image[i];
        output[i] += displacement[i];
    }
}

static void _single_mean_unwrap_around_triclinic(coordinate *coords, int numcoords,
                                                 float *center,
                                                 float *triclinic_box,
                                                 float *output)
{
    int i, j;
    double displacement[3];
    double vector[3];
    double unit_image[3];

    for (i = 0; i < 3; i++)
    {
        unit_image[i] = center[i];
        output[i] = 0;
    }
    minimum_image_triclinic(unit_image, triclinic_box);

    for (i = 0; i < 3; i++)
    {
        displacement[i] = center[i] - unit_image[i];
    }

    for (j = 0; j < numcoords; j++)
    {
        for (i = 0; i < 3; i++)
        {
            vector[i] = coords[j][i] - unit_image[i];
        }
        minimum_image_triclinic(vector, triclinic_box);
        for (i = 0; i < 3; i++)
        {
            output[i] += vector[i];
        }
    }
    for (i = 0; i < 3; i++)
    {
        output[i] /= numcoords;
        output[i] += unit_image[i];
        output[i] += displacement[i];
    }
}

static void _mean_unwrap_around_centers_ortho(coordinate *coords,
                                              coordinate *centers,
                                              int numcoords,
                                              long *resindices,
                                              float *box,
                                              coordinate *output)
{
    int i, i_coord, n_index;
    int i_center = 0;
    int start_index = 0;
    long rix = resindices[0];
    float inverse_box[3];

    for (i = 0; i < 3; i++)
    {
        inverse_box[i] = 1 / box[i];
    }

    for (i_coord = 1; i_coord < numcoords; i_coord++)
    {
        if (resindices[i_coord] != rix)
        {
            n_index = i_coord - start_index;
            _single_mean_unwrap_around_ortho(&coords[start_index],
                                             n_index,
                                             centers[i_center],
                                             box,
                                             inverse_box,
                                             output[i_center]);
            start_index = i_coord;
            i_center += 1;
            rix = resindices[i_coord];
        }
    }
    n_index = numcoords - start_index;
    _single_mean_unwrap_around_ortho(&coords[start_index],
                                     n_index,
                                     centers[i_center],
                                     box,
                                     inverse_box,
                                     output[i_center]);
}

static void _mean_unwrap_around_centers_triclinic(coordinate *coords,
                                                  coordinate *centers,
                                                  int numcoords,
                                                  long *resindices,
                                                  float *box,
                                                  coordinate *output)
{
    int i_coord, n_index;
    int i_center = 0;
    int start_index = 0;
    long rix = resindices[0];
    float triclinic_box[9];

    _box_to_triclinic_vectors(box, triclinic_box);

    for (i_coord = 1; i_coord < numcoords; i_coord++)
    {
        if (resindices[i_coord] != rix)
        {
            n_index = i_coord - start_index;
            _single_mean_unwrap_around_triclinic(&coords[start_index],
                                                 n_index,
                                                 centers[i_center],
                                                 triclinic_box,
                                                 output[i_center]);
            start_index = i_coord;
            i_center += 1;
            rix = resindices[i_coord];
        }
    }
    n_index = numcoords - start_index;
    _single_mean_unwrap_around_triclinic(&coords[start_index],
                                         n_index,
                                         centers[i_center],
                                         triclinic_box,
                                         output[i_center]);
}

static void _mean_unwrap_around_first_ortho(coordinate *coords,
                                            int numcoords,
                                            long *resindices,
                                            float *box,
                                            coordinate *output)
{
    int i, i_coord, n_index;
    int i_center = 0;
    int start_index = 0;
    long rix = resindices[0];
    float inverse_box[3];

    for (i = 0; i < 3; i++)
    {
        inverse_box[i] = 1 / box[i];
    }

    for (i_coord = 1; i_coord < numcoords; i_coord++)
    {
        if (resindices[i_coord] != rix)
        {
            n_index = i_coord - start_index;
            _single_mean_unwrap_around_ortho(&coords[start_index],
                                             n_index,
                                             coords[start_index],
                                             box,
                                             inverse_box,
                                             output[i_center]);
            start_index = i_coord;
            i_center += 1;
            rix = resindices[i_coord];
        }
    }

    n_index = numcoords - start_index;
    _single_mean_unwrap_around_ortho(&coords[start_index],
                                     n_index,
                                     coords[start_index],
                                     box,
                                     inverse_box,
                                     output[i_center]);
}

static void _mean_unwrap_around_first_triclinic(coordinate *coords,
                                                int numcoords,
                                                long *resindices,
                                                float *box,
                                                coordinate *output)
{
    int i_coord, n_index;
    int i_center = 0;
    int start_index = 0;
    long rix = resindices[0];
    float triclinic_box[9];

    _box_to_triclinic_vectors(box, triclinic_box);
    for (i_coord = 1; i_coord < numcoords; i_coord++)
    {
        if (resindices[i_coord] != rix)
        {
            n_index = i_coord - start_index;
            _single_mean_unwrap_around_triclinic(&coords[start_index],
                                                 n_index,
                                                 coords[start_index],
                                                 triclinic_box,
                                                 output[i_center]);
            start_index = i_coord;
            i_center += 1;
            rix = resindices[i_coord];
        }
    }

    n_index = numcoords - start_index;
    _single_mean_unwrap_around_triclinic(&coords[start_index],
                                         n_index,
                                         coords[start_index],
                                         triclinic_box,
                                         output[i_center]);
}

static void _single_unwrap_around_ortho(coordinate *coords, int numcoords,
                                        float *center,
                                        float *box, float *inverse_box,
                                        coordinate *output)
{
    int i, j;
    double displacement[3];
    double vector[3];
    double unit_image[3];

    for (i = 0; i < 3; i++)
    {
        unit_image[i] = center[i];
    }
    // minimum_image_ortho(unit_image, box, inverse_box);

    for (i = 0; i < 3; i++)
    {
        displacement[i] = center[i] - unit_image[i];
    }

    for (j = 0; j < numcoords; j++)
    {
        for (i = 0; i < 3; i++)
        {
            output[j][i] = 42;
            vector[i] = coords[j][i] - unit_image[i];
        }
        minimum_image_ortho(vector, box, inverse_box);
        for (i = 0; i < 3; i++)
        {
            output[j][i] = unit_image[i] + vector[i]; // + displacement[i];
        }
    }
}

static void _single_unwrap_around_triclinic(coordinate *coords,
                                            int numcoords,
                                            float *center,
                                            float *triclinic_box,
                                            coordinate *output)
{
    int i, j;
    double displacement[3];
    double vector[3];
    double unit_image[3];

    for (i = 0; i < 3; i++)
    {
        unit_image[i] = center[i];
    }
    // minimum_image_triclinic(unit_image, triclinic_box);

    for (i = 0; i < 3; i++)
    {
        displacement[i] = center[i] - unit_image[i];
    }

    for (j = 0; j < numcoords; j++)
    {
        for (i = 0; i < 3; i++)
        {
            vector[i] = coords[j][i] - unit_image[i];
        }
        minimum_image_triclinic(vector, triclinic_box);
        for (i = 0; i < 3; i++)
        {
            output[j][i] = unit_image[i] + vector[i]; // + displacement[i];
        }
    }
}

#endif

static void _calc_cosine_similarity(float *a,
                                    coordinate *bs,
                                    int n_bs,
                                    double *cosines)
{
    int i;
    double norm_b, norm_ab;
    double norm_a = norm(a);
    for (i = 0; i < n_bs; i++)
    {
        norm_b = norm(bs[i]);
        norm_ab = norm_a * norm_b;
        if (norm_ab > 0)
        {
            cosines[i] = dot(a, bs[i]) / norm_ab;
        }
        else
        {
            cosines[i] = 1;
        }
    }
}

static inline void _copy_coordinates(coordinate *coordinates,
                                     int *neighbor_indices,
                                     int n_neighbors,
                                     coordinate *wrapped)
{
    int i, j;
    for (i = 0; i < n_neighbors; i++)
    {
        for (j = 0; j < 3; j++)
        {
            wrapped[i][j] = coordinates[neighbor_indices[i]][j];
        }
    }
}

static inline void _translate_project(coordinate *coordinates,
                                      coordinate *orientations,
                                      int center_index,
                                      int n_neighbors,
                                      float angle_factor,
                                      double *distances,
                                      coordinate *unwrapped)
{
    int i, j;
    double angles[n_neighbors];
    double projection;
    for (i = 0; i < n_neighbors; i++)
    {
        for (j = 0; j < 3; j++)
        {
            unwrapped[i][j] -= coordinates[center_index][j];
        }
    }

    _calc_cosine_similarity(orientations[center_index], unwrapped, n_neighbors, angles);
    for (i = 0; i < n_neighbors; i++)
    {
        projection = abs(distances[i] * angles[i]) * angle_factor;
        distances[i] += projection;
    }
}

static void _single_project_distances_nobox(coordinate *coordinates,
                                            coordinate *orientations,
                                            int center_index,
                                            int *neighbor_indices,
                                            int n_neighbors,
                                            float angle_factor,
                                            double *distances)
{
    coordinate wrapped[n_neighbors];

    _copy_coordinates(coordinates, neighbor_indices, n_neighbors, wrapped);
    _translate_project(coordinates, orientations, center_index, n_neighbors,
                       angle_factor, distances, wrapped);
}

static void _single_project_distances_ortho(coordinate *coordinates,
                                            coordinate *orientations,
                                            int center_index,
                                            int *neighbor_indices,
                                            int n_neighbors,
                                            float *box,
                                            float *inverse_box,
                                            float angle_factor,
                                            double *distances)
{
    coordinate wrapped[n_neighbors];
    coordinate unwrapped[n_neighbors];

    _copy_coordinates(coordinates, neighbor_indices, n_neighbors, wrapped);
    _single_unwrap_around_ortho(wrapped, n_neighbors, coordinates[center_index],
                                box, inverse_box, unwrapped);
    _translate_project(coordinates, orientations, center_index, n_neighbors,
                       angle_factor, distances, unwrapped);
}

static void _single_project_distances_triclinic(coordinate *coordinates,
                                                coordinate *orientations,
                                                int center_index,
                                                int *neighbor_indices,
                                                int n_neighbors,
                                                float *triclinic_box,
                                                float angle_factor,
                                                double *distances)
{
    coordinate wrapped[n_neighbors];
    coordinate unwrapped[n_neighbors];

    _copy_coordinates(coordinates, neighbor_indices, n_neighbors, wrapped);
    _single_unwrap_around_triclinic(wrapped, n_neighbors, coordinates[center_index],
                                    triclinic_box, unwrapped);
    _translate_project(coordinates, orientations, center_index, n_neighbors,
                       angle_factor, distances, unwrapped);
}

static void _project_distances_ortho(coordinate *coordinates,
                                     coordinate *orientations,
                                     int *index_as,
                                     int *index_bs,
                                     double *distances,
                                     int n_pairs,
                                     float *box,
                                     float angle_factor)
{
    int i, n_neighbors;
    float inverse_box[3];
    int start_index = 0;
    int center_index = index_as[0];

    for (i = 0; i < 3; i++)
    {
        inverse_box[i] = 1 / box[i];
    }

    for (i = 1; i < n_pairs; i++)
    {
        if (index_as[i] != center_index)
        {
            n_neighbors = i - start_index - 1;
            start_index += 1;

            _single_project_distances_ortho(coordinates, orientations, center_index,
                                            &index_bs[start_index], n_neighbors, box,
                                            inverse_box, angle_factor,
                                            &distances[start_index]);

            start_index = i;
            center_index = index_as[i];
        }
    }
    n_neighbors = n_pairs - start_index - 1;
    start_index += 1;
    _single_project_distances_ortho(coordinates, orientations, center_index,
                                    &index_bs[start_index], n_neighbors, box,
                                    inverse_box, angle_factor,
                                    &distances[start_index]);
}

static void _project_distances_triclinic(coordinate *coordinates,
                                         coordinate *orientations,
                                         int *index_as,
                                         int *index_bs,
                                         double *distances,
                                         int n_pairs,
                                         float *box,
                                         float angle_factor)
{
    int i, n_neighbors;
    int start_index = 0;
    int center_index = index_as[0];
    float triclinic_box[9];

    _box_to_triclinic_vectors(box, triclinic_box);

    for (i = 1; i < n_pairs; i++)
    {
        if (index_as[i] != center_index)
        {
            n_neighbors = i - start_index - 1;
            start_index += 1;

            _single_project_distances_triclinic(coordinates, orientations, center_index,
                                                &index_bs[start_index], n_neighbors,
                                                triclinic_box, angle_factor,
                                                &distances[start_index]);

            start_index = i;
            center_index = index_as[i];
        }
    }
    n_neighbors = n_pairs - start_index - 1;
    start_index += 1;
    _single_project_distances_triclinic(coordinates, orientations, center_index,
                                        &index_bs[start_index], n_neighbors,
                                        triclinic_box, angle_factor,
                                        &distances[start_index]);
}

static void _project_distances_nobox(coordinate *coordinates,
                                     coordinate *orientations,
                                     int *index_as,
                                     int *index_bs,
                                     double *distances,
                                     int n_pairs,
                                     float angle_factor)
{
    int i, n_neighbors;
    int start_index = 0;
    int center_index = index_as[0];

    for (i = 1; i < n_pairs; i++)
    {
        if (index_as[i] != center_index)
        {
            n_neighbors = i - start_index - 1;
            start_index += 1;

            _single_project_distances_nobox(coordinates, orientations, center_index,
                                            &index_bs[start_index], n_neighbors,
                                            angle_factor,
                                            &distances[start_index]);

            start_index = i;
            center_index = index_as[i];
        }
    }
    n_neighbors = n_pairs - start_index - 1;
    start_index += 1;
    _single_project_distances_nobox(coordinates, orientations, center_index,
                                    &index_bs[start_index], n_neighbors,
                                    angle_factor,
                                    &distances[start_index]);
}

static void _unwrap_around_ortho(coordinate *coords,
                                 int numcoords,
                                 float *center,
                                 float *box,
                                 coordinate *output)
{
    int i;
    float inverse_box[3];

    for (i = 0; i < 3; i++)
    {
        inverse_box[i] = 1 / box[i];
    }
    _single_unwrap_around_ortho(coords, numcoords, center,
                                box, inverse_box, output);
}

static void _unwrap_around_triclinic(coordinate *coords,
                                     int numcoords,
                                     float *center,
                                     float *box,
                                     coordinate *output)
{
    int i;
    float triclinic_box[9];

    _box_to_triclinic_vectors(box, triclinic_box);
    _single_unwrap_around_triclinic(coords, numcoords, center,
                                    triclinic_box, output);
}
