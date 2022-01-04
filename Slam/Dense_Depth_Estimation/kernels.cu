#include "kernels.cuh"

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include "constants.h"


__device__
inline double getBilinearInterpolatedValue_cuda(const unsigned char *img, double pt[2]) {
    const unsigned char* d = &img[(int)pt[1] * width + (int)pt[0]];
    double xx = pt[0] - floor(pt[0]);
    double yy = pt[1] - floor(pt[1]);
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[width]) +
            xx * yy * double(d[width + 1])) / 255.0;
}

__device__
inline void pix2cam_cuda(const double in[2], double out[3]) {
    out[0] = (in[0] - cx) / fx;
    out[1] =  (in[1] - cy) / fy;
    out[2] = 1.0;
}

__device__
inline void cam2pix_cuda(const double in[3], double out[2]) {
    out[0] = in[0] * fx / in[2] + cx;
    out[1] = in[1] * fy / in[2] + cy;
}

__device__
double norm3_cuda(const double in[3])
{
    return sqrt(in[0]*in[0] + in[1]*in[1] + in[2]*in[2]);
}

__device__
double norm2_cuda(const double in[2])
{
    return sqrt(in[0]*in[0] + in[1]*in[1]);
}


// inplace normalization vec 3
__device__
inline void normalize3_cuda(double in_out[3]) {
    double d = sqrt(in_out[0]*in_out[0] 
                    + in_out[1]*in_out[1]
                    + in_out[2]*in_out[2]);
    in_out[0] /= d;
    in_out[1] /= d;
    in_out[2] /= d;
}

// inplace normalization vec 2
__device__
inline void normalize2_cuda(double in_out[2]) {
    double d = sqrt(in_out[0]*in_out[0] + in_out[1]*in_out[1]);
    in_out[0] /= d;
    in_out[1] /= d;
}

__device__
void transform_cuda(double x[3], const double T[12], double out[3])
{
    for (int i = 0; i < 3; ++i)
    {
        out[i] = x[0] * T[i*4] + x[1] * T[i*4+1] + x[2] * T[i*4+2] +  T[i*4+3];
    }
}


__device__
double ZNCC_cuda(const unsigned char *im1, const double pt1[2], const unsigned char *im2, const double pt2[2])
{
    // no need to consider block partly outside because of boarder
    double v1[ncc_area], v2[ncc_area];
    double s1 = 0.0, s2 = 0.0;
    int idx = 0;
    for (int i = -ncc_window_size; i <= ncc_window_size; ++i)
    {
        for (int j = -ncc_window_size; j <= ncc_window_size; ++j)
        {
            double val_1 = ((double) im1[((int)pt1[1] + i) * width + (int)pt1[0] + j]) / 255;
            double temp_p2[2] = {pt2[0] + j, pt2[1] + i};
            double val_2 = getBilinearInterpolatedValue_cuda(im2, temp_p2);
            s1 += val_1;
            s2 += val_2;
            v1[idx] = val_1;
            v2[idx] = val_2;
            ++idx;
        }
    }

    double mean_1 = s1 / ncc_area;
    double mean_2 = s2 / ncc_area;

    double numerator = 0.0;
    double den1 = 0.0, den2 = 0.0;
    for (int i = 0; i < ncc_area; ++i)
    {
        double zv1 = v1[i] - mean_1;
        double zv2 = v2[i] - mean_2;
        numerator += zv1*zv2;
        den1 += zv1 * zv1;
        den2 += zv2 * zv2;
    }
    auto zncc =  numerator / (sqrt(den1 * den2 + epsilon));
    // std::cout << "zncc = " << zncc << "\n";
    return zncc;
}

__device__
bool epipolar_search_cuda(const unsigned char* ref, const unsigned char* cur, 
                          const double Tcr[12], const double pt[2],
                          double depth_mu, double depth_sigma2, 
                          double best_pc[2], double epipolar_dir[2])
{

    double depth_sigma = sqrt(depth_sigma2);
    double dmax = depth_mu + 3 * depth_sigma;
    double dmin = depth_mu - 3 * depth_sigma;
    dmin = max(0.1, dmin);

    double pn[3];
    pix2cam_cuda(pt, pn);
    normalize3_cuda(pn);
    double P_max[3] = {pn[0] * dmax, pn[1] * dmax, pn[2] * dmax};
    double P_min[3] = {pn[0] * dmin, pn[1] * dmin, pn[2] * dmin};
    double P_mu[3] = {pn[0] * depth_mu, pn[1] * depth_mu, pn[2] * depth_mu};

    double P_max_cur[3], P_min_cur[3], P_mu_cur[3];
    transform_cuda(P_max, Tcr, P_max_cur);
    transform_cuda(P_min, Tcr, P_min_cur);
    transform_cuda(P_mu, Tcr, P_mu_cur);


    double pc_max[2], pc_min[2], pc_mu[2];
    cam2pix_cuda(P_max_cur, pc_max);
    cam2pix_cuda(P_min_cur, pc_min);
    cam2pix_cuda(P_mu_cur, pc_mu);


    double epipolar_line[2] = {pc_max[0] - pc_min[0], pc_max[1] - pc_min[1]};
    epipolar_dir[0] = epipolar_line[0];
    epipolar_dir[1] = epipolar_line[1];
    normalize2_cuda(epipolar_dir);
    double epipolar_line_norm = norm2_cuda(epipolar_line);

    // double step = 0.7;
    // int nb_samples = std::ceil(epipolar_line.norm() / step);

    double half_range = 0.5 * epipolar_line_norm;
    if (half_range > 100) half_range = 100;

    double best_zncc = -1.0;
    for (double l = -half_range; l<= half_range; l+= 0.7)
    {
        double p[2] = {pc_mu[0] + l * epipolar_dir[0], pc_mu[1] + l * epipolar_dir[1]};

        if (p[0] < boarder || p[0] >= width-boarder || p[1] < boarder || p[1] >= height-boarder)
            continue; // p is outside the cur image

        double zncc = ZNCC_cuda(ref, pt, cur, p);
        if (zncc > best_zncc)
        {
            best_zncc = zncc;
            best_pc[0] = p[0];
            best_pc[1] = p[1];
        }
    }
    if (best_zncc < 0.85)
        return false;
    else
        return true;
}

__device__
double dot3_cuda(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__
double det2_cuda(const double A[2][2])
{
    return A[0][0] * A[1][1] - A[1][0] * A[0][1];
}

__device__
void solve_Axb2_cuda(const double A[2][2], const double b[2], double res[2])
{
    double det_inv = 1.0 / det2_cuda(A);
    double A_inv[2][2];
    A_inv[0][0] = det_inv * A[1][1];
    A_inv[0][1] = -det_inv * A[0][1];
    A_inv[1][0] = -det_inv * A[1][0];
    A_inv[1][1] = det_inv * A[0][0];

    res[0] = A_inv[0][0] * b[0] + A_inv[0][1] * b[1];
    res[1] = A_inv[1][0] * b[0] + A_inv[1][1] * b[1];
}

__device__
void update_depth_filter_cuda(const double pr[2], const double pc[2], const double Trc[12], const double epipolar_dir[2], double *depth, double *cov2)
{
    double fr[3];
    pix2cam_cuda(pr, fr);
    normalize3_cuda(fr);

    double fc[3];
    pix2cam_cuda(pc, fc);
    normalize3_cuda(fc);
    
    double f2[3] = {dot3_cuda(Trc, fc),
                    dot3_cuda(Trc+4, fc),
                    dot3_cuda(Trc+8, fc)};

    double trc[3] = {Trc[3], Trc[7], Trc[11]};
    double A[2][2];
    double b[2];

    A[0][0] = dot3_cuda(fr, fr);
    A[0][1] = dot3_cuda(fr, f2);
    A[1][0] = dot3_cuda(f2, fr);
    A[1][1] = dot3_cuda(f2, f2);
    A[0][1] *= -1;
    A[1][1] *= -1;
    
    b[0] = dot3_cuda(fr, trc);
    b[1] = dot3_cuda(f2, trc);

    if (abs(det2_cuda(A)) < 1e-20) // not invertible
        return;

    double res[2];
    solve_Axb2_cuda(A, b, res);
    double P1[3] = {fr[0] * res[0], fr[1] * res[0], fr[2] * res[0]};
    double P2[3] = {trc[0] + fc[0] * res[1], trc[1] + fc[1] * res[1], trc[2] + fc[2] * res[1]};
    double P_est[3] = {(P1[0] + P2[0]) * 0.5, 
                       (P1[1] + P2[1]) * 0.5, 
                       (P1[2] + P2[2]) * 0.5};
    double depth_obs = norm3_cuda(P_est);

    double P[3] = {fr[0] * depth_obs, fr[1] * depth_obs, fr[2] * depth_obs};
    double a[3] = {P[0] - trc[0], P[1] - trc[1], P[2] - trc[2]};

    double t[3] = {trc[0], trc[1], trc[2]};
    normalize3_cuda(t);

    double alpha = acos(dot3_cuda(fr, t));
    double beta = acos(-dot3_cuda(a, t) / norm3_cuda(a));

    double pc2[2] = {pc[0] + epipolar_dir[0], pc[1] + epipolar_dir[1]};
    double fc2[3];
    pix2cam_cuda(pc2, fc2);
    normalize3_cuda(fc2);
    double beta_2 = acos(-dot3_cuda(fc2, t));

    double gamma = M_PI - alpha - beta_2;
    double d_noise = norm3_cuda(trc) * sin(beta_2) / sin(gamma); // sinus law
    double sigma_obs = depth_obs - d_noise;
    double sigma2_obs = sigma_obs * sigma_obs;


    // Depth fusion
    double d = depth[(int)pr[1] * width + (int)pr[0]];
    double sigma2 = cov2[(int)pr[1] * width + (int)pr[0]];

    double d_fused = (sigma2_obs * d + sigma2 * depth_obs) / (sigma2 + sigma2_obs);
    double sigma2_fused = (sigma2 * sigma2_obs) / (sigma2 + sigma2_obs);

    depth[(int)pr[1] * width + (int)pr[0]] = d_fused;
    cov2[(int)pr[1] * width + (int)pr[0]] = sigma2_fused;

}


__device__ double Tcr_global[12];
__device__ double Trc_global[12];


__global__
void process_pixel_cuda(const unsigned char* ref, const unsigned char* cur, double *depth, double *cov2)
{

    int j = boarder + (blockIdx.x * blockDim.x) + threadIdx.x;
    int i = boarder + (blockIdx.y * blockDim.y) + threadIdx.y;

    double depth_mu = depth[i*width+j];
    double depth_sigma2 = cov2[i*width+j];

    if (depth_sigma2 < min_cov || depth_sigma2 > max_cov)
        return;

    double pr[2] = {(double)j, (double)i};
    double pc[2];
    double epipolar_dir[2];

    bool found = epipolar_search_cuda(ref, cur, Tcr_global, pr, depth_mu, depth_sigma2, pc, epipolar_dir);
    
    if (!found)
        return;

    update_depth_filter_cuda(pr, pc, Trc_global, epipolar_dir, depth, cov2);
}


void wrapper_update_cuda(const unsigned char* ref, const unsigned char* cur, double Tcr[3][4], double Trc[3][4], double *depth, double *cov2)
{

    size_t size_uchar = sizeof(unsigned char) * width * height;
    size_t size_double = sizeof(double) * width * height;
    unsigned char *ref_cuda, *cur_cuda;
    cudaMalloc(&ref_cuda, size_uchar);
    cudaMalloc(&cur_cuda, size_uchar);

    double *depth_cuda, *cov2_cuda;
    cudaMalloc(&depth_cuda, size_double);
    cudaMalloc(&cov2_cuda,  size_double);

    cudaMemcpy(ref_cuda, ref, size_uchar, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_cuda, cur, size_uchar, cudaMemcpyHostToDevice);
    
    cudaMemcpy(depth_cuda, depth, size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(cov2_cuda, cov2, size_double, cudaMemcpyHostToDevice);


    int A = 480 - 2 * boarder; // height
    int B = 640 - 2 * boarder; // width

    dim3 block_dim(16, 16);
    dim3 grid_dim(B / 16 + 1, A / 16 + 1);
    // std::cout << "grid_dim " << grid_dim.x << " " << grid_dim.y << "\n";
    // std::cout << "block_dim " << block_dim.x << " " << block_dim.y << "\n";
   

    cudaMemcpyToSymbol(Tcr_global, &Tcr[0][0], 12 * sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Trc_global, &Trc[0][0], 12 * sizeof(double), 0, cudaMemcpyHostToDevice);

    process_pixel_cuda<<<grid_dim, block_dim>>>(ref_cuda, cur_cuda, depth_cuda, cov2_cuda);
    cudaDeviceSynchronize();


    cudaMemcpy(depth, depth_cuda, size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(cov2, cov2_cuda, size_double, cudaMemcpyDeviceToHost);


    cudaFree(ref_cuda);
    cudaFree(cur_cuda);
    cudaFree(depth_cuda);
    cudaFree(cov2_cuda);
}
