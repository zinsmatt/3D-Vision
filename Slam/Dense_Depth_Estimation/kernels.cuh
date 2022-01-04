#pragma once


void wrapper_update_cuda(const unsigned char* ref, const unsigned char* cur, double Tcr[3][4], double Trc[3][4], double *depth, double *cov2);