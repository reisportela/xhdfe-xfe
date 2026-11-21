#pragma once

struct NetworkDeviceInput {
    int n, nnz, periods, rhs, firms, budget;
    const int *ptr, *col, *index;
    const double *values, *border, *inverse, *b;
};
using NetworkProbe = bool (*)(void *, int, const double *);
using NetworkAccept = bool (*)(void *, int, const double *);
// Positive on acceptance, zero on memory admission, negative for charged work.
int solve_network_device(const NetworkDeviceInput &, void *, NetworkProbe, NetworkAccept);
