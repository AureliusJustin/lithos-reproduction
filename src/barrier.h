/*
 * barrier.h — lookup for the generated ordering forwarders (see barrier.def).
 */
#ifndef LITHOS_BARRIER_H
#define LITHOS_BARRIER_H

typedef struct BarrierFn {
    const char* name;      /* exported driver symbol, e.g. "cuMemcpyHtoD_v2" */
    void*       wrapper;   /* our forwarder                                   */
    void*       real;      /* the driver's, once cuGetProcAddress reveals it  */
} BarrierFn;

/* Record for `name`, or NULL if it is not one of ours. Accepts the unversioned
 * name that cuGetProcAddress is asked for as well as the exported symbol. */
BarrierFn* barrier_lookup(const char* name);

#endif /* LITHOS_BARRIER_H */
