/* Test-only MIOpen ABI stub for the cuDNN bridge. SPDX-License-Identifier: MIT */
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct StubHandle { void *stream; } StubHandle;
typedef struct StubTensor {
    int type;
    int nb;
    int dims[8];
    int strides[8];
} StubTensor;
typedef struct StubConv {
    int spatial;
    int pad[3];
    int stride[3];
    int dilation[3];
    int mode;
    int groups;
} StubConv;

typedef struct Perf {
    int fwd_algo;
    float time;
    size_t memory;
} Perf;

typedef struct Solution {
    float time;
    size_t workspace_size;
    unsigned long long solution_id;
    int algorithm;
} Solution;

__declspec(dllexport) int miopenCreate(void **out) {
    StubHandle *h;
    if (!out) return 3;
    h = (StubHandle *)calloc(1, sizeof(*h));
    if (!h) return 4;
    *out = h;
    return 0;
}
__declspec(dllexport) int miopenDestroy(void *h) { free(h); return 0; }
__declspec(dllexport) int miopenSetStream(void *h, void *stream) {
    if (!h) return 3;
    ((StubHandle *)h)->stream = stream;
    return 0;
}
__declspec(dllexport) int miopenCreateTensorDescriptor(void **out) {
    StubTensor *d;
    if (!out) return 3;
    d = (StubTensor *)calloc(1, sizeof(*d));
    if (!d) return 4;
    *out = d;
    return 0;
}
__declspec(dllexport) int miopenDestroyTensorDescriptor(void *d) { free(d); return 0; }
__declspec(dllexport) int miopenSetTensorDescriptor(
    void *desc, int type, int nb, const int *dims, const int *strides) {
    StubTensor *d = (StubTensor *)desc;
    if (!d || !dims || !strides || nb <= 0 || nb > 8) return 3;
    d->type = type; d->nb = nb;
    memcpy(d->dims, dims, sizeof(int) * nb);
    memcpy(d->strides, strides, sizeof(int) * nb);
    return 0;
}
__declspec(dllexport) int miopenCreateConvolutionDescriptor(void **out) {
    StubConv *c;
    if (!out) return 3;
    c = (StubConv *)calloc(1, sizeof(*c));
    if (!c) return 4;
    c->groups = 1;
    *out = c;
    return 0;
}
__declspec(dllexport) int miopenDestroyConvolutionDescriptor(void *d) { free(d); return 0; }
__declspec(dllexport) int miopenInitConvolutionNdDescriptor(
    void *desc, int spatial, const int *pad, const int *stride, const int *dilation, int mode) {
    StubConv *c = (StubConv *)desc;
    if (!c || !pad || !stride || !dilation || spatial != 2 || mode != 0) return 6;
    c->spatial = spatial; c->mode = mode;
    memcpy(c->pad, pad, sizeof(int) * spatial);
    memcpy(c->stride, stride, sizeof(int) * spatial);
    memcpy(c->dilation, dilation, sizeof(int) * spatial);
    return 0;
}
__declspec(dllexport) int miopenSetConvolutionGroupCount(void *desc, int groups) {
    if (!desc || groups <= 0) return 3;
    ((StubConv *)desc)->groups = groups;
    return 0;
}
__declspec(dllexport) int miopenConvolutionForwardGetWorkSpaceSize(
    void *handle, void *w, void *x, void *conv, void *y, size_t *size) {
    (void)handle; (void)w; (void)x; (void)conv; (void)y;
    if (!size) return 3;
    *size = 0;
    return 0;
}
__declspec(dllexport) int miopenConvolutionForwardGetSolutionCount(
    void *handle, void *wDesc, void *xDesc, void *convDesc, void *yDesc, size_t *count) {
    (void)handle; (void)wDesc; (void)xDesc; (void)convDesc; (void)yDesc;
    if (!count) return 3;
    *count = 1;
    return 0;
}
__declspec(dllexport) int miopenConvolutionForwardGetSolution(
    void *handle, void *wDesc, void *xDesc, void *convDesc, void *yDesc,
    size_t max_count, size_t *returned, Solution *solutions) {
    (void)handle; (void)wDesc; (void)xDesc; (void)convDesc; (void)yDesc;
    if (!returned || !solutions || max_count < 1) return 3;
    memset(&solutions[0], 0, sizeof(solutions[0]));
    solutions[0].time = 0.01f;
    solutions[0].workspace_size = 0;
    solutions[0].solution_id = 1;
    solutions[0].algorithm = 5; /* implicit GEMM */
    *returned = 1;
    return 0;
}
__declspec(dllexport) int miopenConvolutionForwardGetSolutionWorkspaceSize(
    void *handle, void *wDesc, void *xDesc, void *convDesc, void *yDesc,
    unsigned long long solution_id, size_t *workspace_size) {
    (void)handle; (void)wDesc; (void)xDesc; (void)convDesc; (void)yDesc;
    if (!workspace_size || solution_id != 1) return 3;
    *workspace_size = 0;
    return 0;
}

__declspec(dllexport) int miopenFindConvolutionForwardAlgorithm(
    void *handle, void *xDesc, const void *x, void *wDesc, const void *w,
    void *convDesc, void *yDesc, void *y, int request, int *returned,
    Perf *perf, void *workspace, size_t workspace_size, bool exhaustive) {
    (void)handle; (void)xDesc; (void)x; (void)wDesc; (void)w; (void)convDesc;
    (void)yDesc; (void)y; (void)workspace; (void)workspace_size; (void)exhaustive;
    if (!returned || !perf || request <= 0) return 3;
    perf[0].fwd_algo = 5; /* MIOPEN_FWD_IMPLICIT_GEMM */
    perf[0].time = 0.01f;
    perf[0].memory = 0;
    *returned = 1;
    return 0;
}
static size_t off4(const StubTensor *d, int a, int b, int c, int e) {
    return (size_t)a*d->strides[0] + (size_t)b*d->strides[1] +
           (size_t)c*d->strides[2] + (size_t)e*d->strides[3];
}

static int stub_conv_core(
    void *xDescV, const void *xV, void *wDescV, const void *wV,
    void *convDescV, void *yDescV, void *yV, float alpha, float beta) {
    StubTensor *xD=(StubTensor *)xDescV, *wD=(StubTensor *)wDescV, *yD=(StubTensor *)yDescV;
    StubConv *cD=(StubConv *)convDescV;
    const float *x=(const float *)xV, *w=(const float *)wV;
    float *y=(float *)yV;
    int n,k,p,q,ci,r,s;
    if (!xD || !wD || !yD || !cD || !x || !w || !y) return 3;
    if (xD->type != 1 || wD->type != 1 || yD->type != 1 || cD->groups != 1) return 6;
    for(n=0;n<yD->dims[0];++n) for(k=0;k<yD->dims[1];++k)
    for(p=0;p<yD->dims[2];++p) for(q=0;q<yD->dims[3];++q) {
        float sum=0.0f;
        for(ci=0;ci<xD->dims[1];++ci) for(r=0;r<wD->dims[2];++r) for(s=0;s<wD->dims[3];++s) {
            int ih=p*cD->stride[0]-cD->pad[0]+r*cD->dilation[0];
            int iw=q*cD->stride[1]-cD->pad[1]+s*cD->dilation[1];
            if(ih>=0 && ih<xD->dims[2] && iw>=0 && iw<xD->dims[3]) {
                sum += x[off4(xD,n,ci,ih,iw)] * w[off4(wD,k,ci,r,s)];
            }
        }
        {
            size_t yo=off4(yD,n,k,p,q);
            y[yo]=alpha*sum + beta*y[yo];
        }
    }
    return 0;
}

static int stub_solution_count(size_t *count) {
    if (!count) return 3;
    *count = 1;
    return 0;
}

static int stub_solution_get(size_t max_count, size_t *returned, Solution *solutions,
                             unsigned long long solution_id) {
    if (!returned || !solutions || max_count < 1) return 3;
    memset(&solutions[0], 0, sizeof(solutions[0]));
    solutions[0].time = 0.02f;
    solutions[0].workspace_size = 0;
    solutions[0].solution_id = solution_id;
    solutions[0].algorithm = 5; /* implicit GEMM */
    *returned = 1;
    return 0;
}

static int stub_solution_workspace(unsigned long long got, unsigned long long expected,
                                   size_t *workspace_size) {
    if (!workspace_size || got != expected) return 3;
    *workspace_size = 0;
    return 0;
}

__declspec(dllexport) int miopenConvolutionBackwardDataGetSolutionCount(
    void *handle, void *dyDesc, void *wDesc, void *convDesc, void *dxDesc, size_t *count) {
    (void)handle; (void)dyDesc; (void)wDesc; (void)convDesc; (void)dxDesc;
    return stub_solution_count(count);
}

__declspec(dllexport) int miopenConvolutionBackwardDataGetSolution(
    void *handle, void *dyDesc, void *wDesc, void *convDesc, void *dxDesc,
    size_t max_count, size_t *returned, Solution *solutions) {
    (void)handle; (void)dyDesc; (void)wDesc; (void)convDesc; (void)dxDesc;
    return stub_solution_get(max_count, returned, solutions, 2);
}

__declspec(dllexport) int miopenConvolutionBackwardDataGetSolutionWorkspaceSize(
    void *handle, void *dyDesc, void *wDesc, void *convDesc, void *dxDesc,
    unsigned long long solution_id, size_t *workspace_size) {
    (void)handle; (void)dyDesc; (void)wDesc; (void)convDesc; (void)dxDesc;
    return stub_solution_workspace(solution_id, 2, workspace_size);
}

static int stub_bwd_data_core(
    void *dyDescV, const void *dyV, void *wDescV, const void *wV,
    void *convDescV, void *dxDescV, void *dxV) {
    StubTensor *dyD=(StubTensor *)dyDescV, *wD=(StubTensor *)wDescV, *dxD=(StubTensor *)dxDescV;
    StubConv *cD=(StubConv *)convDescV;
    const float *dy=(const float *)dyV, *w=(const float *)wV;
    float *dx=(float *)dxV;
    int n,k,p,q,ci,r,s;
    size_t total;
    if (!dyD || !wD || !dxD || !cD || !dy || !w || !dx) return 3;
    if (dyD->type != 1 || wD->type != 1 || dxD->type != 1 || cD->groups != 1) return 6;
    total=(size_t)dxD->dims[0]*dxD->dims[1]*dxD->dims[2]*dxD->dims[3];
    memset(dx,0,total*sizeof(float));
    for(n=0;n<dyD->dims[0];++n) for(k=0;k<dyD->dims[1];++k)
    for(p=0;p<dyD->dims[2];++p) for(q=0;q<dyD->dims[3];++q)
    for(ci=0;ci<dxD->dims[1];++ci) for(r=0;r<wD->dims[2];++r) for(s=0;s<wD->dims[3];++s) {
        int ih=p*cD->stride[0]-cD->pad[0]+r*cD->dilation[0];
        int iw=q*cD->stride[1]-cD->pad[1]+s*cD->dilation[1];
        if(ih>=0 && ih<dxD->dims[2] && iw>=0 && iw<dxD->dims[3]) {
            dx[off4(dxD,n,ci,ih,iw)] +=
                dy[off4(dyD,n,k,p,q)] * w[off4(wD,k,ci,r,s)];
        }
    }
    return 0;
}

__declspec(dllexport) int miopenConvolutionBackwardDataImmediate(
    void *handle, void *dyDescV, const void *dyV, void *wDescV, const void *wV,
    void *convDescV, void *dxDescV, void *dxV, void *workspace, size_t workspace_size,
    unsigned long long solution_id) {
    (void)handle; (void)workspace; (void)workspace_size;
    if (solution_id != 2) return 3;
    return stub_bwd_data_core(dyDescV,dyV,wDescV,wV,convDescV,dxDescV,dxV);
}

__declspec(dllexport) int miopenConvolutionBackwardWeightsGetSolutionCount(
    void *handle, void *dyDesc, void *xDesc, void *convDesc, void *dwDesc, size_t *count) {
    (void)handle; (void)dyDesc; (void)xDesc; (void)convDesc; (void)dwDesc;
    return stub_solution_count(count);
}

__declspec(dllexport) int miopenConvolutionBackwardWeightsGetSolution(
    void *handle, void *dyDesc, void *xDesc, void *convDesc, void *dwDesc,
    size_t max_count, size_t *returned, Solution *solutions) {
    (void)handle; (void)dyDesc; (void)xDesc; (void)convDesc; (void)dwDesc;
    return stub_solution_get(max_count, returned, solutions, 3);
}

__declspec(dllexport) int miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    void *handle, void *dyDesc, void *xDesc, void *convDesc, void *dwDesc,
    unsigned long long solution_id, size_t *workspace_size) {
    (void)handle; (void)dyDesc; (void)xDesc; (void)convDesc; (void)dwDesc;
    return stub_solution_workspace(solution_id, 3, workspace_size);
}

static int stub_bwd_weights_core(
    void *dyDescV, const void *dyV, void *xDescV, const void *xV,
    void *convDescV, void *dwDescV, void *dwV) {
    StubTensor *dyD=(StubTensor *)dyDescV, *xD=(StubTensor *)xDescV, *dwD=(StubTensor *)dwDescV;
    StubConv *cD=(StubConv *)convDescV;
    const float *dy=(const float *)dyV, *x=(const float *)xV;
    float *dw=(float *)dwV;
    int n,k,p,q,ci,r,s;
    size_t total;
    if (!dyD || !xD || !dwD || !cD || !dy || !x || !dw) return 3;
    if (dyD->type != 1 || xD->type != 1 || dwD->type != 1 || cD->groups != 1) return 6;
    total=(size_t)dwD->dims[0]*dwD->dims[1]*dwD->dims[2]*dwD->dims[3];
    memset(dw,0,total*sizeof(float));
    for(k=0;k<dwD->dims[0];++k) for(ci=0;ci<dwD->dims[1];++ci)
    for(r=0;r<dwD->dims[2];++r) for(s=0;s<dwD->dims[3];++s) {
        float sum=0.0f;
        for(n=0;n<dyD->dims[0];++n) for(p=0;p<dyD->dims[2];++p) for(q=0;q<dyD->dims[3];++q) {
            int ih=p*cD->stride[0]-cD->pad[0]+r*cD->dilation[0];
            int iw=q*cD->stride[1]-cD->pad[1]+s*cD->dilation[1];
            if(ih>=0 && ih<xD->dims[2] && iw>=0 && iw<xD->dims[3]) {
                sum += dy[off4(dyD,n,k,p,q)] * x[off4(xD,n,ci,ih,iw)];
            }
        }
        dw[off4(dwD,k,ci,r,s)] = sum;
    }
    return 0;
}

__declspec(dllexport) int miopenConvolutionBackwardWeightsImmediate(
    void *handle, void *dyDescV, const void *dyV, void *xDescV, const void *xV,
    void *convDescV, void *dwDescV, void *dwV, void *workspace, size_t workspace_size,
    unsigned long long solution_id) {
    (void)handle; (void)workspace; (void)workspace_size;
    if (solution_id != 3) return 3;
    return stub_bwd_weights_core(dyDescV,dyV,xDescV,xV,convDescV,dwDescV,dwV);
}

__declspec(dllexport) int miopenConvolutionForwardImmediate(
    void *handle, void *wDescV, const void *wV, void *xDescV, const void *xV,
    void *convDescV, void *yDescV, void *yV, void *workspace, size_t workspace_size,
    unsigned long long solution_id) {
    (void)handle; (void)workspace; (void)workspace_size;
    if (solution_id != 1) return 3;
    return stub_conv_core(xDescV, xV, wDescV, wV, convDescV, yDescV, yV, 1.0f, 0.0f);
}

__declspec(dllexport) int miopenConvolutionForward(
    void *handle, const void *alpha_ptr, void *xDescV, const void *xV,
    void *wDescV, const void *wV, void *convDescV, int algo,
    const void *beta_ptr, void *yDescV, void *yV, void *workspace, size_t workspace_size) {
    float alpha, beta;
    (void)handle; (void)workspace; (void)workspace_size;
    if (!alpha_ptr || !beta_ptr || algo != 5) return 3;
    alpha=*(const float *)alpha_ptr;
    beta=*(const float *)beta_ptr;
    return stub_conv_core(xDescV, xV, wDescV, wV, convDescV, yDescV, yV, alpha, beta);
}