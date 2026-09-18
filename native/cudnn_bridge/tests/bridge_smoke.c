/* ABI/correctness smoke for the experimental cuDNN bridge. SPDX-License-Identifier: MIT */
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

typedef int (*PF_create)(void **);
typedef int (*PF_destroy)(void *);
typedef int (*PF_set_stream)(void *, void *);
typedef int (*PF_create_desc)(void **);
typedef int (*PF_destroy_desc)(void *);
typedef int (*PF_set_tensor4d)(void *, int, int, int, int, int, int);
typedef int (*PF_set_filter4d)(void *, int, int, int, int, int, int);
typedef int (*PF_set_conv2d)(void *, int,int,int,int,int,int,int,int);
typedef int (*PF_outdim)(void *, void *, void *, int *, int *, int *, int *);
typedef struct CudnnFwdPerf {
    int algo;
    int status;
    float time;
    size_t memory;
    int determinism;
    int mathType;
    int reserved[3];
} CudnnFwdPerf;
typedef int (*PF_algo_max)(void *, int *);
typedef int (*PF_algo_v7)(void *, void *, void *, void *, void *, int, int *, CudnnFwdPerf *);
typedef int (*PF_workspace)(void *, void *, void *, void *, void *, int, size_t *);
typedef int (*PF_fwd)(void *, const void *, void *, const void *, void *, const void *,
                      void *, int, void *, size_t, const void *, void *, void *);
typedef const char *(*PF_error)(int);

#define LOAD(name, type) \
    type name = NULL; \
    do { \
        FARPROC p = GetProcAddress(m, #name); \
        if (!p) { fprintf(stderr, "missing %s\\n", #name); return 3; } \
        memcpy(&name, &p, sizeof(name)); \
    } while (0);

int main(int argc, char **argv) {
    HMODULE m;
    void *h=NULL,*xd=NULL,*wd=NULL,*yd=NULL,*cd=NULL;
    int rc,n,c,oh,ow,i,algo_max=0,algo_count=0;
    CudnnFwdPerf perf[5];
    size_t ws=999;
    float x[9]={1,2,3,4,5,6,7,8,9};
    float wgt[4]={1,0,0,-1};
    float y[4]={0,0,0,0};
    float dy[4]={1,1,1,1};
    float dx[9]={0,0,0,0,0,0,0,0,0};
    float dw[4]={0,0,0,0};
    const float dx_expected[9]={1,1,0,1,0,-1,0,-1,-1};
    const float dw_expected[4]={12,16,24,28};
    float alpha=1.0f,beta=0.0f;
    if(argc<2){fprintf(stderr,"usage: bridge_smoke <cudnn64_8.dll>\n");return 2;}
    m=LoadLibraryA(argv[1]);
    if(!m){fprintf(stderr,"LoadLibrary failed: %lu\n",GetLastError());return 2;}
    LOAD(cudnnCreate,PF_create)
    LOAD(cudnnDestroy,PF_destroy)
    LOAD(cudnnSetStream,PF_set_stream)
    LOAD(cudnnCreateTensorDescriptor,PF_create_desc)
    LOAD(cudnnDestroyTensorDescriptor,PF_destroy_desc)
    LOAD(cudnnSetTensor4dDescriptor,PF_set_tensor4d)
    LOAD(cudnnCreateFilterDescriptor,PF_create_desc)
    LOAD(cudnnDestroyFilterDescriptor,PF_destroy_desc)
    LOAD(cudnnSetFilter4dDescriptor,PF_set_filter4d)
    LOAD(cudnnCreateConvolutionDescriptor,PF_create_desc)
    LOAD(cudnnDestroyConvolutionDescriptor,PF_destroy_desc)
    LOAD(cudnnSetConvolution2dDescriptor,PF_set_conv2d)
    LOAD(cudnnGetConvolution2dForwardOutputDim,PF_outdim)
    LOAD(cudnnGetConvolutionForwardAlgorithmMaxCount,PF_algo_max)
    LOAD(cudnnGetConvolutionForwardAlgorithm_v7,PF_algo_v7)
    LOAD(cudnnGetConvolutionForwardWorkspaceSize,PF_workspace)
    LOAD(cudnnConvolutionForward,PF_fwd)
    LOAD(cudnnGetConvolutionBackwardDataAlgorithmMaxCount,PF_algo_max)
    LOAD(cudnnGetConvolutionBackwardDataAlgorithm_v7,PF_algo_v7)
    LOAD(cudnnGetConvolutionBackwardDataWorkspaceSize,PF_workspace)
    LOAD(cudnnConvolutionBackwardData,PF_fwd)
    LOAD(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount,PF_algo_max)
    LOAD(cudnnGetConvolutionBackwardFilterAlgorithm_v7,PF_algo_v7)
    LOAD(cudnnGetConvolutionBackwardFilterWorkspaceSize,PF_workspace)
    LOAD(cudnnConvolutionBackwardFilter,PF_fwd)
    LOAD(cudnnGetErrorString,PF_error)

    if((rc=cudnnCreate(&h))!=0){fprintf(stderr,"cudnnCreate: %s\n",cudnnGetErrorString(rc));return 4;}
    if((rc=cudnnSetStream(h,(void *)1))!=9){fprintf(stderr,"non-default stream must fail closed, got %d\n",rc);return 5;}
    if((rc=cudnnSetStream(h,NULL))!=0){fprintf(stderr,"default stream failed %d\n",rc);return 5;}

    if(cudnnCreateTensorDescriptor(&xd)||cudnnCreateTensorDescriptor(&yd)||
       cudnnCreateFilterDescriptor(&wd)||cudnnCreateConvolutionDescriptor(&cd)) return 6;
    if(cudnnSetTensor4dDescriptor(xd,0,0,1,1,3,3)) return 7; /* NCHW, float */
    if(cudnnSetFilter4dDescriptor(wd,0,0,1,1,2,2)) return 8;
    if((rc=cudnnSetConvolution2dDescriptor(cd,0,0,1,1,1,1,0,0))!=9) {
        fprintf(stderr,"true convolution must fail closed, got %d\n",rc); return 9;
    }
    if(cudnnSetConvolution2dDescriptor(cd,0,0,1,1,1,1,1,0)) return 10; /* cross correlation */
    if(cudnnGetConvolution2dForwardOutputDim(cd,xd,wd,&n,&c,&oh,&ow)) return 11;
    if(n!=1||c!=1||oh!=2||ow!=2){fprintf(stderr,"bad output dims %d %d %d %d\n",n,c,oh,ow);return 12;}
    if(cudnnSetTensor4dDescriptor(yd,0,0,n,c,oh,ow)) return 13;
    if(cudnnGetConvolutionForwardAlgorithmMaxCount(h,&algo_max) || algo_max!=5) {
        fprintf(stderr,"bad algorithm max count %d\n",algo_max); return 14;
    }
    memset(perf,0,sizeof(perf));
    if(cudnnGetConvolutionForwardAlgorithm_v7(h,xd,wd,cd,yd,5,&algo_count,perf)) {
        fprintf(stderr,"v7 algorithm query failed\n"); return 15;
    }
    if(algo_count!=1 || perf[0].algo!=0 || perf[0].status!=0 ||
       perf[0].memory!=0 || perf[0].determinism!=0) {
        fprintf(stderr,"unexpected v7 result count=%d algo=%d status=%d memory=%llu det=%d\n",
                algo_count,perf[0].algo,perf[0].status,
                (unsigned long long)perf[0].memory,perf[0].determinism);
        return 16;
    }
    if(cudnnGetConvolutionForwardWorkspaceSize(h,xd,wd,cd,yd,perf[0].algo,&ws)) return 17;
    if(ws!=0){fprintf(stderr,"stub workspace expected 0, got %llu\n",(unsigned long long)ws);return 18;}
    rc=cudnnConvolutionForward(h,&alpha,xd,x,wd,wgt,cd,perf[0].algo,NULL,0,&beta,yd,y);
    if(rc){fprintf(stderr,"forward failed: %s (%d)\n",cudnnGetErrorString(rc),rc);return 19;}
    for(i=0;i<4;++i){
        if(fabsf(y[i]-(-4.0f))>1e-6f){fprintf(stderr,"incorrect y[%d]=%f\n",i,y[i]);return 20;}
    }

    /* Backward data: dy=ones -> reference dx. */
    memset(perf,0,sizeof(perf));
    algo_max=0; algo_count=0; ws=999;
    if(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(h,&algo_max) || algo_max!=4) {
        fprintf(stderr,"bad backward-data max count %d\n",algo_max); return 21;
    }
    if(cudnnGetConvolutionBackwardDataAlgorithm_v7(h,wd,yd,cd,xd,5,&algo_count,perf)) {
        fprintf(stderr,"backward-data v7 query failed\n"); return 22;
    }
    if(algo_count<1 || perf[0].algo!=1 || perf[0].status!=0 || perf[0].determinism!=0) {
        fprintf(stderr,"unexpected backward-data candidate count=%d algo=%d status=%d det=%d\n",
                algo_count,perf[0].algo,perf[0].status,perf[0].determinism); return 23;
    }
    if(cudnnGetConvolutionBackwardDataWorkspaceSize(h,wd,yd,cd,xd,perf[0].algo,&ws) || ws!=0)
        return 24;
    rc=cudnnConvolutionBackwardData(h,&alpha,wd,wgt,yd,dy,cd,perf[0].algo,NULL,0,&beta,xd,dx);
    if(rc){fprintf(stderr,"backward-data failed: %s (%d)\n",cudnnGetErrorString(rc),rc);return 25;}
    for(i=0;i<9;++i){
        if(fabsf(dx[i]-dx_expected[i])>1e-6f){
            fprintf(stderr,"incorrect dx[%d]=%f expected=%f\n",i,dx[i],dx_expected[i]);return 26;
        }
    }

    /* Backward filter: dy=ones -> reference dw. */
    memset(perf,0,sizeof(perf));
    algo_max=0; algo_count=0; ws=999;
    if(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(h,&algo_max) || algo_max!=3) {
        fprintf(stderr,"bad backward-filter max count %d\n",algo_max); return 27;
    }
    if(cudnnGetConvolutionBackwardFilterAlgorithm_v7(h,xd,yd,cd,wd,5,&algo_count,perf)) {
        fprintf(stderr,"backward-filter v7 query failed\n"); return 28;
    }
    if(algo_count<1 || perf[0].algo!=1 || perf[0].status!=0 || perf[0].determinism!=0) {
        fprintf(stderr,"unexpected backward-filter candidate count=%d algo=%d status=%d det=%d\n",
                algo_count,perf[0].algo,perf[0].status,perf[0].determinism); return 29;
    }
    if(cudnnGetConvolutionBackwardFilterWorkspaceSize(h,xd,yd,cd,wd,perf[0].algo,&ws) || ws!=0)
        return 30;
    rc=cudnnConvolutionBackwardFilter(h,&alpha,xd,x,yd,dy,cd,perf[0].algo,NULL,0,&beta,wd,dw);
    if(rc){fprintf(stderr,"backward-filter failed: %s (%d)\n",cudnnGetErrorString(rc),rc);return 31;}
    for(i=0;i<4;++i){
        if(fabsf(dw[i]-dw_expected[i])>1e-6f){
            fprintf(stderr,"incorrect dw[%d]=%f expected=%f\n",i,dw[i],dw_expected[i]);return 32;
        }
    }

    cudnnDestroyConvolutionDescriptor(cd);
    cudnnDestroyFilterDescriptor(wd);
    cudnnDestroyTensorDescriptor(yd);
    cudnnDestroyTensorDescriptor(xd);
    cudnnDestroy(h);
    FreeLibrary(m);
    puts("PASS: cuDNN bridge forward/backward ABI + v7 heuristics + numerical correctness + fail-closed stream/mode");
    return 0;
}