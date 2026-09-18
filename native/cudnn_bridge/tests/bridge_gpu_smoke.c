/* Real-GPU smoke for cuDNN bridge -> MIOpen. No vendor headers required. SPDX-License-Identifier: MIT */
#include <windows.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

typedef int (*PF_hipMalloc)(void **, size_t);
typedef int (*PF_hipFree)(void *);
typedef int (*PF_hipMemcpy)(void *, const void *, size_t, int);
typedef int (*PF_hipDeviceSynchronize)(void);
typedef int (*PF_hipGetDeviceCount)(int *);

typedef int (*PF_create)(void **);
typedef int (*PF_destroy)(void *);
typedef int (*PF_create_desc)(void **);
typedef int (*PF_destroy_desc)(void *);
typedef int (*PF_set_tensor4d)(void *, int, int, int, int, int, int);
typedef int (*PF_set_filter4d)(void *, int, int, int, int, int, int);
typedef int (*PF_set_conv2d)(void *, int,int,int,int,int,int,int,int);
typedef int (*PF_outdim)(void *, void *, void *, int *, int *, int *, int *);
typedef int (*PF_workspace)(void *, void *, void *, void *, void *, int, size_t *);
typedef int (*PF_fwd)(void *, const void *, void *, const void *, void *, const void *,
                      void *, int, void *, size_t, const void *, void *, void *);
typedef const char *(*PF_error)(int);

#define LOAD_FROM(module, name, type) \
    type name = NULL; \
    do { \
        FARPROC p_ = GetProcAddress((module), #name); \
        if (!p_) { fprintf(stderr, "missing %s (winerr=%lu)\n", #name, GetLastError()); return 3; } \
        memcpy(&name, &p_, sizeof(name)); \
    } while (0)

static int hip_ok(int rc, const char *what) {
    if (rc != 0) {
        fprintf(stderr, "%s failed: HIP status %d\n", what, rc);
        return 0;
    }
    return 1;
}

static void cpu_xcorr_2d(const float *x, const float *w, float *y) {
    int p,q,r,s;
    for (p=0;p<2;++p) for(q=0;q<2;++q) {
        float sum=0.0f;
        for(r=0;r<2;++r) for(s=0;s<2;++s) {
            sum += x[(p+r)*3+(q+s)] * w[r*2+s];
        }
        y[p*2+q]=sum;
    }
}

int main(int argc, char **argv) {
    HMODULE hip=NULL, cudnn=NULL;
    void *h=NULL,*xd=NULL,*wd=NULL,*yd=NULL,*cd=NULL;
    void *dx=NULL,*dw=NULL,*dy=NULL,*dws=NULL;
    float hx[9]={1,2,3,4,5,6,7,8,9};
    float hw[4]={1,0,0,-1};
    float hy[4]={0,0,0,0}, expected[4]={0,0,0,0};
    float alpha=1.0f,beta=0.0f;
    size_t ws=0;
    int n,c,oh,ow,count=0,rc,i,algo,success_algo=-1;
    const int algos[] = {0,1,2,3,4,6};
    if(argc<3){fprintf(stderr,"usage: bridge_gpu_smoke <cudnn64_8.dll> <amdhip64_7.dll>\n");return 2;}

    hip=LoadLibraryA(argv[2]);
    if(!hip){fprintf(stderr,"LoadLibrary HIP failed: %lu\n",GetLastError());return 2;}
    LOAD_FROM(hip, hipMalloc, PF_hipMalloc);
    LOAD_FROM(hip, hipFree, PF_hipFree);
    LOAD_FROM(hip, hipMemcpy, PF_hipMemcpy);
    LOAD_FROM(hip, hipDeviceSynchronize, PF_hipDeviceSynchronize);
    LOAD_FROM(hip, hipGetDeviceCount, PF_hipGetDeviceCount);
    if(!hip_ok(hipGetDeviceCount(&count),"hipGetDeviceCount") || count<1){fprintf(stderr,"no HIP GPU\n");return 4;}

    cudnn=LoadLibraryA(argv[1]);
    if(!cudnn){fprintf(stderr,"LoadLibrary cuDNN bridge failed: %lu\n",GetLastError());return 5;}
    LOAD_FROM(cudnn, cudnnCreate, PF_create);
    LOAD_FROM(cudnn, cudnnDestroy, PF_destroy);
    LOAD_FROM(cudnn, cudnnCreateTensorDescriptor, PF_create_desc);
    LOAD_FROM(cudnn, cudnnDestroyTensorDescriptor, PF_destroy_desc);
    LOAD_FROM(cudnn, cudnnSetTensor4dDescriptor, PF_set_tensor4d);
    LOAD_FROM(cudnn, cudnnCreateFilterDescriptor, PF_create_desc);
    LOAD_FROM(cudnn, cudnnDestroyFilterDescriptor, PF_destroy_desc);
    LOAD_FROM(cudnn, cudnnSetFilter4dDescriptor, PF_set_filter4d);
    LOAD_FROM(cudnn, cudnnCreateConvolutionDescriptor, PF_create_desc);
    LOAD_FROM(cudnn, cudnnDestroyConvolutionDescriptor, PF_destroy_desc);
    LOAD_FROM(cudnn, cudnnSetConvolution2dDescriptor, PF_set_conv2d);
    LOAD_FROM(cudnn, cudnnGetConvolution2dForwardOutputDim, PF_outdim);
    LOAD_FROM(cudnn, cudnnGetConvolutionForwardWorkspaceSize, PF_workspace);
    LOAD_FROM(cudnn, cudnnConvolutionForward, PF_fwd);
    LOAD_FROM(cudnn, cudnnGetErrorString, PF_error);

    rc=cudnnCreate(&h);
    if(rc){fprintf(stderr,"cudnnCreate failed: %s (%d)\n",cudnnGetErrorString(rc),rc);return 6;}
    if(cudnnCreateTensorDescriptor(&xd)||cudnnCreateTensorDescriptor(&yd)||
       cudnnCreateFilterDescriptor(&wd)||cudnnCreateConvolutionDescriptor(&cd)) return 7;
    if(cudnnSetTensor4dDescriptor(xd,0,0,1,1,3,3)) return 8;
    if(cudnnSetFilter4dDescriptor(wd,0,0,1,1,2,2)) return 9;
    if(cudnnSetConvolution2dDescriptor(cd,0,0,1,1,1,1,1,0)) return 10;
    if(cudnnGetConvolution2dForwardOutputDim(cd,xd,wd,&n,&c,&oh,&ow)) return 11;
    if(n!=1||c!=1||oh!=2||ow!=2){fprintf(stderr,"bad output dims\n");return 12;}
    if(cudnnSetTensor4dDescriptor(yd,0,0,n,c,oh,ow)) return 13;

    if(!hip_ok(hipMalloc(&dx,sizeof(hx)),"hipMalloc x") ||
       !hip_ok(hipMalloc(&dw,sizeof(hw)),"hipMalloc w") ||
       !hip_ok(hipMalloc(&dy,sizeof(hy)),"hipMalloc y")) return 14;
    if(!hip_ok(hipMemcpy(dx,hx,sizeof(hx),1),"H2D x") ||
       !hip_ok(hipMemcpy(dw,hw,sizeof(hw),1),"H2D w") ||
       !hip_ok(hipMemcpy(dy,hy,sizeof(hy),1),"H2D y")) return 15;

    cpu_xcorr_2d(hx,hw,expected);
    for(i=0;i<(int)(sizeof(algos)/sizeof(algos[0]));++i) {
        algo=algos[i];
        ws=0;
        rc=cudnnGetConvolutionForwardWorkspaceSize(h,xd,wd,cd,yd,algo,&ws);
        if(rc!=0) {
            printf("algo %d workspace: %s (%d)\n",algo,cudnnGetErrorString(rc),rc);
            continue;
        }
        if(dws){hipFree(dws);dws=NULL;}
        if(ws>0 && !hip_ok(hipMalloc(&dws,ws),"hipMalloc workspace")) return 16;
        memset(hy,0,sizeof(hy));
        if(!hip_ok(hipMemcpy(dy,hy,sizeof(hy),1),"reset y")) return 17;
        rc=cudnnConvolutionForward(h,&alpha,xd,dx,wd,dw,cd,algo,dws,ws,&beta,yd,dy);
        printf("algo %d forward: %s (%d), workspace=%llu\n",algo,cudnnGetErrorString(rc),rc,(unsigned long long)ws);
        if(rc==0){success_algo=algo;break;}
    }
    if(success_algo<0){fprintf(stderr,"no mapped forward algorithm succeeded\n");return 18;}
    if(!hip_ok(hipDeviceSynchronize(),"hipDeviceSynchronize")) return 19;
    if(!hip_ok(hipMemcpy(hy,dy,sizeof(hy),2),"D2H y")) return 20;
    for(i=0;i<4;++i){
        float err=fabsf(hy[i]-expected[i]);
        printf("y[%d]=%.9g expected=%.9g abs_err=%.9g\n",i,hy[i],expected[i],err);
        if(!isfinite(hy[i])||err>1e-5f){fprintf(stderr,"numerical mismatch\n");return 21;}
    }

    if(dws) hipFree(dws);
    hipFree(dy); hipFree(dw); hipFree(dx);
    cudnnDestroyConvolutionDescriptor(cd);
    cudnnDestroyFilterDescriptor(wd);
    cudnnDestroyTensorDescriptor(yd);
    cudnnDestroyTensorDescriptor(xd);
    cudnnDestroy(h);
    FreeLibrary(cudnn);
    FreeLibrary(hip);
    printf("PASS: real GPU cuDNN bridge -> MIOpen convolution, algorithm=%d\n",success_algo);
    return 0;
}
