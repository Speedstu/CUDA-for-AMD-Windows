/* Regression: cuDNN host float scalars -> MIOpen binary16 scalar ABI.
 * This would silently zero FP16 tensors before the conversion fix.
 * The MIOpen backend here is a CPU-only synthetic ABI stub. */
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef int (*PF_create)(void **);
typedef int (*PF_destroy)(void *);
typedef int (*PF_set_tensor4d_ex)(void *,int,int,int,int,int,int,int,int,int);
typedef int (*PF_transform)(void *,const void *,void *,const void *,
                            const void *,void *,void *);
typedef int (*PF_set_scale)(void *,void *,void *,const void *);
#define LOAD(name,type) type name=NULL;do { \
  FARPROC p=GetProcAddress(module,#name); \
  if(!p){fprintf(stderr,"missing %s\n",#name);return 2;} \
  memcpy(&name,&p,sizeof(name)); \
}while(0)
int main(int argc,char **argv) {
    HMODULE module;
    void *handle=NULL,*src=NULL,*dst=NULL;
    _Float16 x[4]={1,2,3,4};
    _Float16 y[8]={10,10,10,10,10,10,10,10};
    float alpha=2.f,beta=.5f,fill=3.f,scale=2.f;
    float nonhalf=1.0001f;
    int i,rc;
    const float transformed[4]={7.f,11.f,9.f,13.f};
    if(argc<2)return 2;
    module=LoadLibraryA(argv[1]);
    if(!module){fprintf(stderr,"Could not load bridge: %lu\n",GetLastError());return 2;}
    LOAD(cudnnCreate,PF_create);
    LOAD(cudnnDestroy,PF_destroy);
    LOAD(cudnnCreateTensorDescriptor,PF_create);
    LOAD(cudnnDestroyTensorDescriptor,PF_destroy);
    LOAD(cudnnSetTensor4dDescriptorEx,PF_set_tensor4d_ex);
    LOAD(cudnnTransformTensor,PF_transform);
    LOAD(cudnnSetTensor,PF_set_scale);
    LOAD(cudnnScaleTensor,PF_set_scale);
    if(cudnnCreate(&handle) || cudnnCreateTensorDescriptor(&src) ||
       cudnnCreateTensorDescriptor(&dst)) return 3;
    if(cudnnSetTensor4dDescriptorEx(src,2,1,2,1,2,4,2,2,1) ||
       cudnnSetTensor4dDescriptorEx(dst,2,1,2,1,2,8,1,4,2))return 4;
    rc=cudnnTransformTensor(handle,&alpha,src,x,&beta,dst,y);
    if(rc){fprintf(stderr,"half transform status %d\n",rc);return 5;}
    for(i=0;i<4;++i)
        if(fabsf((float)y[i]-transformed[i])>1e-5f){
            fprintf(stderr,"FP16 transform[%d] wrong: got %g expected %g\n",
                    i,(double)y[i],transformed[i]);return 6;
        }
    if(y[4]!=10) return 7;
    rc=cudnnSetTensor(handle,dst,y,&fill);
    if(rc)return 8;
    for(i=0;i<4;++i)if(y[i]!=3)return 9;
    if(y[4]!=10)return 10;
    rc=cudnnScaleTensor(handle,dst,y,&scale);
    if(rc)return 11;
    for(i=0;i<4;++i)if(y[i]!=6)return 12;
    if(y[4]!=10)return 13;

    rc=cudnnTransformTensor(handle,&alpha,src,(const void *)y,&beta,dst,y);
    if(rc!=9){fprintf(stderr,"in-place transform must refuse: %d\n",rc);return 20;}
    rc=cudnnSetTensor(handle,dst,y,&nonhalf);
    if(rc!=9){fprintf(stderr,"inexact fill must refuse: %d\n",rc);return 14;}
    rc=cudnnScaleTensor(handle,dst,y,&nonhalf);
    if(rc!=9){fprintf(stderr,"inexact scale must refuse: %d\n",rc);return 15;}
    rc=cudnnTransformTensor(handle,&nonhalf,src,x,&beta,dst,y);
    if(rc!=9){fprintf(stderr,"inexact transform must refuse: %d\n",rc);return 16;}
    {
        float nanv=NAN;
        rc=cudnnSetTensor(handle,dst,y,&nanv);
        if(rc!=9)return 17;
    }
    for(i=0;i<4;++i)if(y[i]!=6)return 18;
    if(y[4]!=10)return 19;
    cudnnDestroyTensorDescriptor(dst);
    cudnnDestroyTensorDescriptor(src);
    cudnnDestroy(handle);
    FreeLibrary(module);
    puts("PASS: cuDNN host float FP16 scalar conversion and fail-closed precision safeguards");
    return 0;
}
