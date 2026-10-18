/* Offline cuDNN v8 activation ABI/numerics, backed by CPU MIOpen stub.
 * No GPU hardware is used by this test. */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef int (*fn_create)(void **);
typedef int (*fn_destroy)(void *);
typedef int (*fn_tensor)(void *,int,int,int,int,int,int);
typedef int (*fn_setact)(void *,int,int,double);
typedef int (*fn_getact)(void *,int *,int *,double *);
typedef int (*fn_forward)(void *,void *,const void *,void *,const void *,
                          const void *,void *,void *);
typedef int (*fn_backward)(void *,void *,const void *,void *,const void *,
                           void *,const void *,void *,const void *,
                           const void *,void *,void *);

#define RESOLVE(n, typ) typ n=NULL; do { \
    FARPROC proc=GetProcAddress(dll,#n); \
    if(!proc){fprintf(stderr,"missing export %s\n",#n);return 2;} \
    memcpy(&n,&proc,sizeof(n)); \
}while(0)

static float expected(float x,int mode) {
    if(mode==0) return 1.f/(1.f+expf(-x));
    if(mode==1) return x>0.f?x:0.f;
    return tanhf(x);
}

static float expected_grad(float x,float y,int mode) {
    if(mode==0) return y*(1.f-y);
    if(mode==1) return x>0.f?1.f:0.f;
    return 1.f-y*y;
}

int main(int argc,char **argv) {
    HMODULE dll;
    void *handle=NULL,*a=NULL,*xd=NULL,*yd=NULL;
    int mode,nan,rc,k,j;
    double coef;
    float alpha=1.f,beta=0.f,x[4]={-2.f,-1.f,0.f,2.f},y[4],dy[4]={1.f,1.f,1.f,1.f},dx[4];
    int without=(argc==3 && strcmp(argv[2],"--without-activation")==0);
    if(argc<2){fprintf(stderr,"usage: bridge_activation_smoke PATH [--without-activation]\n");return 2;}
    dll=LoadLibraryA(argv[1]);
    if(!dll){fprintf(stderr,"DLL not loadable: %lu\n",GetLastError());return 2;}
    RESOLVE(cudnnCreate,fn_create);
    RESOLVE(cudnnDestroy,fn_destroy);
    RESOLVE(cudnnCreateActivationDescriptor,fn_create);
    RESOLVE(cudnnDestroyActivationDescriptor,fn_destroy);
    RESOLVE(cudnnSetActivationDescriptor,fn_setact);
    RESOLVE(cudnnGetActivationDescriptor,fn_getact);
    RESOLVE(cudnnCreateTensorDescriptor,fn_create);
    RESOLVE(cudnnDestroyTensorDescriptor,fn_destroy);
    RESOLVE(cudnnSetTensor4dDescriptor,fn_tensor);
    RESOLVE(cudnnActivationForward,fn_forward);
    RESOLVE(cudnnActivationBackward,fn_backward);
    if(cudnnCreate(&handle)!=0 || cudnnCreateActivationDescriptor(&a)!=0 ||
       cudnnCreateTensorDescriptor(&xd)!=0 || cudnnCreateTensorDescriptor(&yd)!=0)
       return 3;
    if(cudnnSetTensor4dDescriptor(xd,0,0,1,1,1,4)!=0 ||
       cudnnSetTensor4dDescriptor(yd,0,0,1,1,1,4)!=0) return 4;
    if(cudnnSetActivationDescriptor(a,6,0,0.)!=9) return 5;
    if(cudnnSetActivationDescriptor(a,1,1,0.)!=9) return 6;
    if(cudnnSetActivationDescriptor(a,1,0,1.)!=9) return 7;
    for(k=0;k<3;++k) {
        mode=k;
        if(cudnnSetActivationDescriptor(a,mode,0,0.)!=0) return 8;
        if(cudnnGetActivationDescriptor(a,&nan,&j,&coef)!=0 ||
           nan!=mode || j!=0 || coef!=0.) return 9;
        for(j=0;j<4;++j){y[j]=-123.f;dx[j]=0.f;}
        rc=cudnnActivationForward(handle,a,&alpha,xd,x,&beta,yd,y);
        if(without) {
            if(rc!=9) {fprintf(stderr,"missing backend mode=%d returned %d\n",mode,rc);return 10;}
            continue;
        }
        if(rc){fprintf(stderr,"forward mode=%d status=%d\n",mode,rc);return 11;}
        for(j=0;j<4;++j) {
            float want=expected(x[j],mode);
            if(fabsf(y[j]-want)>1.e-5f) {
                fprintf(stderr,"forward mismatch mode=%d %d expected %.9g got %.9g\n",mode,j,want,y[j]);
                return 12;
            }
        }
        rc=cudnnActivationBackward(handle,a,&alpha,yd,y,yd,dy,xd,x,&beta,xd,dx);
        if(rc){fprintf(stderr,"backward mode=%d status=%d\n",mode,rc);return 13;}
        for(j=0;j<4;++j) {
            float want=expected_grad(x[j],y[j],mode);
            if(fabsf(dx[j]-want)>1.e-5f) {
                fprintf(stderr,"backward mismatch mode=%d %d expected %.9g got %.9g\n",mode,j,want,dx[j]);
                return 14;
            }
        }
    }
    cudnnDestroyTensorDescriptor(yd);
    cudnnDestroyTensorDescriptor(xd);
    cudnnDestroyActivationDescriptor(a);
    cudnnDestroy(handle);
    FreeLibrary(dll);
    puts("PASS: cuDNN activation sigmoid/ReLU/tanh ABI, numerics and unsupported modes");
    return 0;
}
