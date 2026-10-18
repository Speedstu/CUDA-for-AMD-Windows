/* Offline, CPU-backed MIOpen cuDNN softmax ABI and numerical test. */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef int (*fn_create)(void **);
typedef int (*fn_destroy)(void *);
typedef int (*fn_tensor)(void *,int,int,int,int,int,int);
typedef int (*fn_forward)(void *,int,int,const void *,void *,const void *,
                          const void *,void *,void *);
typedef int (*fn_backward)(void *,int,int,const void *,void *,const void *,
                           void *,const void *,const void *,void *,void *);
#define LOAD(n,t) t n=NULL;do{FARPROC p=GetProcAddress(dll,#n);\
if(!p){fprintf(stderr,"missing %s\n",#n);return 2;}\
memcpy(&n,&p,sizeof(n));}while(0)
int main(int argc,char **argv){
    HMODULE dll;
    void *h=NULL,*xd=NULL,*yd=NULL;
    int n,c,w,rc,missing;
    float x[6]={0.f,1.f,2.f,0.f,-1.f,3.f};
    float y[6]={0},dy[6]={1.f,-0.3f,2.f,0.4f,-0.7f,1.f},dx[6]={0};
    float alpha=1.f,beta=0.f;
    if(argc<2) return 2;
    missing=(argc>2 && strcmp(argv[2],"--without-softmax")==0);
    dll=LoadLibraryA(argv[1]);
    if(!dll){fprintf(stderr,"DLL load failed %lu\n",GetLastError());return 2;}
    LOAD(cudnnCreate,fn_create);
    LOAD(cudnnDestroy,fn_destroy);
    LOAD(cudnnCreateTensorDescriptor,fn_create);
    LOAD(cudnnDestroyTensorDescriptor,fn_destroy);
    LOAD(cudnnSetTensor4dDescriptor,fn_tensor);
    LOAD(cudnnSoftmaxForward,fn_forward);
    LOAD(cudnnSoftmaxBackward,fn_backward);
    if(cudnnCreate(&h) || cudnnCreateTensorDescriptor(&xd) ||
       cudnnCreateTensorDescriptor(&yd))return 3;
    if(cudnnSetTensor4dDescriptor(xd,0,0,1,3,1,2) ||
       cudnnSetTensor4dDescriptor(yd,0,0,1,3,1,2))return 4;
    rc=cudnnSoftmaxForward(h,0,1,&alpha,xd,x,&beta,yd,y);
    if(rc!=9){fprintf(stderr,"FAST mode should be unsupported, got %d\n",rc);return 5;}
    rc=cudnnSoftmaxForward(h,1,0,&alpha,xd,x,&beta,yd,y);
    if(rc!=9){fprintf(stderr,"INSTANCE mode should be unsupported, got %d\n",rc);return 6;}
    rc=cudnnSoftmaxForward(h,1,1,&alpha,xd,x,&beta,yd,y);
    if(missing) {
        if(rc!=9) {fprintf(stderr,"missing optional softmax returned %d\n",rc);return 7;}
        goto cleanup;
    }
    if(rc){fprintf(stderr,"softmax forward returned %d\n",rc);return 8;}
    for(n=0;n<1;++n)for(w=0;w<2;++w) {
        float denom=0.f,maxval=-INFINITY,sum_grad=0.f;
        for(c=0;c<3;++c) {
            float v=x[c*2+w];
            if(v>maxval)maxval=v;
        }
        for(c=0;c<3;++c)denom+=expf(x[c*2+w]-maxval);
        for(c=0;c<3;++c) {
            float expected=expf(x[c*2+w]-maxval)/denom;
            if(fabsf(y[c*2+w]-expected)>1e-5f){
                fprintf(stderr,"fwd mismatch c=%d w=%d expected=%f got=%f\n",
                        c,w,expected,y[c*2+w]);return 9;
            }
            sum_grad+=expected*dy[c*2+w];
        }
        /* Run backward only after entire y matrix has been computed. */
        (void)sum_grad;
    }
    rc=cudnnSoftmaxBackward(h,1,1,&alpha,yd,y,yd,dy,&beta,xd,dx);
    if(rc){fprintf(stderr,"softmax backward returned %d\n",rc);return 10;}
    for(w=0;w<2;++w) {
        float dot=0.f;
        for(c=0;c<3;++c)dot+=dy[c*2+w]*y[c*2+w];
        for(c=0;c<3;++c) {
            float expected=y[c*2+w]*(dy[c*2+w]-dot);
            if(fabsf(dx[c*2+w]-expected)>1e-5f){
                fprintf(stderr,"bwd mismatch c=%d w=%d expected=%f got=%f\n",
                        c,w,expected,dx[c*2+w]);return 11;
            }
        }
    }
cleanup:
    cudnnDestroyTensorDescriptor(yd);
    cudnnDestroyTensorDescriptor(xd);
    cudnnDestroy(h);
    FreeLibrary(dll);
    puts("PASS: cuDNN channel-wise accurate softmax forward/backward and safe refusal");
    return 0;
}
