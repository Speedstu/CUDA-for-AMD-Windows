/* cuDNN v8 2D MAX pooling ABI and deterministic numerical CPU reference. */
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef int (*fn_create)(void **);
typedef int (*fn_destroy)(void *);
typedef int (*fn_tensor4d)(void *,int,int,int,int,int,int);
typedef int (*fn_poolset)(void *,int,int,int,int,int,int,int,int);
typedef int (*fn_poolout)(void *,void *,int *,int *,int *,int *);
typedef int (*fn_poolfwd)(void *,void *,const void *,void *,const void *,
                          const void *,void *,void *);
#define LOAD(n,typ) typ n=NULL;do { FARPROC proc=GetProcAddress(dll,#n); \
 if(!proc) {fprintf(stderr,"missing %s\n",#n);return 2;} memcpy(&n,&proc,sizeof(n)); }while(0)

int main(int argc,char **argv){
    HMODULE dll;
    void *handle=NULL,*pool=NULL,*xd=NULL,*yd=NULL;
    int n,c,h,w,rc,i;
    float x[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float y[4]={5,5,5,5};
    const float expected[4]={14.5f,18.5f,30.5f,34.5f};
    float alpha=2.f,beta=.5f;
    int missing=argc>2 && strcmp(argv[2],"--without-pooling")==0;
    if(argc<2)return 2;
    dll=LoadLibraryA(argv[1]);
    if(!dll){fprintf(stderr,"LoadLibrary failed %lu\n",GetLastError());return 2;}
    LOAD(cudnnCreate,fn_create);
    LOAD(cudnnDestroy,fn_destroy);
    LOAD(cudnnCreatePoolingDescriptor,fn_create);
    LOAD(cudnnDestroyPoolingDescriptor,fn_destroy);
    LOAD(cudnnSetPooling2dDescriptor,fn_poolset);
    LOAD(cudnnGetPooling2dForwardOutputDim,fn_poolout);
    LOAD(cudnnCreateTensorDescriptor,fn_create);
    LOAD(cudnnDestroyTensorDescriptor,fn_destroy);
    LOAD(cudnnSetTensor4dDescriptor,fn_tensor4d);
    LOAD(cudnnPoolingForward,fn_poolfwd);
    if(cudnnCreate(&handle)||cudnnCreatePoolingDescriptor(&pool)||
       cudnnCreateTensorDescriptor(&xd)||cudnnCreateTensorDescriptor(&yd))return 3;
    if(cudnnSetPooling2dDescriptor(pool,1,0,2,2,0,0,2,2)!=9)return 4; /* avg unsupported */
    if(cudnnSetPooling2dDescriptor(pool,0,0,2,2,0,0,1,1)!=9)return 5; /* overlap unsupported */
    if(cudnnSetPooling2dDescriptor(pool,0,0,2,2,1,0,2,2)!=9)return 6; /* padded unsupported */
    if(cudnnSetPooling2dDescriptor(pool,0,0,2,2,0,0,2,2))return 7;
    if(cudnnSetTensor4dDescriptor(xd,0,0,1,1,4,4))return 8;
    if(cudnnGetPooling2dForwardOutputDim(pool,xd,&n,&c,&h,&w))return 9;
    if(n!=1 || c!=1 || h!=2 || w!=2)return 10;
    if(cudnnSetTensor4dDescriptor(yd,0,0,n,c,h,w))return 11;
    rc=cudnnPoolingForward(handle,pool,&alpha,xd,x,&beta,yd,y);
    if(missing) {
        if(rc!=9){fprintf(stderr,"expected NOT_SUPPORTED for missing pool, got %d\n",rc);return 12;}
    } else {
        if(rc){fprintf(stderr,"pool forward returned %d\n",rc);return 13;}
        for(i=0;i<4;++i) {
            if(fabsf(y[i]-expected[i])>1e-5f){
                fprintf(stderr,"pool output[%d] expected %g got %g\n",i,expected[i],y[i]);
                return 14;
            }
        }
    }
    cudnnDestroyTensorDescriptor(yd);
    cudnnDestroyTensorDescriptor(xd);
    cudnnDestroyPoolingDescriptor(pool);
    cudnnDestroy(handle);
    FreeLibrary(dll);
    puts("PASS: cuDNN v8 2D max-pooling forward ABI values and fail-closed subset");
    return 0;
}
