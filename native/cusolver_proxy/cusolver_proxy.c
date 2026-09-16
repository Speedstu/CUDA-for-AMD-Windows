#include <windows.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* hipsolverHandle_t;
typedef int hipsolverStatus_t;
typedef void* hipStream_t;

static HMODULE g_hipsolver = NULL;

static int process_shutdown_in_progress(void) {
    typedef unsigned char (WINAPI *rtl_shutdown_fn)(void);
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll) return 0;
    rtl_shutdown_fn fn = (rtl_shutdown_fn)GetProcAddress(ntdll, "RtlDllShutdownInProgress");
    return fn ? (fn() != 0) : 0;
}

static int cuda_status_from_hip(int s) {
    switch (s) {
        case 0: return 0;  /* SUCCESS */
        case 1: return 1;  /* NOT_INITIALIZED */
        case 2: return 2;  /* ALLOC_FAILED */
        case 3: return 3;  /* INVALID_VALUE */
        case 8: return 4;  /* ARCH_MISMATCH */
        case 4: return 5;  /* MAPPING_ERROR */
        case 5: return 6;  /* EXECUTION_FAILED */
        case 6: return 7;  /* INTERNAL_ERROR */
        case 13: return 8; /* MATRIX_TYPE_NOT_SUPPORTED */
        case 7: return 9;  /* NOT_SUPPORTED */
        case 12: return 10;/* ZERO_PIVOT */
        default: return 7;
    }
}

static int hip_op_from_cuda(int op) {
    switch (op) {
        case 0: return 111; /* CUBLAS_OP_N -> HIPBLAS_OP_N */
        case 1: return 112; /* CUBLAS_OP_T -> HIPBLAS_OP_T */
        case 2: return 113; /* CUBLAS_OP_C -> HIPBLAS_OP_C */
        default: return -1;
    }
}

static int hip_fill_from_cuda(int uplo) {
    switch (uplo) {
        case 0: return 122; /* CUBLAS_FILL_MODE_LOWER -> HIPBLAS_FILL_MODE_LOWER */
        case 1: return 121; /* CUBLAS_FILL_MODE_UPPER -> HIPBLAS_FILL_MODE_UPPER */
        case 2: return 123; /* CUBLAS_FILL_MODE_FULL  -> HIPBLAS_FILL_MODE_FULL */
        default: return -1;
    }
}

static int hip_side_from_cuda(int side) {
    switch (side) {
        case 0: return 141; /* CUBLAS_SIDE_LEFT  -> HIPBLAS_SIDE_LEFT */
        case 1: return 142; /* CUBLAS_SIDE_RIGHT -> HIPBLAS_SIDE_RIGHT */
        default: return -1;
    }
}

static int hip_data_type_from_cuda(int data_type) {
    switch (data_type) {
        case 0: return 0; /* CUDA_R_32F -> HIP_R_32F */
        case 1: return 1; /* CUDA_R_64F -> HIP_R_64F */
        case 4: return 4; /* CUDA_C_32F -> HIP_C_32F */
        case 5: return 5; /* CUDA_C_64F -> HIP_C_64F */
        default: return -1;
    }
}

static int hip_alg_from_cuda(int alg) {
    switch (alg) {
        case 0: return 231; /* CUSOLVER_ALG_0 -> HIPSOLVER_ALG_0 */
        case 1: return 232; /* CUSOLVER_ALG_1 -> HIPSOLVER_ALG_1 */
        default: return -1;
    }
}

static FARPROC hip_symbol(const char* name) {
    if (!g_hipsolver) {
        g_hipsolver = LoadLibraryW(L"hipsolver.dll");
        if (!g_hipsolver) return NULL;
    }
    return GetProcAddress(g_hipsolver, name);
}

#define LOAD_FN(name, type) type fn = (type)hip_symbol(name); if (!fn) return 1
#define EXPORT __declspec(dllexport)

static int resolve_params(void* params, void** effective, void** owned) {
    *effective = params;
    *owned = NULL;
    if (params) return 0;
    typedef int (*create_fn_t)(void**);
    create_fn_t create_fn = (create_fn_t)hip_symbol("hipsolverDnCreateParams");
    if (!create_fn) return 1;
    int status = create_fn(owned);
    if (status != 0) return cuda_status_from_hip(status);
    *effective = *owned;
    return 0;
}

static void release_owned_params(void* owned) {
    if (!owned || process_shutdown_in_progress()) return;
    typedef int (*destroy_fn_t)(void*);
    destroy_fn_t destroy_fn = (destroy_fn_t)hip_symbol("hipsolverDnDestroyParams");
    if (destroy_fn) (void)destroy_fn(owned);
}

EXPORT int cusolverDnCreate(void** handle) {
    typedef int (*fn_t)(hipsolverHandle_t*);
    LOAD_FN("hipsolverDnCreate", fn_t);
    return cuda_status_from_hip(fn((hipsolverHandle_t*)handle));
}

EXPORT int cusolverDnDestroy(void* handle) {
    if (process_shutdown_in_progress()) return 0;
    typedef int (*fn_t)(hipsolverHandle_t);
    LOAD_FN("hipsolverDnDestroy", fn_t);
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle));
}

EXPORT int cusolverDnSetStream(void* handle, void* stream) {
    typedef int (*fn_t)(hipsolverHandle_t, hipStream_t);
    LOAD_FN("hipsolverDnSetStream", fn_t);
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, (hipStream_t)stream));
}

EXPORT int cusolverDnCreateParams(void** params) {
    typedef int (*fn_t)(void**);
    LOAD_FN("hipsolverDnCreateParams", fn_t);
    return cuda_status_from_hip(fn(params));
}

EXPORT int cusolverDnDestroyParams(void* params) {
    if (process_shutdown_in_progress()) return 0;
    typedef int (*fn_t)(void*);
    LOAD_FN("hipsolverDnDestroyParams", fn_t);
    return cuda_status_from_hip(fn(params));
}

EXPORT int cusolverDnSetAdvOptions(void* params, int func, int alg) {
    if (func != 0) return 9; /* only CUSOLVERDN_GETRF is defined for this API family */
    int halg = hip_alg_from_cuda(alg);
    if (halg < 0) return 9;
    typedef int (*fn_t)(void*, int, int);
    LOAD_FN("hipsolverDnSetAdvOptions", fn_t);
    return cuda_status_from_hip(fn(params, 0, halg));
}

EXPORT int cusolverDnXpotrf_bufferSize(void* handle, void* params, int uplo, int64_t n, int dataTypeA, const void* A, int64_t lda, int computeType, size_t* lworkOnDevice, size_t* lworkOnHost) {
    int huplo = hip_fill_from_cuda(uplo);
    int htypeA = hip_data_type_from_cuda(dataTypeA);
    int hcompute = hip_data_type_from_cuda(computeType);
    if (huplo < 0) return 3;
    if (htypeA < 0 || hcompute < 0) return 9;
    void *effective = NULL, *owned = NULL;
    int ps = resolve_params(params, &effective, &owned);
    if (ps != 0) return ps;
    typedef int (*fn_t)(hipsolverHandle_t, void*, int, int64_t, int, const void*, int64_t, int, size_t*, size_t*);
    LOAD_FN("hipsolverDnXpotrf_bufferSize", fn_t);
    int hs = fn((hipsolverHandle_t)handle, effective, huplo, n, htypeA, A, lda, hcompute, lworkOnDevice, lworkOnHost);
    release_owned_params(owned);
    return cuda_status_from_hip(hs);
}

EXPORT int cusolverDnXpotrf(void* handle, void* params, int uplo, int64_t n, int dataTypeA, void* A, int64_t lda, int computeType, void* workOnDevice, size_t lworkOnDevice, void* workOnHost, size_t lworkOnHost, int* info) {
    int huplo = hip_fill_from_cuda(uplo);
    int htypeA = hip_data_type_from_cuda(dataTypeA);
    int hcompute = hip_data_type_from_cuda(computeType);
    if (huplo < 0) return 3;
    if (htypeA < 0 || hcompute < 0) return 9;
    void *effective = NULL, *owned = NULL;
    int ps = resolve_params(params, &effective, &owned);
    if (ps != 0) return ps;
    typedef int (*fn_t)(hipsolverHandle_t, void*, int, int64_t, int, void*, int64_t, int, void*, size_t, void*, size_t, int*);
    LOAD_FN("hipsolverDnXpotrf", fn_t);
    int hs = fn((hipsolverHandle_t)handle, effective, huplo, n, htypeA, A, lda, hcompute, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, info);
    release_owned_params(owned);
    return cuda_status_from_hip(hs);
}

EXPORT int cusolverDnXpotrs(void* handle, void* params, int uplo, int64_t n, int64_t nrhs, int dataTypeA, const void* A, int64_t lda, int dataTypeB, void* B, int64_t ldb, int* info) {
    int huplo = hip_fill_from_cuda(uplo);
    int htypeA = hip_data_type_from_cuda(dataTypeA);
    int htypeB = hip_data_type_from_cuda(dataTypeB);
    if (huplo < 0) return 3;
    if (htypeA < 0 || htypeB < 0) return 9;
    void *effective = NULL, *owned = NULL;
    int ps = resolve_params(params, &effective, &owned);
    if (ps != 0) return ps;
    typedef int (*fn_t)(hipsolverHandle_t, void*, int, int64_t, int64_t, int, const void*, int64_t, int, void*, int64_t, int*);
    LOAD_FN("hipsolverDnXpotrs", fn_t);
    int hs = fn((hipsolverHandle_t)handle, effective, huplo, n, nrhs, htypeA, A, lda, htypeB, B, ldb, info);
    release_owned_params(owned);
    return cuda_status_from_hip(hs);
}

EXPORT int cusolverDnXgeqrf_bufferSize(void* handle, void* params, int64_t m, int64_t n, int dataTypeA, const void* A, int64_t lda, int dataTypeTau, const void* tau, int computeType, size_t* lworkOnDevice, size_t* lworkOnHost) {
    int htypeA = hip_data_type_from_cuda(dataTypeA);
    int htypeTau = hip_data_type_from_cuda(dataTypeTau);
    int hcompute = hip_data_type_from_cuda(computeType);
    if (htypeA < 0 || htypeTau < 0 || hcompute < 0) return 9;
    void *effective = NULL, *owned = NULL;
    int ps = resolve_params(params, &effective, &owned);
    if (ps != 0) return ps;
    typedef int (*fn_t)(hipsolverHandle_t, void*, int64_t, int64_t, int, const void*, int64_t, int, const void*, int, size_t*, size_t*);
    LOAD_FN("hipsolverDnXgeqrf_bufferSize", fn_t);
    int hs = fn((hipsolverHandle_t)handle, effective, m, n, htypeA, A, lda, htypeTau, tau, hcompute, lworkOnDevice, lworkOnHost);
    release_owned_params(owned);
    return cuda_status_from_hip(hs);
}

EXPORT int cusolverDnXgeqrf(void* handle, void* params, int64_t m, int64_t n, int dataTypeA, void* A, int64_t lda, int dataTypeTau, void* tau, int computeType, void* workOnDevice, size_t lworkOnDevice, void* workOnHost, size_t lworkOnHost, int* info) {
    int htypeA = hip_data_type_from_cuda(dataTypeA);
    int htypeTau = hip_data_type_from_cuda(dataTypeTau);
    int hcompute = hip_data_type_from_cuda(computeType);
    if (htypeA < 0 || htypeTau < 0 || hcompute < 0) return 9;
    void *effective = NULL, *owned = NULL;
    int ps = resolve_params(params, &effective, &owned);
    if (ps != 0) return ps;
    typedef int (*fn_t)(hipsolverHandle_t, void*, int64_t, int64_t, int, void*, int64_t, int, void*, int, void*, size_t, void*, size_t, int*);
    LOAD_FN("hipsolverDnXgeqrf", fn_t);
    int hs = fn((hipsolverHandle_t)handle, effective, m, n, htypeA, A, lda, htypeTau, tau, hcompute, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, info);
    release_owned_params(owned);
    return cuda_status_from_hip(hs);
}

#define DEFINE_GETRF_BUFFER(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##getrf_bufferSize(void* handle, int m, int n, T* A, int lda, int* lwork) { \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "getrf_bufferSize", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, m, n, A, lda, lwork)); \
}

#define DEFINE_GETRF(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##getrf(void* handle, int m, int n, T* A, int lda, T* work, int* devIpiv, int* devInfo) { \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, T*, int*, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "getrf", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, m, n, A, lda, work, devIpiv, devInfo)); \
}

#define DEFINE_GETRS(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##getrs(void* handle, int trans, int n, int nrhs, const T* A, int lda, const int* devIpiv, T* B, int ldb, int* devInfo) { \
    int hop = hip_op_from_cuda(trans); \
    if (hop < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, int, const T*, int, const int*, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "getrs", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, hop, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)); \
}


#define DEFINE_GEQRF_BUFFER(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##geqrf_bufferSize(void* handle, int m, int n, T* A, int lda, int* lwork) { \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "geqrf_bufferSize", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, m, n, A, lda, lwork)); \
}

#define DEFINE_GEQRF(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##geqrf(void* handle, int m, int n, T* A, int lda, T* tau, T* work, int lwork, int* devInfo) { \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, T*, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "geqrf", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, m, n, A, lda, tau, work, lwork, devInfo)); \
}

#define DEFINE_QGEN_BUFFER(T, PREFIX, HIPNAME) \
EXPORT int cusolverDn##PREFIX##HIPNAME##_bufferSize(void* handle, int m, int n, int k, const T* A, int lda, const T* tau, int* lwork) { \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, int, const T*, int, const T*, int*); \
    LOAD_FN("hipsolverDn" #PREFIX #HIPNAME "_bufferSize", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, m, n, k, A, lda, tau, lwork)); \
}

#define DEFINE_QGEN(T, PREFIX, HIPNAME) \
EXPORT int cusolverDn##PREFIX##HIPNAME(void* handle, int m, int n, int k, T* A, int lda, const T* tau, T* work, int lwork, int* devInfo) { \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, int, T*, int, const T*, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX #HIPNAME, fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, m, n, k, A, lda, tau, work, lwork, devInfo)); \
}

#define DEFINE_QAPPLY_BUFFER(T, PREFIX, HIPNAME) \
EXPORT int cusolverDn##PREFIX##HIPNAME##_bufferSize(void* handle, int side, int trans, int m, int n, int k, const T* A, int lda, const T* tau, const T* C, int ldc, int* lwork) { \
    int hside = hip_side_from_cuda(side); \
    int hop = hip_op_from_cuda(trans); \
    if (hside < 0 || hop < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, int, int, int, const T*, int, const T*, const T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX #HIPNAME "_bufferSize", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, hside, hop, m, n, k, A, lda, tau, C, ldc, lwork)); \
}

#define DEFINE_QAPPLY(T, PREFIX, HIPNAME) \
EXPORT int cusolverDn##PREFIX##HIPNAME(void* handle, int side, int trans, int m, int n, int k, const T* A, int lda, const T* tau, T* C, int ldc, T* work, int lwork, int* devInfo) { \
    int hside = hip_side_from_cuda(side); \
    int hop = hip_op_from_cuda(trans); \
    if (hside < 0 || hop < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, int, int, int, const T*, int, const T*, T*, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX #HIPNAME, fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, hside, hop, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)); \
}

#define DEFINE_POTRF_BUFFER(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##potrf_bufferSize(void* handle, int uplo, int n, T* A, int lda, int* lwork) { \
    int huplo = hip_fill_from_cuda(uplo); \
    if (huplo < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "potrf_bufferSize", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, huplo, n, A, lda, lwork)); \
}

#define DEFINE_POTRF(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##potrf(void* handle, int uplo, int n, T* A, int lda, T* work, int lwork, int* devInfo) { \
    int huplo = hip_fill_from_cuda(uplo); \
    if (huplo < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "potrf", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, huplo, n, A, lda, work, lwork, devInfo)); \
}

#define DEFINE_POTRF_BATCHED(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##potrfBatched(void* handle, int uplo, int n, T** Aarray, int lda, int* devInfo, int batchCount) { \
    int huplo = hip_fill_from_cuda(uplo); \
    if (huplo < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T**, int, int*, int); \
    LOAD_FN("hipsolverDn" #PREFIX "potrfBatched", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, huplo, n, Aarray, lda, devInfo, batchCount)); \
}

#define DEFINE_POTRI_BUFFER(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##potri_bufferSize(void* handle, int uplo, int n, T* A, int lda, int* lwork) { \
    int huplo = hip_fill_from_cuda(uplo); \
    if (huplo < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "potri_bufferSize", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, huplo, n, A, lda, lwork)); \
}

#define DEFINE_POTRI(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##potri(void* handle, int uplo, int n, T* A, int lda, T* work, int lwork, int* devInfo) { \
    int huplo = hip_fill_from_cuda(uplo); \
    if (huplo < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, T*, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "potri", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, huplo, n, A, lda, work, lwork, devInfo)); \
}

#define DEFINE_POTRS(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##potrs(void* handle, int uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int* devInfo) { \
    int huplo = hip_fill_from_cuda(uplo); \
    if (huplo < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, int, const T*, int, T*, int, int*); \
    LOAD_FN("hipsolverDn" #PREFIX "potrs", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, huplo, n, nrhs, A, lda, B, ldb, devInfo)); \
}

#define DEFINE_POTRS_BATCHED(T, PREFIX) \
EXPORT int cusolverDn##PREFIX##potrsBatched(void* handle, int uplo, int n, int nrhs, T** Aarray, int lda, T** Barray, int ldb, int* devInfo, int batchCount) { \
    int huplo = hip_fill_from_cuda(uplo); \
    if (huplo < 0) return 3; \
    typedef int (*fn_t)(hipsolverHandle_t, int, int, int, T**, int, T**, int, int*, int); \
    LOAD_FN("hipsolverDn" #PREFIX "potrsBatched", fn_t); \
    return cuda_status_from_hip(fn((hipsolverHandle_t)handle, huplo, n, nrhs, Aarray, lda, Barray, ldb, devInfo, batchCount)); \
}

typedef struct { float x, y; } cfloat2;
typedef struct { double x, y; } cdouble2;

DEFINE_GETRF_BUFFER(float, S)
DEFINE_GETRF_BUFFER(double, D)
DEFINE_GETRF_BUFFER(cfloat2, C)
DEFINE_GETRF_BUFFER(cdouble2, Z)

DEFINE_GETRF(float, S)
DEFINE_GETRF(double, D)
DEFINE_GETRF(cfloat2, C)
DEFINE_GETRF(cdouble2, Z)

DEFINE_GETRS(float, S)
DEFINE_GETRS(double, D)
DEFINE_GETRS(cfloat2, C)
DEFINE_GETRS(cdouble2, Z)

DEFINE_GEQRF_BUFFER(float, S)
DEFINE_GEQRF_BUFFER(double, D)
DEFINE_GEQRF_BUFFER(cfloat2, C)
DEFINE_GEQRF_BUFFER(cdouble2, Z)
DEFINE_GEQRF(float, S)
DEFINE_GEQRF(double, D)
DEFINE_GEQRF(cfloat2, C)
DEFINE_GEQRF(cdouble2, Z)

DEFINE_QGEN_BUFFER(float, S, orgqr)
DEFINE_QGEN_BUFFER(double, D, orgqr)
DEFINE_QGEN_BUFFER(cfloat2, C, ungqr)
DEFINE_QGEN_BUFFER(cdouble2, Z, ungqr)
DEFINE_QGEN(float, S, orgqr)
DEFINE_QGEN(double, D, orgqr)
DEFINE_QGEN(cfloat2, C, ungqr)
DEFINE_QGEN(cdouble2, Z, ungqr)

DEFINE_QAPPLY_BUFFER(float, S, ormqr)
DEFINE_QAPPLY_BUFFER(double, D, ormqr)
DEFINE_QAPPLY_BUFFER(cfloat2, C, unmqr)
DEFINE_QAPPLY_BUFFER(cdouble2, Z, unmqr)
DEFINE_QAPPLY(float, S, ormqr)
DEFINE_QAPPLY(double, D, ormqr)
DEFINE_QAPPLY(cfloat2, C, unmqr)
DEFINE_QAPPLY(cdouble2, Z, unmqr)

#define DEFINE_CHOLESKY_FAMILY(T, PREFIX) \
    DEFINE_POTRF_BUFFER(T, PREFIX) \
    DEFINE_POTRF(T, PREFIX) \
    DEFINE_POTRF_BATCHED(T, PREFIX) \
    DEFINE_POTRI_BUFFER(T, PREFIX) \
    DEFINE_POTRI(T, PREFIX) \
    DEFINE_POTRS(T, PREFIX) \
    DEFINE_POTRS_BATCHED(T, PREFIX)

DEFINE_CHOLESKY_FAMILY(float, S)
DEFINE_CHOLESKY_FAMILY(double, D)
DEFINE_CHOLESKY_FAMILY(cfloat2, C)
DEFINE_CHOLESKY_FAMILY(cdouble2, Z)

#ifdef __cplusplus
}
#endif
