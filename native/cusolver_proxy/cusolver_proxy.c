#include <windows.h>
#include <stdint.h>

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

static FARPROC hip_symbol(const char* name) {
    if (!g_hipsolver) {
        g_hipsolver = LoadLibraryW(L"hipsolver.dll");
        if (!g_hipsolver) return NULL;
    }
    return GetProcAddress(g_hipsolver, name);
}

#define LOAD_FN(name, type) type fn = (type)hip_symbol(name); if (!fn) return 1
#define EXPORT __declspec(dllexport)

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

#ifdef __cplusplus
}
#endif
