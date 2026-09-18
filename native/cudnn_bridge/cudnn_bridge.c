/*
 * Experimental cuDNN v8 -> MIOpen compatibility bridge for Windows.
 *
 * This file intentionally carries a narrow, fail-closed ABI surface. It does
 * not depend on NVIDIA or AMD headers at build time and loads MIOpen.dll at
 * runtime. Unported cuDNN exports are handled by the generated proxy .def and
 * forwarded to the user's original cuDNN DLL.
 *
 * SPDX-License-Identifier: MIT
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ---- Minimal cuDNN v8 ABI types used by the implemented subset ---------- */

typedef int cudnnStatus_t;
typedef int cudnnDataType_t;
typedef int cudnnTensorFormat_t;
typedef int cudnnConvolutionMode_t;
typedef int cudnnConvolutionFwdAlgo_t;
typedef int cudnnConvolutionBwdDataAlgo_t;
typedef int cudnnConvolutionBwdFilterAlgo_t;
typedef int cudnnMathType_t;

enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    CUDNN_STATUS_VERSION_MISMATCH = 14
};

enum {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64 = 10
};

enum {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2
};

enum {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1
};

enum {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8
};

enum {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
};

enum {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7
};

#define BRIDGE_MAGIC_HANDLE 0x48444e4eU
#define BRIDGE_MAGIC_TENSOR 0x54444e4eU
#define BRIDGE_MAGIC_FILTER 0x46444e4eU
#define BRIDGE_MAGIC_CONV   0x43444e4eU
#define BRIDGE_MAX_DIMS 8

typedef struct BridgeHandle {
    uint32_t magic;
    void *miopen;
    void *cuda_stream;
} BridgeHandle;

typedef struct BridgeTensor {
    uint32_t magic;
    int data_type;
    int format;
    int nb_dims;
    int dims[BRIDGE_MAX_DIMS];
    int strides[BRIDGE_MAX_DIMS];
} BridgeTensor;

typedef struct BridgeFilter {
    uint32_t magic;
    int data_type;
    int format;
    int nb_dims;
    int dims[BRIDGE_MAX_DIMS];
} BridgeFilter;

typedef struct BridgeConv {
    uint32_t magic;
    int spatial_dims;
    int pad[3];
    int stride[3];
    int dilation[3];
    int mode;
    int compute_type;
    int group_count;
    int math_type;
} BridgeConv;

typedef BridgeHandle *cudnnHandle_t;
typedef BridgeTensor *cudnnTensorDescriptor_t;
typedef BridgeFilter *cudnnFilterDescriptor_t;
typedef BridgeConv *cudnnConvolutionDescriptor_t;

/* Forward declarations for public entry points used by convenience wrappers. */
cudnnStatus_t cudnnSetFilterNdDescriptor(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int nbDims, const int filterDimA[]);
cudnnStatus_t cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[],
    const int filterStrideA[], const int dilationA[], cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType);
cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims, int tensorOuputDimA[]);

/* ---- Minimal MIOpen ABI loaded dynamically ------------------------------ */

typedef int miopenStatus_t;
typedef void *miopenHandle_t;
typedef void *miopenTensorDescriptor_t;
typedef void *miopenConvolutionDescriptor_t;
typedef void *miopenAcceleratorQueue_t;
typedef int miopenDataType_t;
typedef int miopenConvolutionMode_t;
typedef int miopenConvFwdAlgorithm_t;
typedef int miopenConvBwdDataAlgorithm_t;
typedef int miopenConvBwdWeightsAlgorithm_t;

enum {
    MIOPEN_STATUS_SUCCESS = 0,
    MIOPEN_STATUS_NOT_INITIALIZED = 1,
    MIOPEN_STATUS_INVALID_VALUE = 2,
    MIOPEN_STATUS_BAD_PARM = 3,
    MIOPEN_STATUS_ALLOC_FAILED = 4,
    MIOPEN_STATUS_INTERNAL_ERROR = 5,
    MIOPEN_STATUS_NOT_IMPLEMENTED = 6,
    MIOPEN_STATUS_UNKNOWN_ERROR = 7,
    MIOPEN_STATUS_UNSUPPORTED_OP = 8,
    MIOPEN_STATUS_GPU_OPERATIONS_SKIPPED = 9,
    MIOPEN_STATUS_VERSION_MISMATCH = 10
};

enum {
    MIOPEN_HALF = 0,
    MIOPEN_FLOAT = 1,
    MIOPEN_INT32 = 2,
    MIOPEN_INT8 = 3,
    MIOPEN_BFLOAT16 = 5,
    MIOPEN_DOUBLE = 6,
    MIOPEN_INT64 = 9
};

enum {
    MIOPEN_CONVOLUTION = 0
};

enum {
    MIOPEN_FWD_GEMM = 0,
    MIOPEN_FWD_DIRECT = 1,
    MIOPEN_FWD_FFT = 2,
    MIOPEN_FWD_WINOGRAD = 3,
    MIOPEN_FWD_IMPLICIT_GEMM = 5
};

enum {
    MIOPEN_BWD_DATA_GEMM = 0,
    MIOPEN_BWD_DATA_DIRECT = 1,
    MIOPEN_BWD_DATA_FFT = 2,
    MIOPEN_BWD_DATA_WINOGRAD = 3,
    MIOPEN_BWD_DATA_IMPLICIT_GEMM = 5
};

enum {
    MIOPEN_BWD_WEIGHTS_GEMM = 0,
    MIOPEN_BWD_WEIGHTS_DIRECT = 1,
    MIOPEN_BWD_WEIGHTS_WINOGRAD = 3,
    MIOPEN_BWD_WEIGHTS_IMPLICIT_GEMM = 5
};

typedef struct MiopenConvAlgoPerf {
    miopenConvFwdAlgorithm_t fwd_algo;
    float time;
    size_t memory;
} MiopenConvAlgoPerf;

typedef struct MiopenConvSolution {
    float time;
    size_t workspace_size;
    uint64_t solution_id;
    int algorithm;
} MiopenConvSolution;

typedef struct CudnnConvolutionFwdAlgoPerf {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    int determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} CudnnConvolutionFwdAlgoPerf;

typedef struct CudnnConvolutionBwdDataAlgoPerf {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    int determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} CudnnConvolutionBwdDataAlgoPerf;

typedef struct CudnnConvolutionBwdFilterAlgoPerf {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    int determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} CudnnConvolutionBwdFilterAlgoPerf;

typedef miopenStatus_t (*PFN_miopenCreate)(miopenHandle_t *);
typedef miopenStatus_t (*PFN_miopenDestroy)(miopenHandle_t);
typedef miopenStatus_t (*PFN_miopenSetStream)(miopenHandle_t, miopenAcceleratorQueue_t);
typedef miopenStatus_t (*PFN_miopenCreateTensorDescriptor)(miopenTensorDescriptor_t *);
typedef miopenStatus_t (*PFN_miopenDestroyTensorDescriptor)(miopenTensorDescriptor_t);
typedef miopenStatus_t (*PFN_miopenSetTensorDescriptor)(
    miopenTensorDescriptor_t, miopenDataType_t, int, const int *, const int *);
typedef miopenStatus_t (*PFN_miopenCreateConvolutionDescriptor)(miopenConvolutionDescriptor_t *);
typedef miopenStatus_t (*PFN_miopenDestroyConvolutionDescriptor)(miopenConvolutionDescriptor_t);
typedef miopenStatus_t (*PFN_miopenInitConvolutionNdDescriptor)(
    miopenConvolutionDescriptor_t, int, const int *, const int *, const int *, miopenConvolutionMode_t);
typedef miopenStatus_t (*PFN_miopenSetConvolutionGroupCount)(miopenConvolutionDescriptor_t, int);
typedef miopenStatus_t (*PFN_miopenConvolutionForwardGetWorkSpaceSize)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, size_t *);
typedef miopenStatus_t (*PFN_miopenFindConvolutionForwardAlgorithm)(
    miopenHandle_t, const miopenTensorDescriptor_t, const void *, const miopenTensorDescriptor_t,
    const void *, const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, void *,
    int, int *, MiopenConvAlgoPerf *, void *, size_t, bool);
typedef miopenStatus_t (*PFN_miopenConvolutionForwardGetSolutionCount)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, size_t *);
typedef miopenStatus_t (*PFN_miopenConvolutionForwardGetSolution)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, size_t, size_t *,
    MiopenConvSolution *);
typedef miopenStatus_t (*PFN_miopenConvolutionForwardGetSolutionWorkspaceSize)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, uint64_t, size_t *);
typedef miopenStatus_t (*PFN_miopenConvolutionForwardImmediate)(
    miopenHandle_t, const miopenTensorDescriptor_t, const void *,
    const miopenTensorDescriptor_t, const void *, const miopenConvolutionDescriptor_t,
    const miopenTensorDescriptor_t, void *, void *, size_t, uint64_t);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardDataGetSolutionCount)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, size_t *);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardDataGetSolution)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, size_t, size_t *,
    MiopenConvSolution *);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardDataGetSolutionWorkspaceSize)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, uint64_t, size_t *);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardDataImmediate)(
    miopenHandle_t, const miopenTensorDescriptor_t, const void *,
    const miopenTensorDescriptor_t, const void *, const miopenConvolutionDescriptor_t,
    const miopenTensorDescriptor_t, void *, void *, size_t, uint64_t);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardWeightsGetSolutionCount)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, size_t *);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardWeightsGetSolution)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, size_t, size_t *,
    MiopenConvSolution *);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize)(
    miopenHandle_t, const miopenTensorDescriptor_t, const miopenTensorDescriptor_t,
    const miopenConvolutionDescriptor_t, const miopenTensorDescriptor_t, uint64_t, size_t *);
typedef miopenStatus_t (*PFN_miopenConvolutionBackwardWeightsImmediate)(
    miopenHandle_t, const miopenTensorDescriptor_t, const void *,
    const miopenTensorDescriptor_t, const void *, const miopenConvolutionDescriptor_t,
    const miopenTensorDescriptor_t, void *, void *, size_t, uint64_t);
typedef miopenStatus_t (*PFN_miopenConvolutionForward)(
    miopenHandle_t, const void *, const miopenTensorDescriptor_t, const void *,
    const miopenTensorDescriptor_t, const void *, const miopenConvolutionDescriptor_t,
    miopenConvFwdAlgorithm_t, const void *, const miopenTensorDescriptor_t, void *, void *, size_t);

typedef struct MiopenApi {
    HMODULE module;
    PFN_miopenCreate create;
    PFN_miopenDestroy destroy;
    PFN_miopenSetStream set_stream;
    PFN_miopenCreateTensorDescriptor create_tensor;
    PFN_miopenDestroyTensorDescriptor destroy_tensor;
    PFN_miopenSetTensorDescriptor set_tensor;
    PFN_miopenCreateConvolutionDescriptor create_conv;
    PFN_miopenDestroyConvolutionDescriptor destroy_conv;
    PFN_miopenInitConvolutionNdDescriptor init_conv_nd;
    PFN_miopenSetConvolutionGroupCount set_group_count;
    PFN_miopenConvolutionForwardGetWorkSpaceSize workspace_size;
    PFN_miopenFindConvolutionForwardAlgorithm find_fwd;
    PFN_miopenConvolutionForwardGetSolutionCount solution_count;
    PFN_miopenConvolutionForwardGetSolution get_solutions;
    PFN_miopenConvolutionForwardGetSolutionWorkspaceSize solution_workspace;
    PFN_miopenConvolutionForwardImmediate conv_fwd_immediate;
    PFN_miopenConvolutionBackwardDataGetSolutionCount bwd_data_solution_count;
    PFN_miopenConvolutionBackwardDataGetSolution bwd_data_get_solutions;
    PFN_miopenConvolutionBackwardDataGetSolutionWorkspaceSize bwd_data_solution_workspace;
    PFN_miopenConvolutionBackwardDataImmediate conv_bwd_data_immediate;
    PFN_miopenConvolutionBackwardWeightsGetSolutionCount bwd_weights_solution_count;
    PFN_miopenConvolutionBackwardWeightsGetSolution bwd_weights_get_solutions;
    PFN_miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize bwd_weights_solution_workspace;
    PFN_miopenConvolutionBackwardWeightsImmediate conv_bwd_weights_immediate;
    PFN_miopenConvolutionForward conv_fwd;
} MiopenApi;

static MiopenApi g_miopen;
static INIT_ONCE g_miopen_once = INIT_ONCE_STATIC_INIT;
static int g_miopen_ready = 0;

static FARPROC bridge_get_proc(HMODULE module, const char *name) {
    return module ? GetProcAddress(module, name) : NULL;
}

static BOOL CALLBACK bridge_load_miopen_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    char configured[MAX_PATH * 4];
    DWORD n;
    HMODULE m = NULL;
    (void)once;
    (void)param;
    (void)ctx;

    memset(&g_miopen, 0, sizeof(g_miopen));
    n = GetEnvironmentVariableA("CUDA_AMD_MIOPEN_DLL", configured, (DWORD)sizeof(configured));
    if (n > 0 && n < sizeof(configured)) {
        m = LoadLibraryA(configured);
    }
    if (!m) m = LoadLibraryA("MIOpen.dll");
    if (!m) m = LoadLibraryA("miopen.dll");
    if (!m) return TRUE;

    g_miopen.module = m;
#define LOAD_REQ(field, symbol) do { \
    FARPROC bridge_proc = bridge_get_proc(m, #symbol); \
    if (!bridge_proc) return TRUE; \
    memcpy(&g_miopen.field, &bridge_proc, sizeof(g_miopen.field)); \
} while (0)
    LOAD_REQ(create, miopenCreate);
    LOAD_REQ(destroy, miopenDestroy);
    LOAD_REQ(set_stream, miopenSetStream);
    LOAD_REQ(create_tensor, miopenCreateTensorDescriptor);
    LOAD_REQ(destroy_tensor, miopenDestroyTensorDescriptor);
    LOAD_REQ(set_tensor, miopenSetTensorDescriptor);
    LOAD_REQ(create_conv, miopenCreateConvolutionDescriptor);
    LOAD_REQ(destroy_conv, miopenDestroyConvolutionDescriptor);
    LOAD_REQ(init_conv_nd, miopenInitConvolutionNdDescriptor);
    LOAD_REQ(set_group_count, miopenSetConvolutionGroupCount);
    LOAD_REQ(workspace_size, miopenConvolutionForwardGetWorkSpaceSize);
    LOAD_REQ(find_fwd, miopenFindConvolutionForwardAlgorithm);
    LOAD_REQ(solution_count, miopenConvolutionForwardGetSolutionCount);
    LOAD_REQ(get_solutions, miopenConvolutionForwardGetSolution);
    LOAD_REQ(solution_workspace, miopenConvolutionForwardGetSolutionWorkspaceSize);
    LOAD_REQ(conv_fwd_immediate, miopenConvolutionForwardImmediate);
    LOAD_REQ(bwd_data_solution_count, miopenConvolutionBackwardDataGetSolutionCount);
    LOAD_REQ(bwd_data_get_solutions, miopenConvolutionBackwardDataGetSolution);
    LOAD_REQ(bwd_data_solution_workspace, miopenConvolutionBackwardDataGetSolutionWorkspaceSize);
    LOAD_REQ(conv_bwd_data_immediate, miopenConvolutionBackwardDataImmediate);
    LOAD_REQ(bwd_weights_solution_count, miopenConvolutionBackwardWeightsGetSolutionCount);
    LOAD_REQ(bwd_weights_get_solutions, miopenConvolutionBackwardWeightsGetSolution);
    LOAD_REQ(bwd_weights_solution_workspace, miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize);
    LOAD_REQ(conv_bwd_weights_immediate, miopenConvolutionBackwardWeightsImmediate);
    LOAD_REQ(conv_fwd, miopenConvolutionForward);
#undef LOAD_REQ
    g_miopen_ready = 1;
    return TRUE;
}

static int bridge_backend_ready(void) {
    InitOnceExecuteOnce(&g_miopen_once, bridge_load_miopen_once, NULL, NULL);
    return g_miopen_ready;
}

static cudnnStatus_t bridge_status(miopenStatus_t s) {
    switch (s) {
        case MIOPEN_STATUS_SUCCESS:
        case MIOPEN_STATUS_GPU_OPERATIONS_SKIPPED:
            return CUDNN_STATUS_SUCCESS;
        case MIOPEN_STATUS_NOT_INITIALIZED:
            return CUDNN_STATUS_NOT_INITIALIZED;
        case MIOPEN_STATUS_ALLOC_FAILED:
            return CUDNN_STATUS_ALLOC_FAILED;
        case MIOPEN_STATUS_INVALID_VALUE:
        case MIOPEN_STATUS_BAD_PARM:
            return CUDNN_STATUS_BAD_PARAM;
        case MIOPEN_STATUS_NOT_IMPLEMENTED:
        case MIOPEN_STATUS_UNSUPPORTED_OP:
            return CUDNN_STATUS_NOT_SUPPORTED;
        case MIOPEN_STATUS_VERSION_MISMATCH:
            return CUDNN_STATUS_VERSION_MISMATCH;
        default:
            return CUDNN_STATUS_EXECUTION_FAILED;
    }
}

static int bridge_dtype_to_miopen(int t, miopenDataType_t *out) {
    if (!out) return 0;
    switch (t) {
        case CUDNN_DATA_FLOAT: *out = MIOPEN_FLOAT; return 1;
        case CUDNN_DATA_DOUBLE: *out = MIOPEN_DOUBLE; return 1;
        case CUDNN_DATA_HALF: *out = MIOPEN_HALF; return 1;
        case CUDNN_DATA_BFLOAT16: *out = MIOPEN_BFLOAT16; return 1;
        case CUDNN_DATA_INT32: *out = MIOPEN_INT32; return 1;
        case CUDNN_DATA_INT8: *out = MIOPEN_INT8; return 1;
        case CUDNN_DATA_INT64: *out = MIOPEN_INT64; return 1;
        default: return 0;
    }
}

static size_t bridge_dtype_size(int t) {
    switch (t) {
        case CUDNN_DATA_FLOAT: return 4;
        case CUDNN_DATA_DOUBLE: return 8;
        case CUDNN_DATA_HALF: return 2;
        case CUDNN_DATA_BFLOAT16: return 2;
        case CUDNN_DATA_INT8: return 1;
        case CUDNN_DATA_INT32: return 4;
        case CUDNN_DATA_INT64: return 8;
        default: return 0;
    }
}

static int bridge_valid_tensor(const BridgeTensor *d) {
    return d && d->magic == BRIDGE_MAGIC_TENSOR && d->nb_dims > 0 && d->nb_dims <= BRIDGE_MAX_DIMS;
}

static int bridge_valid_filter(const BridgeFilter *d) {
    return d && d->magic == BRIDGE_MAGIC_FILTER && d->nb_dims > 0 && d->nb_dims <= BRIDGE_MAX_DIMS;
}

static int bridge_valid_conv(const BridgeConv *d) {
    return d && d->magic == BRIDGE_MAGIC_CONV && d->spatial_dims == 2;
}

static int bridge_contiguous_strides(int nb, const int *dims, int *strides) {
    int i;
    int64_t s = 1;
    if (!dims || !strides || nb <= 0 || nb > BRIDGE_MAX_DIMS) return 0;
    for (i = nb - 1; i >= 0; --i) {
        if (dims[i] <= 0 || s > INT32_MAX) return 0;
        strides[i] = (int)s;
        s *= dims[i];
        if (s > INT32_MAX && i > 0) return 0;
    }
    return 1;
}

static cudnnStatus_t bridge_make_miopen_tensor(
    const BridgeTensor *src, miopenTensorDescriptor_t *out) {
    miopenStatus_t s;
    miopenDataType_t type;
    if (!out || !bridge_valid_tensor(src)) return CUDNN_STATUS_BAD_PARAM;
    *out = NULL;
    if (!bridge_backend_ready()) return CUDNN_STATUS_NOT_INITIALIZED;
    if (!bridge_dtype_to_miopen(src->data_type, &type)) return CUDNN_STATUS_NOT_SUPPORTED;
    s = g_miopen.create_tensor(out);
    if (s != MIOPEN_STATUS_SUCCESS) return bridge_status(s);
    s = g_miopen.set_tensor(*out, type, src->nb_dims, src->dims, src->strides);
    if (s != MIOPEN_STATUS_SUCCESS) {
        g_miopen.destroy_tensor(*out);
        *out = NULL;
        return bridge_status(s);
    }
    return CUDNN_STATUS_SUCCESS;
}

static cudnnStatus_t bridge_make_miopen_filter(
    const BridgeFilter *src, miopenTensorDescriptor_t *out) {
    BridgeTensor t;
    if (!out || !bridge_valid_filter(src)) return CUDNN_STATUS_BAD_PARAM;
    if (src->format != CUDNN_TENSOR_NCHW) return CUDNN_STATUS_NOT_SUPPORTED;
    memset(&t, 0, sizeof(t));
    t.magic = BRIDGE_MAGIC_TENSOR;
    t.data_type = src->data_type;
    t.format = src->format;
    t.nb_dims = src->nb_dims;
    memcpy(t.dims, src->dims, sizeof(int) * src->nb_dims);
    if (!bridge_contiguous_strides(t.nb_dims, t.dims, t.strides)) return CUDNN_STATUS_BAD_PARAM;
    return bridge_make_miopen_tensor(&t, out);
}

static cudnnStatus_t bridge_make_miopen_conv(
    const BridgeConv *src, miopenConvolutionDescriptor_t *out) {
    miopenStatus_t s;
    if (!out || !bridge_valid_conv(src)) return CUDNN_STATUS_BAD_PARAM;
    *out = NULL;
    if (src->mode != CUDNN_CROSS_CORRELATION) return CUDNN_STATUS_NOT_SUPPORTED;
    if (!bridge_backend_ready()) return CUDNN_STATUS_NOT_INITIALIZED;
    s = g_miopen.create_conv(out);
    if (s != MIOPEN_STATUS_SUCCESS) return bridge_status(s);
    s = g_miopen.init_conv_nd(
        *out, src->spatial_dims, src->pad, src->stride, src->dilation, MIOPEN_CONVOLUTION);
    if (s == MIOPEN_STATUS_SUCCESS && src->group_count != 1) {
        s = g_miopen.set_group_count(*out, src->group_count);
    }
    if (s != MIOPEN_STATUS_SUCCESS) {
        g_miopen.destroy_conv(*out);
        *out = NULL;
        return bridge_status(s);
    }
    return CUDNN_STATUS_SUCCESS;
}

static void bridge_destroy_miopen_tensor(miopenTensorDescriptor_t d) {
    if (d && bridge_backend_ready()) g_miopen.destroy_tensor(d);
}

static void bridge_destroy_miopen_conv(miopenConvolutionDescriptor_t d) {
    if (d && bridge_backend_ready()) g_miopen.destroy_conv(d);
}

static int bridge_map_algo(int cudnn_algo, int *miopen_algo) {
    if (!miopen_algo) return 0;
    switch (cudnn_algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            *miopen_algo = MIOPEN_FWD_IMPLICIT_GEMM; return 1;
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            *miopen_algo = MIOPEN_FWD_GEMM; return 1;
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            *miopen_algo = MIOPEN_FWD_DIRECT; return 1;
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            *miopen_algo = MIOPEN_FWD_FFT; return 1;
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            *miopen_algo = MIOPEN_FWD_WINOGRAD; return 1;
        default:
            return 0;
    }
}

static int bridge_unmap_algo(int miopen_algo, int *cudnn_algo) {
    if (!cudnn_algo) return 0;
    switch (miopen_algo) {
        case MIOPEN_FWD_GEMM:
            *cudnn_algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM; return 1;
        case MIOPEN_FWD_DIRECT:
            *cudnn_algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT; return 1;
        case MIOPEN_FWD_FFT:
            *cudnn_algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT; return 1;
        case MIOPEN_FWD_WINOGRAD:
            *cudnn_algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD; return 1;
        case MIOPEN_FWD_IMPLICIT_GEMM:
            *cudnn_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; return 1;
        default:
            return 0;
    }
}

static int bridge_default_scalars(const BridgeConv *conv, const void *alpha, const void *beta) {
    if (!conv || !alpha || !beta) return 0;
    if (conv->compute_type == CUDNN_DATA_DOUBLE) {
        const double a = *(const double *)alpha;
        const double b = *(const double *)beta;
        return a == 1.0 && b == 0.0;
    }
    if (conv->compute_type == CUDNN_DATA_FLOAT) {
        const float a = *(const float *)alpha;
        const float b = *(const float *)beta;
        return a == 1.0f && b == 0.0f;
    }
    return 0;
}

static cudnnStatus_t bridge_pick_solution(
    miopenHandle_t handle, miopenTensorDescriptor_t mw, miopenTensorDescriptor_t mx,
    miopenConvolutionDescriptor_t mc, miopenTensorDescriptor_t my, int cudnn_algo,
    uint64_t *solution_id, size_t *workspace_size, float *time_ms) {
    MiopenConvSolution *solutions = NULL;
    size_t count = 0, returned = 0, query_count, i;
    int mapped = -1;
    miopenStatus_t ms;
    cudnnStatus_t cs = CUDNN_STATUS_NOT_SUPPORTED;

    if (!handle || !mw || !mx || !mc || !my || !solution_id || !workspace_size)
        return CUDNN_STATUS_BAD_PARAM;
    if (!bridge_map_algo(cudnn_algo, &mapped))
        return CUDNN_STATUS_NOT_SUPPORTED;

    ms = g_miopen.solution_count(handle, mw, mx, mc, my, &count);
    if (ms != MIOPEN_STATUS_SUCCESS) return bridge_status(ms);
    if (count == 0) return CUDNN_STATUS_NOT_SUPPORTED;

    query_count = count > 64 ? 64 : count;
    solutions = (MiopenConvSolution *)calloc(query_count, sizeof(*solutions));
    if (!solutions) return CUDNN_STATUS_ALLOC_FAILED;

    ms = g_miopen.get_solutions(handle, mw, mx, mc, my, query_count, &returned, solutions);
    if (ms != MIOPEN_STATUS_SUCCESS) {
        cs = bridge_status(ms);
        goto done;
    }

    for (i = 0; i < returned && i < query_count; ++i) {
        size_t ws;
        if (solutions[i].algorithm != mapped) continue;
        ws = solutions[i].workspace_size;
        ms = g_miopen.solution_workspace(
            handle, mw, mx, mc, my, solutions[i].solution_id, &ws);
        if (ms != MIOPEN_STATUS_SUCCESS) continue;
        *solution_id = solutions[i].solution_id;
        *workspace_size = ws;
        if (time_ms) *time_ms = solutions[i].time;
        cs = CUDNN_STATUS_SUCCESS;
        break;
    }

done:
    free(solutions);
    return cs;
}

static int bridge_unmap_bwd_data_algo(int miopen_algo, int *cudnn_algo) {
    if (!cudnn_algo) return 0;
    switch (miopen_algo) {
        case MIOPEN_BWD_DATA_GEMM:
        case MIOPEN_BWD_DATA_IMPLICIT_GEMM:
            *cudnn_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            return 1;
        case MIOPEN_BWD_DATA_DIRECT:
            *cudnn_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            return 1;
        case MIOPEN_BWD_DATA_FFT:
            *cudnn_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
            return 1;
        case MIOPEN_BWD_DATA_WINOGRAD:
            *cudnn_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
            return 1;
        default:
            return 0;
    }
}

static int bridge_unmap_bwd_filter_algo(int miopen_algo, int *cudnn_algo) {
    if (!cudnn_algo) return 0;
    switch (miopen_algo) {
        case MIOPEN_BWD_WEIGHTS_GEMM:
        case MIOPEN_BWD_WEIGHTS_IMPLICIT_GEMM:
            *cudnn_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
            return 1;
        case MIOPEN_BWD_WEIGHTS_DIRECT:
            *cudnn_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
            return 1;
        case MIOPEN_BWD_WEIGHTS_WINOGRAD:
            *cudnn_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
            return 1;
        default:
            return 0;
    }
}

static cudnnStatus_t bridge_pick_bwd_data_solution(
    miopenHandle_t handle, miopenTensorDescriptor_t mdy, miopenTensorDescriptor_t mw,
    miopenConvolutionDescriptor_t mc, miopenTensorDescriptor_t mdx, int cudnn_algo,
    uint64_t *solution_id, size_t *workspace_size, float *time_ms) {
    MiopenConvSolution *solutions = NULL;
    size_t count = 0, returned = 0, query_count, i;
    miopenStatus_t ms;
    cudnnStatus_t cs = CUDNN_STATUS_NOT_SUPPORTED;

    if (!handle || !mdy || !mw || !mc || !mdx || !solution_id || !workspace_size)
        return CUDNN_STATUS_BAD_PARAM;

    ms = g_miopen.bwd_data_solution_count(handle, mdy, mw, mc, mdx, &count);
    if (ms != MIOPEN_STATUS_SUCCESS) return bridge_status(ms);
    if (count == 0) return CUDNN_STATUS_NOT_SUPPORTED;

    query_count = count > 64 ? 64 : count;
    solutions = (MiopenConvSolution *)calloc(query_count, sizeof(*solutions));
    if (!solutions) return CUDNN_STATUS_ALLOC_FAILED;

    ms = g_miopen.bwd_data_get_solutions(
        handle, mdy, mw, mc, mdx, query_count, &returned, solutions);
    if (ms != MIOPEN_STATUS_SUCCESS) {
        cs = bridge_status(ms);
        goto done;
    }

    for (i = 0; i < returned && i < query_count; ++i) {
        int mapped = -1;
        size_t ws = solutions[i].workspace_size;
        if (!bridge_unmap_bwd_data_algo(solutions[i].algorithm, &mapped) || mapped != cudnn_algo)
            continue;
        ms = g_miopen.bwd_data_solution_workspace(
            handle, mdy, mw, mc, mdx, solutions[i].solution_id, &ws);
        if (ms != MIOPEN_STATUS_SUCCESS) continue;
        *solution_id = solutions[i].solution_id;
        *workspace_size = ws;
        if (time_ms) *time_ms = solutions[i].time;
        cs = CUDNN_STATUS_SUCCESS;
        break;
    }

done:
    free(solutions);
    return cs;
}

static cudnnStatus_t bridge_pick_bwd_filter_solution(
    miopenHandle_t handle, miopenTensorDescriptor_t mdy, miopenTensorDescriptor_t mx,
    miopenConvolutionDescriptor_t mc, miopenTensorDescriptor_t mdw, int cudnn_algo,
    uint64_t *solution_id, size_t *workspace_size, float *time_ms) {
    MiopenConvSolution *solutions = NULL;
    size_t count = 0, returned = 0, query_count, i;
    miopenStatus_t ms;
    cudnnStatus_t cs = CUDNN_STATUS_NOT_SUPPORTED;

    if (!handle || !mdy || !mx || !mc || !mdw || !solution_id || !workspace_size)
        return CUDNN_STATUS_BAD_PARAM;

    ms = g_miopen.bwd_weights_solution_count(handle, mdy, mx, mc, mdw, &count);
    if (ms != MIOPEN_STATUS_SUCCESS) return bridge_status(ms);
    if (count == 0) return CUDNN_STATUS_NOT_SUPPORTED;

    query_count = count > 64 ? 64 : count;
    solutions = (MiopenConvSolution *)calloc(query_count, sizeof(*solutions));
    if (!solutions) return CUDNN_STATUS_ALLOC_FAILED;

    ms = g_miopen.bwd_weights_get_solutions(
        handle, mdy, mx, mc, mdw, query_count, &returned, solutions);
    if (ms != MIOPEN_STATUS_SUCCESS) {
        cs = bridge_status(ms);
        goto done;
    }

    for (i = 0; i < returned && i < query_count; ++i) {
        int mapped = -1;
        size_t ws = solutions[i].workspace_size;
        if (!bridge_unmap_bwd_filter_algo(solutions[i].algorithm, &mapped) || mapped != cudnn_algo)
            continue;
        ms = g_miopen.bwd_weights_solution_workspace(
            handle, mdy, mx, mc, mdw, solutions[i].solution_id, &ws);
        if (ms != MIOPEN_STATUS_SUCCESS) continue;
        *solution_id = solutions[i].solution_id;
        *workspace_size = ws;
        if (time_ms) *time_ms = solutions[i].time;
        cs = CUDNN_STATUS_SUCCESS;
        break;
    }

done:
    free(solutions);
    return cs;
}

static cudnnStatus_t bridge_validate_conv_shapes(
    const BridgeTensor *x, const BridgeFilter *w, const BridgeConv *c, const BridgeTensor *y) {
    int64_t oh, ow;
    if (!bridge_valid_tensor(x) || !bridge_valid_filter(w) || !bridge_valid_conv(c) || !bridge_valid_tensor(y))
        return CUDNN_STATUS_BAD_PARAM;
    if (x->nb_dims != 4 || w->nb_dims != 4 || y->nb_dims != 4)
        return CUDNN_STATUS_NOT_SUPPORTED;
    if (x->data_type != w->data_type || x->data_type != y->data_type)
        return CUDNN_STATUS_BAD_PARAM;
    if (c->compute_type != CUDNN_DATA_FLOAT && c->compute_type != CUDNN_DATA_DOUBLE)
        return CUDNN_STATUS_NOT_SUPPORTED;
    if (c->mode != CUDNN_CROSS_CORRELATION) return CUDNN_STATUS_NOT_SUPPORTED;
    if (c->group_count <= 0 || x->dims[1] != w->dims[1] * c->group_count)
        return CUDNN_STATUS_BAD_PARAM;
    if (w->dims[0] % c->group_count != 0 || y->dims[0] != x->dims[0] || y->dims[1] != w->dims[0])
        return CUDNN_STATUS_BAD_PARAM;
    oh = 1 + ((int64_t)x->dims[2] + 2LL * c->pad[0] -
              (int64_t)c->dilation[0] * (w->dims[2] - 1) - 1) / c->stride[0];
    ow = 1 + ((int64_t)x->dims[3] + 2LL * c->pad[1] -
              (int64_t)c->dilation[1] * (w->dims[3] - 1) - 1) / c->stride[1];
    if (oh <= 0 || ow <= 0 || y->dims[2] != oh || y->dims[3] != ow)
        return CUDNN_STATUS_BAD_PARAM;
    return CUDNN_STATUS_SUCCESS;
}

/* ---- cuDNN handle ------------------------------------------------------- */

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    BridgeHandle *h;
    miopenStatus_t s;
    if (!handle) return CUDNN_STATUS_BAD_PARAM;
    *handle = NULL;
    if (!bridge_backend_ready()) return CUDNN_STATUS_NOT_INITIALIZED;
    h = (BridgeHandle *)calloc(1, sizeof(*h));
    if (!h) return CUDNN_STATUS_ALLOC_FAILED;
    s = g_miopen.create(&h->miopen);
    if (s != MIOPEN_STATUS_SUCCESS) {
        free(h);
        return bridge_status(s);
    }
    h->magic = BRIDGE_MAGIC_HANDLE;
    h->cuda_stream = NULL;
    s = g_miopen.set_stream(h->miopen, NULL);
    if (s != MIOPEN_STATUS_SUCCESS) {
        g_miopen.destroy(h->miopen);
        free(h);
        return bridge_status(s);
    }
    *handle = h;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    miopenStatus_t s;
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE) return CUDNN_STATUS_BAD_PARAM;
    s = g_miopen.destroy(handle->miopen);
    handle->magic = 0;
    free(handle);
    return bridge_status(s);
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, void *streamId) {
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE) return CUDNN_STATUS_BAD_PARAM;
    /*
     * ZLUDA CUDA stream handles are opaque and are not proven to be raw
     * hipStream_t values. Only the default stream is safe until an explicit
     * unwrap/translation path is validated.
     */
    if (streamId != NULL) return CUDNN_STATUS_NOT_SUPPORTED;
    if (g_miopen.set_stream(handle->miopen, NULL) != MIOPEN_STATUS_SUCCESS)
        return CUDNN_STATUS_EXECUTION_FAILED;
    handle->cuda_stream = NULL;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, void **streamId) {
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !streamId) return CUDNN_STATUS_BAD_PARAM;
    *streamId = handle->cuda_stream;
    return CUDNN_STATUS_SUCCESS;
}

/* ---- Tensor descriptors ------------------------------------------------ */

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    BridgeTensor *d;
    if (!tensorDesc) return CUDNN_STATUS_BAD_PARAM;
    d = (BridgeTensor *)calloc(1, sizeof(*d));
    if (!d) return CUDNN_STATUS_ALLOC_FAILED;
    d->magic = BRIDGE_MAGIC_TENSOR;
    d->format = CUDNN_TENSOR_NCHW;
    *tensorDesc = d;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    if (!tensorDesc || tensorDesc->magic != BRIDGE_MAGIC_TENSOR) return CUDNN_STATUS_BAD_PARAM;
    tensorDesc->magic = 0;
    free(tensorDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType,
    int n, int c, int h, int w) {
    if (!tensorDesc || tensorDesc->magic != BRIDGE_MAGIC_TENSOR) return CUDNN_STATUS_BAD_PARAM;
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) return CUDNN_STATUS_BAD_PARAM;
    if (format != CUDNN_TENSOR_NCHW && format != CUDNN_TENSOR_NHWC) return CUDNN_STATUS_NOT_SUPPORTED;
    if (!bridge_dtype_size(dataType)) return CUDNN_STATUS_NOT_SUPPORTED;
    tensorDesc->data_type = dataType;
    tensorDesc->format = format;
    tensorDesc->nb_dims = 4;
    tensorDesc->dims[0] = n; tensorDesc->dims[1] = c; tensorDesc->dims[2] = h; tensorDesc->dims[3] = w;
    if (format == CUDNN_TENSOR_NCHW) {
        tensorDesc->strides[3] = 1;
        tensorDesc->strides[2] = w;
        tensorDesc->strides[1] = h * w;
        tensorDesc->strides[0] = c * h * w;
    } else {
        tensorDesc->strides[1] = 1;
        tensorDesc->strides[3] = c;
        tensorDesc->strides[2] = w * c;
        tensorDesc->strides[0] = h * w * c;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType,
    int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride) {
    if (!tensorDesc || tensorDesc->magic != BRIDGE_MAGIC_TENSOR) return CUDNN_STATUS_BAD_PARAM;
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 ||
        nStride <= 0 || cStride <= 0 || hStride <= 0 || wStride <= 0)
        return CUDNN_STATUS_BAD_PARAM;
    if (!bridge_dtype_size(dataType)) return CUDNN_STATUS_NOT_SUPPORTED;
    tensorDesc->data_type = dataType;
    tensorDesc->format = -1;
    tensorDesc->nb_dims = 4;
    tensorDesc->dims[0] = n; tensorDesc->dims[1] = c; tensorDesc->dims[2] = h; tensorDesc->dims[3] = w;
    tensorDesc->strides[0] = nStride; tensorDesc->strides[1] = cStride;
    tensorDesc->strides[2] = hStride; tensorDesc->strides[3] = wStride;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims,
    const int dimA[], const int strideA[]) {
    int i;
    if (!tensorDesc || tensorDesc->magic != BRIDGE_MAGIC_TENSOR || !dimA || !strideA)
        return CUDNN_STATUS_BAD_PARAM;
    if (nbDims <= 0 || nbDims > BRIDGE_MAX_DIMS || !bridge_dtype_size(dataType))
        return CUDNN_STATUS_NOT_SUPPORTED;
    for (i = 0; i < nbDims; ++i) {
        if (dimA[i] <= 0 || strideA[i] <= 0) return CUDNN_STATUS_BAD_PARAM;
    }
    tensorDesc->data_type = dataType;
    tensorDesc->format = -1;
    tensorDesc->nb_dims = nbDims;
    memcpy(tensorDesc->dims, dimA, sizeof(int) * nbDims);
    memcpy(tensorDesc->strides, strideA, sizeof(int) * nbDims);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(
    const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType,
    int *n, int *c, int *h, int *w, int *nStride, int *cStride, int *hStride, int *wStride) {
    if (!bridge_valid_tensor(tensorDesc) || tensorDesc->nb_dims != 4) return CUDNN_STATUS_BAD_PARAM;
    if (dataType) *dataType = tensorDesc->data_type;
    if (n) *n = tensorDesc->dims[0];
    if (c) *c = tensorDesc->dims[1];
    if (h) *h = tensorDesc->dims[2];
    if (w) *w = tensorDesc->dims[3];
    if (nStride) *nStride = tensorDesc->strides[0];
    if (cStride) *cStride = tensorDesc->strides[1];
    if (hStride) *hStride = tensorDesc->strides[2];
    if (wStride) *wStride = tensorDesc->strides[3];
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(
    const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested, cudnnDataType_t *dataType,
    int *nbDims, int dimA[], int strideA[]) {
    int n;
    if (!bridge_valid_tensor(tensorDesc) || nbDimsRequested < 0) return CUDNN_STATUS_BAD_PARAM;
    if (dataType) *dataType = tensorDesc->data_type;
    if (nbDims) *nbDims = tensorDesc->nb_dims;
    n = nbDimsRequested < tensorDesc->nb_dims ? nbDimsRequested : tensorDesc->nb_dims;
    if (dimA) memcpy(dimA, tensorDesc->dims, sizeof(int) * n);
    if (strideA) memcpy(strideA, tensorDesc->strides, sizeof(int) * n);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
    int i;
    int64_t max_index = 0;
    size_t element;
    if (!bridge_valid_tensor(tensorDesc) || !size) return CUDNN_STATUS_BAD_PARAM;
    element = bridge_dtype_size(tensorDesc->data_type);
    if (!element) return CUDNN_STATUS_NOT_SUPPORTED;
    for (i = 0; i < tensorDesc->nb_dims; ++i)
        max_index += (int64_t)(tensorDesc->dims[i] - 1) * tensorDesc->strides[i];
    if (max_index < 0 || (uint64_t)(max_index + 1) > SIZE_MAX / element)
        return CUDNN_STATUS_INVALID_VALUE;
    *size = (size_t)(max_index + 1) * element;
    return CUDNN_STATUS_SUCCESS;
}

/* ---- Filter descriptors ------------------------------------------------ */

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    BridgeFilter *d;
    if (!filterDesc) return CUDNN_STATUS_BAD_PARAM;
    d = (BridgeFilter *)calloc(1, sizeof(*d));
    if (!d) return CUDNN_STATUS_ALLOC_FAILED;
    d->magic = BRIDGE_MAGIC_FILTER;
    d->format = CUDNN_TENSOR_NCHW;
    *filterDesc = d;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    if (!filterDesc || filterDesc->magic != BRIDGE_MAGIC_FILTER) return CUDNN_STATUS_BAD_PARAM;
    filterDesc->magic = 0;
    free(filterDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int k, int c, int h, int w) {
    int dims[4] = {k, c, h, w};
    return cudnnSetFilterNdDescriptor(filterDesc, dataType, format, 4, dims);
}

cudnnStatus_t cudnnSetFilterNdDescriptor(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int nbDims, const int filterDimA[]) {
    int i;
    if (!filterDesc || filterDesc->magic != BRIDGE_MAGIC_FILTER || !filterDimA)
        return CUDNN_STATUS_BAD_PARAM;
    if (format != CUDNN_TENSOR_NCHW) return CUDNN_STATUS_NOT_SUPPORTED;
    if (nbDims <= 0 || nbDims > BRIDGE_MAX_DIMS || !bridge_dtype_size(dataType))
        return CUDNN_STATUS_NOT_SUPPORTED;
    for (i = 0; i < nbDims; ++i) if (filterDimA[i] <= 0) return CUDNN_STATUS_BAD_PARAM;
    filterDesc->data_type = dataType;
    filterDesc->format = format;
    filterDesc->nb_dims = nbDims;
    memcpy(filterDesc->dims, filterDimA, sizeof(int) * nbDims);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(
    const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *k, int *c, int *h, int *w) {
    if (!bridge_valid_filter(filterDesc) || filterDesc->nb_dims != 4) return CUDNN_STATUS_BAD_PARAM;
    if (dataType) *dataType = filterDesc->data_type;
    if (format) *format = filterDesc->format;
    if (k) *k = filterDesc->dims[0];
    if (c) *c = filterDesc->dims[1];
    if (h) *h = filterDesc->dims[2];
    if (w) *w = filterDesc->dims[3];
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterNdDescriptor(
    const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
    int n;
    if (!bridge_valid_filter(filterDesc) || nbDimsRequested < 0) return CUDNN_STATUS_BAD_PARAM;
    if (dataType) *dataType = filterDesc->data_type;
    if (format) *format = filterDesc->format;
    if (nbDims) *nbDims = filterDesc->nb_dims;
    n = nbDimsRequested < filterDesc->nb_dims ? nbDimsRequested : filterDesc->nb_dims;
    if (filterDimA) memcpy(filterDimA, filterDesc->dims, sizeof(int) * n);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size) {
    int i;
    size_t element;
    uint64_t count = 1;
    if (!bridge_valid_filter(filterDesc) || !size) return CUDNN_STATUS_BAD_PARAM;
    element = bridge_dtype_size(filterDesc->data_type);
    if (!element) return CUDNN_STATUS_NOT_SUPPORTED;
    for (i = 0; i < filterDesc->nb_dims; ++i) {
        count *= (uint64_t)filterDesc->dims[i];
        if (count > SIZE_MAX / element) return CUDNN_STATUS_INVALID_VALUE;
    }
    *size = (size_t)count * element;
    return CUDNN_STATUS_SUCCESS;
}

/* ---- Convolution descriptors ------------------------------------------- */

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    BridgeConv *d;
    if (!convDesc) return CUDNN_STATUS_BAD_PARAM;
    d = (BridgeConv *)calloc(1, sizeof(*d));
    if (!d) return CUDNN_STATUS_ALLOC_FAILED;
    d->magic = BRIDGE_MAGIC_CONV;
    d->spatial_dims = 0;
    d->group_count = 1;
    d->math_type = 0;
    *convDesc = d;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    if (!convDesc || convDesc->magic != BRIDGE_MAGIC_CONV) return CUDNN_STATUS_BAD_PARAM;
    convDesc->magic = 0;
    free(convDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v,
    int dilation_h, int dilation_w, cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
    int pad[2] = {pad_h, pad_w};
    int stride[2] = {u, v};
    int dilation[2] = {dilation_h, dilation_w};
    return cudnnSetConvolutionNdDescriptor(convDesc, 2, pad, stride, dilation, mode, computeType);
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[],
    const int filterStrideA[], const int dilationA[], cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType) {
    int i;
    if (!convDesc || convDesc->magic != BRIDGE_MAGIC_CONV ||
        !padA || !filterStrideA || !dilationA) return CUDNN_STATUS_BAD_PARAM;
    if (arrayLength != 2) return CUDNN_STATUS_NOT_SUPPORTED;
    if (mode != CUDNN_CROSS_CORRELATION) return CUDNN_STATUS_NOT_SUPPORTED;
    if (computeType != CUDNN_DATA_FLOAT && computeType != CUDNN_DATA_DOUBLE)
        return CUDNN_STATUS_NOT_SUPPORTED;
    for (i = 0; i < arrayLength; ++i) {
        if (padA[i] < 0 || filterStrideA[i] <= 0 || dilationA[i] <= 0)
            return CUDNN_STATUS_BAD_PARAM;
    }
    convDesc->spatial_dims = arrayLength;
    memcpy(convDesc->pad, padA, sizeof(int) * arrayLength);
    memcpy(convDesc->stride, filterStrideA, sizeof(int) * arrayLength);
    memcpy(convDesc->dilation, dilationA, sizeof(int) * arrayLength);
    convDesc->mode = mode;
    convDesc->compute_type = computeType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
    if (!convDesc || convDesc->magic != BRIDGE_MAGIC_CONV || groupCount <= 0)
        return CUDNN_STATUS_BAD_PARAM;
    convDesc->group_count = groupCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount(const cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
    if (!convDesc || convDesc->magic != BRIDGE_MAGIC_CONV || !groupCount)
        return CUDNN_STATUS_BAD_PARAM;
    *groupCount = convDesc->group_count;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
    if (!convDesc || convDesc->magic != BRIDGE_MAGIC_CONV) return CUDNN_STATUS_BAD_PARAM;
    if (mathType < 0 || mathType > 3) return CUDNN_STATUS_BAD_PARAM;
    /*
     * cuDNN mathType is an optimization/precision hint. MIOpen owns backend
     * selection here; preserve the requested value for API round-tripping but
     * never claim that NVIDIA tensor-core behavior is reproduced.
     */
    convDesc->math_type = mathType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType(const cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
    if (!convDesc || convDesc->magic != BRIDGE_MAGIC_CONV || !mathType)
        return CUDNN_STATUS_BAD_PARAM;
    *mathType = convDesc->math_type;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w, int *u, int *v,
    int *dilation_h, int *dilation_w, cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
    if (!bridge_valid_conv(convDesc)) return CUDNN_STATUS_BAD_PARAM;
    if (pad_h) *pad_h = convDesc->pad[0];
    if (pad_w) *pad_w = convDesc->pad[1];
    if (u) *u = convDesc->stride[0];
    if (v) *v = convDesc->stride[1];
    if (dilation_h) *dilation_h = convDesc->dilation[0];
    if (dilation_w) *dilation_w = convDesc->dilation[1];
    if (mode) *mode = convDesc->mode;
    if (computeType) *computeType = convDesc->compute_type;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested, int *arrayLength,
    int padA[], int strideA[], int dilationA[], cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType) {
    int n;
    if (!convDesc || convDesc->magic != BRIDGE_MAGIC_CONV || arrayLengthRequested < 0)
        return CUDNN_STATUS_BAD_PARAM;
    if (arrayLength) *arrayLength = convDesc->spatial_dims;
    n = arrayLengthRequested < convDesc->spatial_dims ? arrayLengthRequested : convDesc->spatial_dims;
    if (padA) memcpy(padA, convDesc->pad, sizeof(int) * n);
    if (strideA) memcpy(strideA, convDesc->stride, sizeof(int) * n);
    if (dilationA) memcpy(dilationA, convDesc->dilation, sizeof(int) * n);
    if (mode) *mode = convDesc->mode;
    if (computeType) *computeType = convDesc->compute_type;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {
    int out[4];
    cudnnStatus_t s = cudnnGetConvolutionNdForwardOutputDim(
        convDesc, inputTensorDesc, filterDesc, 4, out);
    if (s != CUDNN_STATUS_SUCCESS) return s;
    if (n) *n = out[0];
    if (c) *c = out[1];
    if (h) *h = out[2];
    if (w) *w = out[3];
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims, int tensorOuputDimA[]) {
    int64_t oh, ow;
    if (!bridge_valid_conv(convDesc) || !bridge_valid_tensor(inputTensorDesc) ||
        !bridge_valid_filter(filterDesc) || !tensorOuputDimA) return CUDNN_STATUS_BAD_PARAM;
    if (nbDims < 4 || inputTensorDesc->nb_dims != 4 || filterDesc->nb_dims != 4)
        return CUDNN_STATUS_NOT_SUPPORTED;
    if (convDesc->mode != CUDNN_CROSS_CORRELATION) return CUDNN_STATUS_NOT_SUPPORTED;
    if (inputTensorDesc->dims[1] != filterDesc->dims[1] * convDesc->group_count)
        return CUDNN_STATUS_BAD_PARAM;
    oh = 1 + ((int64_t)inputTensorDesc->dims[2] + 2LL * convDesc->pad[0] -
              (int64_t)convDesc->dilation[0] * (filterDesc->dims[2] - 1) - 1) /
             convDesc->stride[0];
    ow = 1 + ((int64_t)inputTensorDesc->dims[3] + 2LL * convDesc->pad[1] -
              (int64_t)convDesc->dilation[1] * (filterDesc->dims[3] - 1) - 1) /
             convDesc->stride[1];
    if (oh <= 0 || ow <= 0 || oh > INT32_MAX || ow > INT32_MAX)
        return CUDNN_STATUS_BAD_PARAM;
    tensorOuputDimA[0] = inputTensorDesc->dims[0];
    tensorOuputDimA[1] = filterDesc->dims[0];
    tensorOuputDimA[2] = (int)oh;
    tensorOuputDimA[3] = (int)ow;
    return CUDNN_STATUS_SUCCESS;
}

/* ---- Convolution execution --------------------------------------------- */

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(
    cudnnHandle_t handle, int *count) {
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !count)
        return CUDNN_STATUS_BAD_PARAM;
    /*
     * Five MIOpen forward algorithm families can be mapped without changing
     * semantics. This is an upper bound, not a claim that every family is
     * applicable to every convolution descriptor.
     */
    *count = 5;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, int requestedAlgoCount, int *returnedAlgoCount,
    CudnnConvolutionFwdAlgoPerf *perfResults) {
    miopenTensorDescriptor_t mx = NULL, mw = NULL, my = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    MiopenConvSolution *solutions = NULL;
    size_t solution_count = 0, returned_solutions = 0, query_count, i;
    int out_count = 0, mapped;
    cudnnStatus_t cs;
    miopenStatus_t ms;

    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !returnedAlgoCount)
        return CUDNN_STATUS_BAD_PARAM;
    *returnedAlgoCount = 0;
    if (requestedAlgoCount < 0 || (requestedAlgoCount > 0 && !perfResults))
        return CUDNN_STATUS_BAD_PARAM;
    if (requestedAlgoCount == 0) return CUDNN_STATUS_SUCCESS;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;

    cs = bridge_validate_conv_shapes(xDesc, wDesc, convDesc, yDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    cs = bridge_make_miopen_tensor(xDesc, &mx); if (cs) goto done;
    cs = bridge_make_miopen_filter(wDesc, &mw); if (cs) goto done;
    cs = bridge_make_miopen_tensor(yDesc, &my); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;

    ms = g_miopen.solution_count(handle->miopen, mw, mx, mc, my, &solution_count);
    if (ms != MIOPEN_STATUS_SUCCESS) {
        cs = bridge_status(ms);
        goto done;
    }
    if (solution_count == 0) {
        cs = CUDNN_STATUS_NOT_SUPPORTED;
        goto done;
    }

    query_count = solution_count;
    if (query_count > 64) query_count = 64;
    solutions = (MiopenConvSolution *)calloc(query_count, sizeof(*solutions));
    if (!solutions) {
        cs = CUDNN_STATUS_ALLOC_FAILED;
        goto done;
    }
    ms = g_miopen.get_solutions(
        handle->miopen, mw, mx, mc, my, query_count, &returned_solutions, solutions);
    if (ms != MIOPEN_STATUS_SUCCESS) {
        cs = bridge_status(ms);
        goto done;
    }

    for (i = 0; i < returned_solutions && i < query_count && out_count < requestedAlgoCount; ++i) {
        size_t workspace = solutions[i].workspace_size;
        int j, duplicate = 0;
        if (!bridge_unmap_algo(solutions[i].algorithm, &mapped)) continue;

        /* cuDNN v7 reports algorithm families, not multiple solver IDs. */
        for (j = 0; j < out_count; ++j) {
            if (perfResults[j].algo == mapped) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) continue;

        /*
         * Ask MIOpen for the exact workspace associated with this solver ID.
         * If the query fails, do not publish that candidate as usable.
         */
        ms = g_miopen.solution_workspace(
            handle->miopen, mw, mx, mc, my, solutions[i].solution_id, &workspace);
        if (ms != MIOPEN_STATUS_SUCCESS) continue;

        memset(&perfResults[out_count], 0, sizeof(perfResults[out_count]));
        perfResults[out_count].algo = mapped;
        perfResults[out_count].status = CUDNN_STATUS_SUCCESS;
        perfResults[out_count].time = solutions[i].time;
        perfResults[out_count].memory = workspace;
        /*
         * MIOpen's solution record does not expose cuDNN's determinism flag.
         * Mark unknown solutions NON_DETERMINISTIC instead of guessing.
         */
        perfResults[out_count].determinism = 0;
        perfResults[out_count].mathType = convDesc->math_type;
        ++out_count;
    }

    *returnedAlgoCount = out_count;
    cs = out_count > 0 ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_SUPPORTED;
done:
    free(solutions);
    bridge_destroy_miopen_tensor(mx);
    bridge_destroy_miopen_tensor(mw);
    bridge_destroy_miopen_tensor(my);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo, size_t *sizeInBytes) {
    miopenTensorDescriptor_t mx = NULL, mw = NULL, my = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    uint64_t solution_id = 0;
    cudnnStatus_t cs;
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !sizeInBytes)
        return CUDNN_STATUS_BAD_PARAM;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;
    cs = bridge_validate_conv_shapes(xDesc, wDesc, convDesc, yDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;

    cs = bridge_make_miopen_tensor(xDesc, &mx); if (cs) goto done;
    cs = bridge_make_miopen_filter(wDesc, &mw); if (cs) goto done;
    cs = bridge_make_miopen_tensor(yDesc, &my); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;

    cs = bridge_pick_solution(
        handle->miopen, mw, mx, mc, my, algo, &solution_id, sizeInBytes, NULL);
done:
    bridge_destroy_miopen_tensor(mx);
    bridge_destroy_miopen_tensor(mw);
    bridge_destroy_miopen_tensor(my);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
    miopenTensorDescriptor_t mx = NULL, mw = NULL, my = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    uint64_t solution_id = 0;
    size_t required_workspace = 0;
    cudnnStatus_t cs;
    miopenStatus_t ms;

    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !alpha || !beta || !x || !w || !y)
        return CUDNN_STATUS_BAD_PARAM;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;
    cs = bridge_validate_conv_shapes(xDesc, wDesc, convDesc, yDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    /*
     * MIOpen's Immediate API writes the convolution result directly and has
     * no alpha/beta parameters. Preserve cuDNN semantics by supporting only
     * the dominant framework case alpha=1, beta=0 until scaling is explicitly
     * implemented and validated.
     */
    if (!bridge_default_scalars(convDesc, alpha, beta))
        return CUDNN_STATUS_NOT_SUPPORTED;

    cs = bridge_make_miopen_tensor(xDesc, &mx); if (cs) goto done;
    cs = bridge_make_miopen_filter(wDesc, &mw); if (cs) goto done;
    cs = bridge_make_miopen_tensor(yDesc, &my); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;

    cs = bridge_pick_solution(
        handle->miopen, mw, mx, mc, my, algo, &solution_id, &required_workspace, NULL);
    if (cs != CUDNN_STATUS_SUCCESS) goto done;
    if (required_workspace > workSpaceSizeInBytes) {
        cs = CUDNN_STATUS_BAD_PARAM;
        goto done;
    }
    if (required_workspace > 0 && !workSpace) {
        cs = CUDNN_STATUS_BAD_PARAM;
        goto done;
    }

    ms = g_miopen.conv_fwd_immediate(
        handle->miopen, mw, w, mx, x, mc, my, y,
        workSpace, workSpaceSizeInBytes, solution_id);
    cs = bridge_status(ms);
done:
    bridge_destroy_miopen_tensor(mx);
    bridge_destroy_miopen_tensor(mw);
    bridge_destroy_miopen_tensor(my);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

/* ---- Backward convolution execution ------------------------------------ */

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, int *count) {
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !count)
        return CUDNN_STATUS_BAD_PARAM;
    *count = 4;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, int requestedAlgoCount, int *returnedAlgoCount,
    CudnnConvolutionBwdDataAlgoPerf *perfResults) {
    miopenTensorDescriptor_t mdy = NULL, mw = NULL, mdx = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    MiopenConvSolution *solutions = NULL;
    size_t count = 0, returned = 0, query_count, i;
    int out_count = 0;
    cudnnStatus_t cs;
    miopenStatus_t ms;

    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !returnedAlgoCount)
        return CUDNN_STATUS_BAD_PARAM;
    *returnedAlgoCount = 0;
    if (requestedAlgoCount < 0 || (requestedAlgoCount > 0 && !perfResults))
        return CUDNN_STATUS_BAD_PARAM;
    if (requestedAlgoCount == 0) return CUDNN_STATUS_SUCCESS;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;

    cs = bridge_validate_conv_shapes(dxDesc, wDesc, convDesc, dyDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    cs = bridge_make_miopen_tensor(dyDesc, &mdy); if (cs) goto done;
    cs = bridge_make_miopen_filter(wDesc, &mw); if (cs) goto done;
    cs = bridge_make_miopen_tensor(dxDesc, &mdx); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;

    ms = g_miopen.bwd_data_solution_count(handle->miopen, mdy, mw, mc, mdx, &count);
    if (ms != MIOPEN_STATUS_SUCCESS) { cs = bridge_status(ms); goto done; }
    if (count == 0) { cs = CUDNN_STATUS_NOT_SUPPORTED; goto done; }

    query_count = count > 64 ? 64 : count;
    solutions = (MiopenConvSolution *)calloc(query_count, sizeof(*solutions));
    if (!solutions) { cs = CUDNN_STATUS_ALLOC_FAILED; goto done; }
    ms = g_miopen.bwd_data_get_solutions(
        handle->miopen, mdy, mw, mc, mdx, query_count, &returned, solutions);
    if (ms != MIOPEN_STATUS_SUCCESS) { cs = bridge_status(ms); goto done; }

    for (i = 0; i < returned && i < query_count && out_count < requestedAlgoCount; ++i) {
        int mapped = -1, j, duplicate = 0;
        size_t ws = solutions[i].workspace_size;
        if (!bridge_unmap_bwd_data_algo(solutions[i].algorithm, &mapped)) continue;
        for (j = 0; j < out_count; ++j) {
            if (perfResults[j].algo == mapped) { duplicate = 1; break; }
        }
        if (duplicate) continue;
        ms = g_miopen.bwd_data_solution_workspace(
            handle->miopen, mdy, mw, mc, mdx, solutions[i].solution_id, &ws);
        if (ms != MIOPEN_STATUS_SUCCESS) continue;
        memset(&perfResults[out_count], 0, sizeof(perfResults[out_count]));
        perfResults[out_count].algo = mapped;
        perfResults[out_count].status = CUDNN_STATUS_SUCCESS;
        perfResults[out_count].time = solutions[i].time;
        perfResults[out_count].memory = ws;
        perfResults[out_count].determinism = 0;
        perfResults[out_count].mathType = convDesc->math_type;
        ++out_count;
    }

    *returnedAlgoCount = out_count;
    cs = out_count > 0 ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_SUPPORTED;
done:
    free(solutions);
    bridge_destroy_miopen_tensor(mdy);
    bridge_destroy_miopen_tensor(mw);
    bridge_destroy_miopen_tensor(mdx);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
    miopenTensorDescriptor_t mdy = NULL, mw = NULL, mdx = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    uint64_t solution_id = 0;
    cudnnStatus_t cs;
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !sizeInBytes)
        return CUDNN_STATUS_BAD_PARAM;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;
    cs = bridge_validate_conv_shapes(dxDesc, wDesc, convDesc, dyDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    cs = bridge_make_miopen_tensor(dyDesc, &mdy); if (cs) goto done;
    cs = bridge_make_miopen_filter(wDesc, &mw); if (cs) goto done;
    cs = bridge_make_miopen_tensor(dxDesc, &mdx); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;
    cs = bridge_pick_bwd_data_solution(
        handle->miopen, mdy, mw, mc, mdx, algo, &solution_id, sizeInBytes, NULL);
done:
    bridge_destroy_miopen_tensor(mdy);
    bridge_destroy_miopen_tensor(mw);
    bridge_destroy_miopen_tensor(mdx);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    miopenTensorDescriptor_t mdy = NULL, mw = NULL, mdx = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    uint64_t solution_id = 0;
    size_t required_workspace = 0;
    cudnnStatus_t cs;
    miopenStatus_t ms;

    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !alpha || !beta || !w || !dy || !dx)
        return CUDNN_STATUS_BAD_PARAM;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;
    cs = bridge_validate_conv_shapes(dxDesc, wDesc, convDesc, dyDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    if (!bridge_default_scalars(convDesc, alpha, beta)) return CUDNN_STATUS_NOT_SUPPORTED;

    cs = bridge_make_miopen_tensor(dyDesc, &mdy); if (cs) goto done;
    cs = bridge_make_miopen_filter(wDesc, &mw); if (cs) goto done;
    cs = bridge_make_miopen_tensor(dxDesc, &mdx); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;
    cs = bridge_pick_bwd_data_solution(
        handle->miopen, mdy, mw, mc, mdx, algo,
        &solution_id, &required_workspace, NULL);
    if (cs != CUDNN_STATUS_SUCCESS) goto done;
    if (required_workspace > workSpaceSizeInBytes ||
        (required_workspace > 0 && !workSpace)) {
        cs = CUDNN_STATUS_BAD_PARAM;
        goto done;
    }

    ms = g_miopen.conv_bwd_data_immediate(
        handle->miopen, mdy, dy, mw, w, mc, mdx, dx,
        workSpace, workSpaceSizeInBytes, solution_id);
    cs = bridge_status(ms);
done:
    bridge_destroy_miopen_tensor(mdy);
    bridge_destroy_miopen_tensor(mw);
    bridge_destroy_miopen_tensor(mdx);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    cudnnHandle_t handle, int *count) {
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !count)
        return CUDNN_STATUS_BAD_PARAM;
    *count = 3;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, int requestedAlgoCount, int *returnedAlgoCount,
    CudnnConvolutionBwdFilterAlgoPerf *perfResults) {
    miopenTensorDescriptor_t mx = NULL, mdy = NULL, mdw = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    MiopenConvSolution *solutions = NULL;
    size_t count = 0, returned = 0, query_count, i;
    int out_count = 0;
    cudnnStatus_t cs;
    miopenStatus_t ms;

    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !returnedAlgoCount)
        return CUDNN_STATUS_BAD_PARAM;
    *returnedAlgoCount = 0;
    if (requestedAlgoCount < 0 || (requestedAlgoCount > 0 && !perfResults))
        return CUDNN_STATUS_BAD_PARAM;
    if (requestedAlgoCount == 0) return CUDNN_STATUS_SUCCESS;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;

    cs = bridge_validate_conv_shapes(xDesc, dwDesc, convDesc, dyDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    cs = bridge_make_miopen_tensor(xDesc, &mx); if (cs) goto done;
    cs = bridge_make_miopen_tensor(dyDesc, &mdy); if (cs) goto done;
    cs = bridge_make_miopen_filter(dwDesc, &mdw); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;

    ms = g_miopen.bwd_weights_solution_count(handle->miopen, mdy, mx, mc, mdw, &count);
    if (ms != MIOPEN_STATUS_SUCCESS) { cs = bridge_status(ms); goto done; }
    if (count == 0) { cs = CUDNN_STATUS_NOT_SUPPORTED; goto done; }

    query_count = count > 64 ? 64 : count;
    solutions = (MiopenConvSolution *)calloc(query_count, sizeof(*solutions));
    if (!solutions) { cs = CUDNN_STATUS_ALLOC_FAILED; goto done; }
    ms = g_miopen.bwd_weights_get_solutions(
        handle->miopen, mdy, mx, mc, mdw, query_count, &returned, solutions);
    if (ms != MIOPEN_STATUS_SUCCESS) { cs = bridge_status(ms); goto done; }

    for (i = 0; i < returned && i < query_count && out_count < requestedAlgoCount; ++i) {
        int mapped = -1, j, duplicate = 0;
        size_t ws = solutions[i].workspace_size;
        if (!bridge_unmap_bwd_filter_algo(solutions[i].algorithm, &mapped)) continue;
        for (j = 0; j < out_count; ++j) {
            if (perfResults[j].algo == mapped) { duplicate = 1; break; }
        }
        if (duplicate) continue;
        ms = g_miopen.bwd_weights_solution_workspace(
            handle->miopen, mdy, mx, mc, mdw, solutions[i].solution_id, &ws);
        if (ms != MIOPEN_STATUS_SUCCESS) continue;
        memset(&perfResults[out_count], 0, sizeof(perfResults[out_count]));
        perfResults[out_count].algo = mapped;
        perfResults[out_count].status = CUDNN_STATUS_SUCCESS;
        perfResults[out_count].time = solutions[i].time;
        perfResults[out_count].memory = ws;
        perfResults[out_count].determinism = 0;
        perfResults[out_count].mathType = convDesc->math_type;
        ++out_count;
    }

    *returnedAlgoCount = out_count;
    cs = out_count > 0 ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_SUPPORTED;
done:
    free(solutions);
    bridge_destroy_miopen_tensor(mx);
    bridge_destroy_miopen_tensor(mdy);
    bridge_destroy_miopen_tensor(mdw);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    size_t *sizeInBytes) {
    miopenTensorDescriptor_t mx = NULL, mdy = NULL, mdw = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    uint64_t solution_id = 0;
    cudnnStatus_t cs;
    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !sizeInBytes)
        return CUDNN_STATUS_BAD_PARAM;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;
    cs = bridge_validate_conv_shapes(xDesc, dwDesc, convDesc, dyDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    cs = bridge_make_miopen_tensor(xDesc, &mx); if (cs) goto done;
    cs = bridge_make_miopen_tensor(dyDesc, &mdy); if (cs) goto done;
    cs = bridge_make_miopen_filter(dwDesc, &mdw); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;
    cs = bridge_pick_bwd_filter_solution(
        handle->miopen, mdy, mx, mc, mdw, algo, &solution_id, sizeInBytes, NULL);
done:
    bridge_destroy_miopen_tensor(mx);
    bridge_destroy_miopen_tensor(mdy);
    bridge_destroy_miopen_tensor(mdw);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {
    miopenTensorDescriptor_t mx = NULL, mdy = NULL, mdw = NULL;
    miopenConvolutionDescriptor_t mc = NULL;
    uint64_t solution_id = 0;
    size_t required_workspace = 0;
    cudnnStatus_t cs;
    miopenStatus_t ms;

    if (!handle || handle->magic != BRIDGE_MAGIC_HANDLE || !alpha || !beta || !x || !dy || !dw)
        return CUDNN_STATUS_BAD_PARAM;
    if (handle->cuda_stream != NULL) return CUDNN_STATUS_NOT_SUPPORTED;
    cs = bridge_validate_conv_shapes(xDesc, dwDesc, convDesc, dyDesc);
    if (cs != CUDNN_STATUS_SUCCESS) return cs;
    if (!bridge_default_scalars(convDesc, alpha, beta)) return CUDNN_STATUS_NOT_SUPPORTED;

    cs = bridge_make_miopen_tensor(xDesc, &mx); if (cs) goto done;
    cs = bridge_make_miopen_tensor(dyDesc, &mdy); if (cs) goto done;
    cs = bridge_make_miopen_filter(dwDesc, &mdw); if (cs) goto done;
    cs = bridge_make_miopen_conv(convDesc, &mc); if (cs) goto done;
    cs = bridge_pick_bwd_filter_solution(
        handle->miopen, mdy, mx, mc, mdw, algo,
        &solution_id, &required_workspace, NULL);
    if (cs != CUDNN_STATUS_SUCCESS) goto done;
    if (required_workspace > workSpaceSizeInBytes ||
        (required_workspace > 0 && !workSpace)) {
        cs = CUDNN_STATUS_BAD_PARAM;
        goto done;
    }

    ms = g_miopen.conv_bwd_weights_immediate(
        handle->miopen, mdy, dy, mx, x, mc, mdw, dw,
        workSpace, workSpaceSizeInBytes, solution_id);
    cs = bridge_status(ms);
done:
    bridge_destroy_miopen_tensor(mx);
    bridge_destroy_miopen_tensor(mdy);
    bridge_destroy_miopen_tensor(mdw);
    bridge_destroy_miopen_conv(mc);
    return cs;
}

/* ---- Diagnostics ------------------------------------------------------- */

const char *cudnnGetErrorString(cudnnStatus_t status) {
    switch (status) {
        case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR";
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING: return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS: return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW: return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
        case CUDNN_STATUS_VERSION_MISMATCH: return "CUDNN_STATUS_VERSION_MISMATCH";
        default: return "CUDNN_STATUS_UNKNOWN";
    }
}