#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define HIPCHK(x) do { auto _e=(x); if (_e!=hipSuccess) { std::fprintf(stderr,"HIP error %d at %s:%d\n",(int)_e,__FILE__,__LINE__); return 2; } } while(0)
#define RBCHK(x) do { auto _e=(x); if (_e!=rocblas_status_success) { std::fprintf(stderr,"rocBLAS error %d at %s:%d\n",(int)_e,__FILE__,__LINE__); return 3; } } while(0)

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 2048;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 10;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 30;
    if (n <= 0 || warmup < 0 || iterations <= 0) return 4;

    hipDeviceProp_t prop{}; HIPCHK(hipGetDeviceProperties(&prop, 0));
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    float *a=nullptr, *b=nullptr, *c=nullptr;
    HIPCHK(hipMalloc(reinterpret_cast<void**>(&a), bytes));
    HIPCHK(hipMalloc(reinterpret_cast<void**>(&b), bytes));
    HIPCHK(hipMalloc(reinterpret_cast<void**>(&c), bytes));
    HIPCHK(hipMemset(a, 0, bytes)); HIPCHK(hipMemset(b, 0, bytes)); HIPCHK(hipMemset(c, 0, bytes));

    rocblas_handle handle{}; RBCHK(rocblas_create_handle(&handle));
    const float alpha=1.0f, beta=0.0f;
    for (int i=0; i<warmup; ++i)
        RBCHK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, n,n,n, &alpha, a,n,b,n, &beta,c,n));
    HIPCHK(hipDeviceSynchronize());

    hipEvent_t start{}, stop{}; HIPCHK(hipEventCreate(&start)); HIPCHK(hipEventCreate(&stop));
    HIPCHK(hipEventRecord(start, nullptr));
    const auto wall_start = std::chrono::steady_clock::now();
    for (int i=0; i<iterations; ++i)
        RBCHK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, n,n,n, &alpha, a,n,b,n, &beta,c,n));
    HIPCHK(hipEventRecord(stop, nullptr)); HIPCHK(hipDeviceSynchronize());
    const auto wall_stop = std::chrono::steady_clock::now();
    float event_total_ms=0.0f; HIPCHK(hipEventElapsedTime(&event_total_ms, start, stop));
    const double wall_total_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    const double wall_per_ms = wall_total_ms / iterations;
    const double event_per_ms = event_total_ms / iterations;
    const double wall_tflops = (2.0 * n * n * n) / (wall_per_ms * 1.0e9);
    std::printf("{\"backend\":\"rocblas-direct\",\"device\":\"%s\",\"n\":%d,\"iterations\":%d,\"wall_ms_per_gemm\":%.9f,\"event_ms_per_gemm\":%.9f,\"wall_total_ms\":%.9f,\"event_total_ms\":%.9f,\"wall_tflops\":%.9f}\n", prop.name, n, iterations, wall_per_ms, event_per_ms, wall_total_ms, (double)event_total_ms, wall_tflops);

    HIPCHK(hipEventDestroy(start)); HIPCHK(hipEventDestroy(stop)); RBCHK(rocblas_destroy_handle(handle));
    HIPCHK(hipFree(a)); HIPCHK(hipFree(b)); HIPCHK(hipFree(c)); return 0;
}
