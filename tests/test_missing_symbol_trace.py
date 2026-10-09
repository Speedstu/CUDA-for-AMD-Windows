"""Safe ZLUDA symbol-lookup diagnostic: loads PTX but launches no GPU kernels.

Use only in an isolated candidate runtime. Expect a failed lookup (not a crash),
and with the optional trace patch enabled, a [zluda-symbol] line on stderr.
"""
import ctypes
import json
import os
import sys

if sys.platform != 'win32':
    raise SystemExit('Windows-only probe')

cuda = ctypes.WinDLL('nvcuda.dll')

def fn(name, argtypes):
    value = getattr(cuda, name)
    value.argtypes = argtypes
    value.restype = ctypes.c_int
    return value

def ok(result, operation):
    if result != 0:
        raise RuntimeError(f'{operation}: CUDA error {result}')

cu_init = fn('cuInit', [ctypes.c_uint])
cu_device_get = fn('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
cu_device_name = fn('cuDeviceGetName', [ctypes.c_void_p, ctypes.c_int, ctypes.c_int])
cu_ctx_get = fn('cuCtxGetCurrent', [ctypes.POINTER(ctypes.c_void_p)])
cu_ctx_create = fn('cuCtxCreate_v2', [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int])
cu_ctx_destroy = fn('cuCtxDestroy_v2', [ctypes.c_void_p])
cu_module_load = fn('cuModuleLoadData', [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
cu_module_function = fn('cuModuleGetFunction', [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p])
cu_module_unload = fn('cuModuleUnload', [ctypes.c_void_p])

ok(cu_init(0), 'cuInit')
device = ctypes.c_int()
ok(cu_device_get(ctypes.byref(device), 0), 'cuDeviceGet')
name = ctypes.create_string_buffer(256)
ok(cu_device_name(name, len(name), device.value), 'cuDeviceGetName')
ctx = ctypes.c_void_p()
ok(cu_ctx_get(ctypes.byref(ctx)), 'cuCtxGetCurrent')
created = not ctx.value
if created:
    ok(cu_ctx_create(ctypes.byref(ctx), 0, device.value), 'cuCtxCreate_v2')
ptx = ctypes.create_string_buffer(b'.version 7.0\n.target sm_80\n.address_size 64\n.visible .entry noop() { ret; }\n')
mod = ctypes.c_void_p()
try:
    ok(cu_module_load(ctypes.byref(mod), ctypes.cast(ptx, ctypes.c_void_p)), 'cuModuleLoadData')
    good_fn = ctypes.c_void_p()
    ok(cu_module_function(ctypes.byref(good_fn), mod, b'noop'), 'cuModuleGetFunction(noop)')
    bad_fn = ctypes.c_void_p()
    err = int(cu_module_function(ctypes.byref(bad_fn), mod, b'cudaamd_intentionally_missing_kernel'))
    report = {'device': name.value.decode('utf-8', errors='replace'),
              'known_kernel_lookup': 0,
              'missing_kernel_lookup': err,
              'trace_requested': os.environ.get('ZLUDA_TRACE_MISSING_SYMBOLS') == '1',
              'kernel_executed': False}
    print('CUDAAMD_SYMBOL_LOOKUP:' + json.dumps(report, sort_keys=True), flush=True)
    if err == 0:
        raise RuntimeError('Missing kernel unexpectedly resolved')
finally:
    if mod.value:
        cu_module_unload(mod)
    if created and ctx.value:
        cu_ctx_destroy(ctx)