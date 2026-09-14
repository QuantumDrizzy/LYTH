//! The CUDA Driver API, just enough of it to load PTX and launch it.
//!
//! ADR-0001 chose the driver API over an object that `nvcc` links: a PTX blob loaded with
//! `cuModuleLoadData` drops into an existing tree today and needs no toolchain coupling.
//!
//! All `unsafe` in LYTH lives in this file. Every entry point below is safe to call, and the
//! invariant each one relies on is written above it. Device memory is owned by [`Buffer`] and
//! freed on drop, so a failed launch cannot leak it.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::marker::PhantomData;

#[allow(non_camel_case_types)]
type CUresult = c_int;
#[allow(non_camel_case_types)]
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

// LINKING, AND WHY IT IS NOT `cuda.lib`.
//
// CUDA 13's `cuda.lib` on Windows carries no `__imp_` entries, so linking against it as an
// import library leaves every driver symbol unresolved. `raw-dylib` sidesteps the question:
// the compiler synthesises the import thunks from the DLL name alone, so **no CUDA toolkit is
// needed to build this crate** — only an NVIDIA driver at run time, which any machine that
// could run the output has anyway.
//
// `nvcuda.dll` ships with the driver on Windows; elsewhere the loader finds `libcuda` by name.
#[cfg_attr(windows, link(name = "nvcuda", kind = "raw-dylib"))]
#[cfg_attr(not(windows), link(name = "cuda"))]
extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: c_int) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut *mut c_void, flags: c_uint, dev: c_int) -> CUresult;
    fn cuCtxDestroy_v2(ctx: *mut c_void) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuModuleLoadDataEx(
        module: *mut *mut c_void,
        image: *const c_void,
        num_options: c_uint,
        options: *mut c_int,
        option_values: *mut *mut c_void,
    ) -> CUresult;
    fn cuModuleUnload(module: *mut c_void) -> CUresult;
    fn cuModuleGetFunction(
        func: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: *mut c_void,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_bytes: c_uint,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

/// JIT compile options. Values from `cuda.h`.
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("{op} failed: {msg} (CUresult {code})")]
    Driver {
        op: &'static str,
        code: CUresult,
        msg: String,
    },
    #[error("{0}")]
    Message(String),
}

/// What a `CUresult` means, in words a caller can act on.
///
/// The driver exports `cuGetErrorString`, but CUDA 13's `cuda.lib` on Windows provides no
/// import entry for it — it is the one symbol of the fourteen used here that does not link.
/// Rather than make the build depend on which toolkit is installed, the codes are named here.
///
/// This is also the better answer: the failures a freshly written code generator actually hits
/// are `INVALID_PTX`, `UNSUPPORTED_PTX_VERSION` and `NO_BINARY_FOR_GPU`, and each one gets a
/// sentence saying what to do, which the driver's own terse string does not.
fn error_text(code: CUresult) -> String {
    let named = match code {
        1 => "INVALID_VALUE — an argument the driver rejected",
        2 => "OUT_OF_MEMORY",
        3 => "NOT_INITIALIZED — cuInit was not called, or it failed",
        4 => "DEINITIALIZED — the driver is shutting down",
        100 => "NO_DEVICE — no CUDA-capable device is visible",
        101 => "INVALID_DEVICE",
        200 => "INVALID_IMAGE — the module is not valid PTX or cubin",
        201 => "INVALID_CONTEXT",
        209 => {
            "NO_BINARY_FOR_GPU — the PTX is valid but targets no architecture this device can run. Check the `.target` line against the machine file."
        }
        218 => {
            "INVALID_PTX — the generated PTX did not assemble. The JIT log printed above names the line."
        }
        222 => {
            "UNSUPPORTED_PTX_VERSION — the `.version` directive is newer than this driver. Lower it, or update the driver."
        }
        400 => "INVALID_HANDLE",
        500 => "NOT_FOUND — no such entry point in the module",
        700 => "ILLEGAL_ADDRESS — a thread read or wrote outside its allocation",
        701 => "LAUNCH_OUT_OF_RESOURCES — too many registers or threads per block",
        702 => "LAUNCH_TIMEOUT",
        719 => "LAUNCH_FAILED",
        _ => "",
    };
    if named.is_empty() {
        format!("CUresult {code} (no name known for this code)")
    } else {
        named.to_string()
    }
}

fn check(op: &'static str, code: CUresult) -> Result<(), CudaError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Driver {
            op,
            code,
            msg: error_text(code),
        })
    }
}

/// An initialised device and a context bound to it.
///
/// Holding one is what makes every other call in this module legal, which is why allocations
/// and modules borrow from it and cannot outlive it.
pub struct Context {
    ctx: *mut c_void,
    pub device_name: String,
}

impl Context {
    /// Initialise the driver and create a context on `ordinal`.
    pub fn new(ordinal: i32) -> Result<Self, CudaError> {
        // SAFETY: cuInit takes no pointers. It must precede every other driver call, which is
        // why it is here and not anywhere a caller could skip.
        unsafe { check("cuInit", cuInit(0))? };

        let mut dev: c_int = 0;
        // SAFETY: `dev` is a live, aligned i32 for the duration of the call.
        unsafe { check("cuDeviceGet", cuDeviceGet(&mut dev, ordinal))? };

        let mut name = vec![0i8 as c_char; 128];
        // SAFETY: the driver writes at most `len` bytes into a buffer we own of that size.
        unsafe {
            check(
                "cuDeviceGetName",
                cuDeviceGetName(name.as_mut_ptr(), name.len() as c_int, dev),
            )?
        };
        // SAFETY: the driver NUL-terminates within the buffer it was given.
        let device_name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut ctx: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ctx` is a live out-parameter; the context it returns is freed in Drop.
        unsafe { check("cuCtxCreate", cuCtxCreate_v2(&mut ctx, 0, dev))? };
        Ok(Context { ctx, device_name })
    }

    /// Load a PTX module, capturing the JIT assembler's log.
    ///
    /// The log is the whole point: when a generated module does not assemble, `INVALID_PTX`
    /// on its own is useless, and the log names the line and the token. A code generator
    /// without it is debugged by bisection.
    pub fn load_ptx(&self, ptx: &str) -> Result<Module<'_>, CudaError> {
        let src = CString::new(ptx)
            .map_err(|e| CudaError::Message(format!("PTX contains an interior NUL: {e}")))?;
        let mut module: *mut c_void = std::ptr::null_mut();

        let mut log = vec![0u8; 8192];
        let mut options = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        // The size option is passed by value cast to a pointer, which is how the driver's
        // option array works for integer options.
        let mut values: [*mut c_void; 2] =
            [log.as_mut_ptr() as *mut c_void, log.len() as *mut c_void];

        // SAFETY: `src`, `options` and `values` all outlive the call, and `log` is exactly
        // the length handed to the driver in the size option.
        let code = unsafe {
            cuModuleLoadDataEx(
                &mut module,
                src.as_ptr() as *const c_void,
                options.len() as c_uint,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if code != CUDA_SUCCESS {
            let text = log
                .split(|b| *b == 0)
                .next()
                .map(|b| String::from_utf8_lossy(b).trim().to_string())
                .unwrap_or_default();
            let mut msg = error_text(code);
            if !text.is_empty() {
                msg.push_str("\n  ptx jit log:\n    ");
                msg.push_str(&text.replace('\n', "\n    "));
            }
            return Err(CudaError::Driver {
                op: "cuModuleLoadDataEx",
                code,
                msg,
            });
        }
        Ok(Module {
            module,
            _ctx: PhantomData,
        })
    }

    /// Allocate `len` f32 on the device and fill it from `host`.
    pub fn upload(&self, host: &[f32]) -> Result<Buffer<'_>, CudaError> {
        let buf = self.alloc(host.len())?;
        let bytes = std::mem::size_of_val(host);
        // SAFETY: `buf` is at least `bytes` long by construction, and `host` is a live slice
        // of exactly that many bytes.
        unsafe {
            check(
                "cuMemcpyHtoD",
                cuMemcpyHtoD_v2(buf.ptr, host.as_ptr() as *const c_void, bytes),
            )?
        };
        Ok(buf)
    }

    /// Allocate `len` f32 of uninitialised device memory.
    pub fn alloc(&self, len: usize) -> Result<Buffer<'_>, CudaError> {
        let bytes = len * std::mem::size_of::<f32>();
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: `ptr` is a live out-parameter; the allocation is freed in Buffer::drop.
        unsafe { check("cuMemAlloc", cuMemAlloc_v2(&mut ptr, bytes.max(1)))? };
        Ok(Buffer {
            ptr,
            len,
            _ctx: PhantomData,
        })
    }

    /// Block until every launch on this context has finished, and report a launch failure
    /// that was only detectable asynchronously.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: no pointers; requires a current context, which holding `&self` implies.
        unsafe { check("cuCtxSynchronize", cuCtxSynchronize()) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: `ctx` came from cuCtxCreate and is destroyed exactly once.
            unsafe {
                let _ = cuCtxDestroy_v2(self.ctx);
            }
        }
    }
}

/// Device memory holding `len` f32, freed when this value is dropped.
#[derive(Debug)]
pub struct Buffer<'ctx> {
    ptr: CUdeviceptr,
    len: usize,
    _ctx: PhantomData<&'ctx Context>,
}

impl Buffer<'_> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy the whole buffer back into a fresh Vec.
    pub fn download(&self) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0.0f32; self.len];
        let bytes = std::mem::size_of_val(out.as_slice());
        // SAFETY: `out` is exactly `bytes` long and the device allocation is at least that.
        unsafe {
            check(
                "cuMemcpyDtoH",
                cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
            )?
        };
        Ok(out)
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: `ptr` came from cuMemAlloc and is freed exactly once.
            unsafe {
                let _ = cuMemFree_v2(self.ptr);
            }
        }
    }
}

pub struct Module<'ctx> {
    module: *mut c_void,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Module<'ctx> {
    pub fn function(&self, name: &str) -> Result<Function<'ctx>, CudaError> {
        let cname = CString::new(name)
            .map_err(|e| CudaError::Message(format!("kernel name has an interior NUL: {e}")))?;
        let mut f: *mut c_void = std::ptr::null_mut();
        // SAFETY: `cname` outlives the call; `self.module` is a live module.
        unsafe {
            check(
                "cuModuleGetFunction",
                cuModuleGetFunction(&mut f, self.module, cname.as_ptr()),
            )?
        };
        Ok(Function {
            f,
            _ctx: PhantomData,
        })
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: `module` came from cuModuleLoadData and is unloaded exactly once.
            unsafe {
                let _ = cuModuleUnload(self.module);
            }
        }
    }
}

pub struct Function<'ctx> {
    f: *mut c_void,
    _ctx: PhantomData<&'ctx Context>,
}

/// One launch argument.
///
/// `cuLaunchKernel` takes an array of pointers *to* the argument values, so the values must
/// outlive the call. Making them an owned enum rather than raw pointers is what stops a
/// caller passing a pointer to a temporary.
#[derive(Debug, Clone, Copy)]
pub enum Arg<'a> {
    U32(u32),
    F32(f32),
    Buf(&'a Buffer<'a>),
}

impl Function<'_> {
    /// Launch with a 1-D grid.
    pub fn launch(&self, grid: u32, block: u32, args: &[Arg]) -> Result<(), CudaError> {
        if block == 0 || grid == 0 {
            return Err(CudaError::Message(
                "a launch with an empty grid or block does no work".into(),
            ));
        }
        // Materialise the argument values into a buffer that outlives the call, then an array
        // of pointers into it. Both stay alive until after cuLaunchKernel returns.
        let mut values: Vec<[u8; 8]> = Vec::with_capacity(args.len());
        for a in args {
            let mut slot = [0u8; 8];
            match a {
                Arg::U32(v) => slot[..4].copy_from_slice(&v.to_ne_bytes()),
                Arg::F32(v) => slot[..4].copy_from_slice(&v.to_ne_bytes()),
                Arg::Buf(b) => slot.copy_from_slice(&b.ptr.to_ne_bytes()),
            }
            values.push(slot);
        }
        let mut ptrs: Vec<*mut c_void> = values
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();

        // SAFETY: `ptrs` points into `values`, both live until the end of this function, and
        // the driver copies the argument bytes during the call. `extra` is null because the
        // parameters are passed through `kernel_params`.
        unsafe {
            check(
                "cuLaunchKernel",
                cuLaunchKernel(
                    self.f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )?
        };
        Ok(())
    }
}

/// Blocks needed to cover `n` elements at `block` threads each.
pub fn grid_for(n: u32, block: u32) -> u32 {
    n.div_ceil(block).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_covers_every_element_including_a_ragged_tail() {
        assert_eq!(grid_for(1024, 256), 4);
        assert_eq!(grid_for(1025, 256), 5, "the tail needs its own block");
        assert_eq!(grid_for(1, 256), 1);
        assert_eq!(grid_for(0, 256), 1, "never launch an empty grid");
    }

    #[test]
    fn an_unknown_error_code_still_produces_a_message() {
        // Must not panic or return an empty string: a driver failure is reported to a human.
        assert!(!error_text(-12345).is_empty());
    }
}
