//! Point a non-Windows linker at the CUDA driver library.
//!
//! On Windows nothing is needed: the extern block uses `raw-dylib`, so the import thunks are
//! synthesised from `nvcuda.dll` and no toolkit import library is involved. Elsewhere the
//! loader usually finds `libcuda` on its own, and the toolkit path is added when it exists.

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    if cfg!(windows) {
        return;
    }
    if let Ok(root) = std::env::var("CUDA_PATH") {
        for sub in ["lib64", "lib", "lib64/stubs"] {
            let dir = std::path::Path::new(&root).join(sub);
            if dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
    }
}
