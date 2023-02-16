pub struct PythonVersion {
    pub major: u32,
    pub minor: u32,
    pub micro: u32,
    pub url: &'static str,
    pub sha256: &'static str,
}

/// Lists the python releases we want to build against
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
pub const PYTHON_VERSIONS: [PythonVersion; 4] = [
        PythonVersion {
            major: 3,
            minor: 10,
            micro: 9,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "d196347aeb701a53fe2bb2b095abec38d27d0fa0443f8a1c2023a1bed6e18cdf",
        },
        PythonVersion {
            major: 3,
            minor: 11,
            micro: 1,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.11.1+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "02a551fefab3750effd0e156c25446547c238688a32fabde2995c941c03a6423",
        },
        PythonVersion {
            major: 3,
            minor: 8,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.8.16+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "c890de112f1ae31283a31fefd2061d5c97bdd4d1bdd795552c7abddef2697ea1",
        },
        PythonVersion {
            major: 3,
            minor: 9,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.9.16+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "7ba397787932393e65fc2fb9fcfabf54f2bb6751d5da2b45913cb25b2d493758",
        },
    ];

/// Lists the python releases we want to build against
#[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
pub const PYTHON_VERSIONS: [PythonVersion; 4] = [
        PythonVersion {
            major: 3,
            minor: 10,
            micro: 9,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-aarch64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "2003750f40cd09d4bf7a850342613992f8d9454f03b3c067989911fb37e7a4d1",
        },
        PythonVersion {
            major: 3,
            minor: 11,
            micro: 1,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.11.1+20230116-aarch64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "debf15783bdcb5530504f533d33fda75a7b905cec5361ae8f33da5ba6599f8b4",
        },
        PythonVersion {
            major: 3,
            minor: 8,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.8.16+20230116-aarch64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "15d00bc8400ed6d94c665a797dc8ed7a491ae25c5022e738dcd665cd29beec42",
        },
        PythonVersion {
            major: 3,
            minor: 9,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.9.16+20230116-aarch64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "1ba520c0db431c84305677f56eb9a4254f5097430ed443e92fc8617f8fba973d",
        },
    ];

/// Lists the python releases we want to build against
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub const PYTHON_VERSIONS: [PythonVersion; 4] = [
        PythonVersion {
            major: 3,
            minor: 10,
            micro: 9,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-aarch64-apple-darwin-install_only.tar.gz",
            sha256: "018d05a779b2de7a476f3b3ff2d10f503d69d14efcedd0774e6dab8c22ef84ff",
        },
        PythonVersion {
            major: 3,
            minor: 11,
            micro: 1,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.11.1+20230116-aarch64-apple-darwin-install_only.tar.gz",
            sha256: "4918cdf1cab742a90f85318f88b8122aeaa2d04705803c7b6e78e81a3dd40f80",
        },
        PythonVersion {
            major: 3,
            minor: 8,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.8.16+20230116-aarch64-apple-darwin-install_only.tar.gz",
            sha256: "d1f408569d8807c1053939d7822b082a17545e363697e1ce3cfb1ee75834c7be",
        },
        PythonVersion {
            major: 3,
            minor: 9,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.9.16+20230116-aarch64-apple-darwin-install_only.tar.gz",
            sha256: "d732d212d42315ac27c6da3e0b69636737a8d72086c980daf844344c010cab80",
        },
    ];

/// Lists the python releases we want to build against
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub const PYTHON_VERSIONS: [PythonVersion; 4] = [
        PythonVersion {
            major: 3,
            minor: 10,
            micro: 9,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-x86_64-apple-darwin-install_only.tar.gz",
            sha256: "0e685f98dce0e5bc8da93c7081f4e6c10219792e223e4b5886730fd73a7ba4c6",
        },
        PythonVersion {
            major: 3,
            minor: 11,
            micro: 1,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.11.1+20230116-x86_64-apple-darwin-install_only.tar.gz",
            sha256: "20a4203d069dc9b710f70b09e7da2ce6f473d6b1110f9535fb6f4c469ed54733",
        },
        PythonVersion {
            major: 3,
            minor: 8,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.8.16+20230116-x86_64-apple-darwin-install_only.tar.gz",
            sha256: "484ba901f64fc7888bec5994eb49343dc3f9d00ed43df17ee9c40935aad4aa18",
        },
        PythonVersion {
            major: 3,
            minor: 9,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.9.16+20230116-x86_64-apple-darwin-install_only.tar.gz",
            sha256: "3948384af5e8d4ee7e5ccc648322b99c1c5cf4979954ed5e6b3382c69d6db71e",
        },
    ];
