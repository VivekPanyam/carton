pub struct PythonVersion {
    pub major: u32,
    pub minor: u32,
    pub micro: u32,
    pub url: &'static str,
    pub sha256: &'static str,
}

/// Lists the python releases we want to build against
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
