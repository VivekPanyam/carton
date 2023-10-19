// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// libtorch

// x86 linux
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
pub mod libtorch {
    pub const URL: &str = "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip";
    pub const SHA256: &str = "843ad19e769a189758fd6a21bfced9024494b52344f4bc4fb75f75d36e6ea0c7";
}

// aarch64 linux
#[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
pub mod libtorch {
    pub const URL: &str = "https://github.com/VivekPanyam/libtorch-prebuilts/releases/download/2.0.1/libtorch-aarch64-cxx11-abi-shared-with-deps-2.0.1-cpu.zip";
    pub const SHA256: &str = "bff875b93ec7f5712bba4dd98a3a8b99ca814fdd575b1eb4afa33b0f1bd51fa2";
}

// x86 mac
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub mod libtorch {
    pub const URL: &str = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.1.zip";
    pub const SHA256: &str = "002876b74d8432ee8ab9d0e710159353149d92ada73ef81844e81bccbfa52e95";
}

// aarch64 mac
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod libtorch {
    pub const URL: &str = "https://github.com/VivekPanyam/libtorch-prebuilts/releases/download/2.0.1/libtorch-amd64-macos-2.0.1.zip";
    pub const SHA256: &str = "ad304b78d24f7c52f7926b836d9aadee89d145039ff47c9b313d2d816028c05d";
}

pub const LIBTORCH_VERSION: semver::Version = semver::Version::new(2, 0, 1);

// XLA

// x86 linux
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
pub mod xla {
    pub const URL: &str = "https://github.com/elixir-nx/xla/releases/download/v0.5.1/xla_extension-x86_64-linux-gnu-cuda120.tar.gz";
    pub const SHA256: &str = "7fb09643285ab85facba52a021c421188549fa0193a58feb52a6b9a129e7920c";
}

// aarch64 linux
#[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
pub mod xla {
    pub const URL: &str = "https://github.com/elixir-nx/xla/releases/download/v0.5.1/xla_extension-aarch64-linux-gnu-cpu.tar.gz";
    pub const SHA256: &str = "6580deebdd44b345bbdb48eb1450eaad96ccdc85c41dd6c7034ee838cb1d78b6";
}

// x86 mac
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub mod xla {
    pub const URL: &str = "https://github.com/elixir-nx/xla/releases/download/v0.5.1/xla_extension-x86_64-darwin-cpu.tar.gz";
    pub const SHA256: &str = "734ee140c9521505bc3182954095e461042de0af63236d8d63681ed4339d6ac9";
}

// aarch64 mac
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod xla {
    pub const URL: &str = "https://github.com/elixir-nx/xla/releases/download/v0.5.1/xla_extension-aarch64-darwin-cpu.tar.gz";
    pub const SHA256: &str = "975cde36e94835139ca5d6e5232a0e4d9131b3e17df05a7fbbce09c2a2e4695e";
}

// TODO: this is not the actual version of XLA, but the version of the elixir xla prebuilts
pub const XLA_VERSION: semver::Version = semver::Version::new(0, 5, 1);
