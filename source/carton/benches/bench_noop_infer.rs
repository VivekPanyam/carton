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

//! This benchmark tests overhead of inference with a noop runner (which should be ~carton's overhead)
use std::collections::HashMap;

use carton::{
    info::RunnerInfo,
    types::{CartonInfo, GenericStorage, LoadOpts, PackOpts},
    Carton,
};
use criterion::{criterion_group, criterion_main, Criterion};
use semver::VersionReq;

fn infer_noop_benchmark(c: &mut Criterion) {
    console_subscriber::init();

    // Make sure the noop runner is built
    let runner_path = escargot::CargoBuild::new()
        .package("carton-runner-noop")
        .release()
        .arg("--timings")
        .run()
        .unwrap()
        .path()
        .display()
        .to_string();
    println!("Runner Path: {}", runner_path);

    println!("Creating runner.toml");
    let runner_toml = format!(
        r#"
# This is a runner.toml that runs against the release runner
version = 1

[[runner]]
runner_name = "noop"
framework_version = "1.0.0"
runner_compat_version = 1
runner_interface_version = 1
runner_release_date = "1979-05-27T07:32:00Z"

# A path to the runner binary. This can be absolute or relative to this file
runner_path = "{runner_path}"

# A target triple
platform = "{}"
"#,
        target_lexicon::HOST.to_string()
    );

    let tempdir = tempfile::tempdir().unwrap();
    std::fs::write(tempdir.path().join("runner.toml"), runner_toml).unwrap();

    // TODO don't do this
    std::env::set_var("CARTON_RUNNER_DIR", tempdir.path().as_os_str());

    println!("Creating runtime");
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let info: CartonInfo<GenericStorage> = CartonInfo {
        model_name: None,
        short_description: None,
        model_description: None,
        license: None,
        repository: None,
        homepage: None,
        required_platforms: None,
        inputs: None,
        outputs: None,
        self_tests: None,
        examples: None,
        runner: RunnerInfo {
            runner_name: "noop".into(),
            required_framework_version: VersionReq::parse("*").unwrap(),
            runner_compat_version: None,
            opts: None,
        },
        misc_files: None,
    };

    let load_opts = LoadOpts::default();
    let carton = runtime
        .block_on(Carton::load_unpacked(
            "/tmp",
            PackOpts {
                info,
                linked_files: None,
            },
            load_opts,
        ))
        .unwrap();

    c.bench_function("infer_noop", |b| {
        b.to_async(&runtime).iter(|| async {
            let tensors: HashMap<String, carton::types::Tensor<GenericStorage>> = HashMap::new();
            carton.infer(tensors).await.unwrap();
        })
    });
}

criterion_group!(benches, infer_noop_benchmark);
criterion_main!(benches);
