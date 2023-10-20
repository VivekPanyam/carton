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

use std::sync::Arc;

use s3::{creds::Credentials, Bucket, Region};

/// Upload the C and C++ bindings in CI
#[tokio::main]
async fn main() {
    let bucket = Arc::new(
        Bucket::new(
            &std::env::var("CARTON_NIGHTLY_S3_BUCKET").unwrap(),
            Region::Custom {
                region: std::env::var("CARTON_NIGHTLY_S3_REGION").unwrap(),
                endpoint: std::env::var("CARTON_NIGHTLY_S3_ENDPOINT").unwrap(),
            },
            Credentials::new(
                Some(&std::env::var("CARTON_NIGHTLY_ACCESS_KEY_ID").unwrap()),
                Some(&std::env::var("CARTON_NIGHTLY_SECRET_ACCESS_KEY").unwrap()),
                None,
                None,
                None,
            )
            .unwrap(),
        )
        .unwrap(),
    );

    let handles = [
        "/tmp/artifacts/c-cpp-bindings-linux",
        "/tmp/artifacts/c-cpp-bindings-macos-aarch64",
        "/tmp/artifacts/c-cpp-bindings-x86_64-apple-darwin",
    ]
    .into_iter()
    .flat_map(|dir| {
        std::fs::read_dir(dir)
            .unwrap()
            .into_iter()
            .filter_map(|item| item.ok())
            .map(|item| (item.path(), item.file_name().to_str().unwrap().to_owned()))
    })
    .map(|(path, filename)| {
        let bucket = bucket.clone();
        tokio::spawn(async move {
            let content = tokio::fs::read(path).await.unwrap();

            // Upload the zip file
            bucket
                .put_object(format!("/bindings/{}", filename), &content)
                .await
                .unwrap();
        })
    });

    for handle in handles {
        handle.await.unwrap();
    }
}
