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

//! Utility function to log if a task is taking a long time

use std::{
    fmt::Display,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use tokio::sync::oneshot;

pub struct Progress<T> {
    progress: Option<T>,
    total: Option<T>,
}

impl<T> Default for Progress<T> {
    fn default() -> Self {
        Self {
            progress: Default::default(),
            total: Default::default(),
        }
    }
}

pub struct SlowLog<T> {
    done: Option<oneshot::Sender<()>>,

    // This is okay because it's likely not going to have any significant contention
    progress: Arc<Mutex<Progress<T>>>,
}

impl<T> SlowLog<T> {
    pub fn done(&mut self) {
        self.done.take().map(|d| d.send(()).unwrap());
    }

    pub fn set_total(&self, total: Option<T>) {
        self.progress.lock().unwrap().total = total;
    }

    pub fn set_progress(&self, progress: Option<T>) {
        self.progress.lock().unwrap().progress = progress;
    }
}

pub struct WithoutProgress;
impl Display for WithoutProgress {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl SlowLog<WithoutProgress> {
    /// Just a hint to the compiler so it can figure out the type of T if we
    /// never call `set_progress` or `set_total`
    pub fn without_progress(self) -> Self {
        self
    }
}

impl<T> Drop for SlowLog<T> {
    fn drop(&mut self) {
        self.done();
    }
}

pub async fn slowlog<S, T>(msg: S, interval_seconds: u64) -> SlowLog<T>
where
    S: Into<String>,
    T: Send + 'static + Display,
{
    let msg = msg.into();

    // Holds progress information
    let progress = Arc::new(Mutex::new(Progress::default()));

    let progress2 = progress.clone();
    let (tx, mut rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let start = Instant::now();
        loop {
            match tokio::time::timeout(Duration::from_secs(interval_seconds), &mut rx).await {
                Ok(_) => break,
                Err(_) => {
                    // Check if we have progress info
                    let p = {
                        let guard = progress2.lock().unwrap();
                        match (&guard.progress, &guard.total) {
                            (None, None) => "".to_string(),
                            (None, Some(total)) => format!(" ({total})"),
                            (Some(progress), None) => format!(" ({progress} / unknown)"),
                            (Some(progress), Some(total)) => format!(" ({progress} / {total})"),
                        }
                    };

                    // Get the duration since we started and log
                    let duration = start.elapsed().as_secs();
                    log::info!(target: "slowlog", "Task running for {duration} seconds: {msg}{p}")
                }
            }
        }
    });

    SlowLog {
        done: Some(tx),
        progress,
    }
}
