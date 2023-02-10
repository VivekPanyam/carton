//! Utility function to log if a task is taking a long time

use std::time::{Duration, Instant};

use tokio::sync::oneshot;

pub struct SlowLog {
    done: Option<oneshot::Sender<()>>,
}

impl SlowLog {
    pub fn done(&mut self) {
        self.done.take().map(|d| d.send(()).unwrap());
    }
}

impl Drop for SlowLog {
    fn drop(&mut self) {
        self.done();
    }
}

pub async fn slowlog<S>(msg: S, interval_seconds: u64) -> SlowLog
where
    S: Into<String>,
{
    let msg = msg.into();
    let (tx, mut rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let start = Instant::now();
        loop {
            match tokio::time::timeout(Duration::from_secs(interval_seconds), &mut rx).await {
                Ok(_) => break,
                Err(_) => {
                    let duration = start.elapsed().as_secs();
                    log::info!(target: "slowlog", "Task running for {duration} seconds: {msg}")
                }
            }
        }
    });

    SlowLog { done: Some(tx) }
}
