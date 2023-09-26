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

use wasm_bindgen::prelude::*;

use std::io::Error;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::io::ReadBuf;
use tokio::sync::mpsc;
use std::fmt::Debug;

use serde::{de::DeserializeOwned, Serialize};

use super::{framed::frame, types::ChannelId};

#[wasm_bindgen]
struct Channel {
    read: wasm_streams::ReadableStream,
    write: wasm_streams::WritableStream,
}

#[wasm_bindgen]
extern "C" {
    pub type WasmRunnerLauncher;
    type WasmRunner;

    // Starts a new runner
    // TODO: async
    #[wasm_bindgen(method)]
    fn launch_runner(this: &WasmRunnerLauncher) -> Option<WasmRunner>;

    // These can be turned into tokio streams that we can read and write to
    // TODO: async
    // Note that the channel id is just types::ChannelId as a u8
    #[wasm_bindgen(method)]
    fn get_channel(this: &WasmRunner, channel_id: u8) -> Channel;
}

#[wasm_bindgen]
pub fn register_launcher(launcher: &WasmRunnerLauncher) {}

pub struct Comms {
    inner: WasmRunner,
}

impl Comms {
    pub async fn new() -> Self {
        // todo: get and store inner
        todo!();
    }

    /// A framed transport that can transport serializable things on top of a bidirectional stream.
    /// Note: this can only be called once per channel id
    pub(crate) async fn get_channel<T, U>(
        &self,
        channel_id: ChannelId,
    ) -> (mpsc::Sender<T>, mpsc::Receiver<U>)
    where
        T: Debug + Serialize + Send + 'static,
        U: Debug + DeserializeOwned + Send + 'static,
    {
        let channel = self.inner.get_channel(channel_id as _);
        frame(Wrapper { inner: channel.read }, Wrapper { inner: channel.write }).await
    }
}

struct Wrapper<T> {
    inner: T,
}

impl<T> Wrapper<T> {
    fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl AsyncRead for Wrapper<wasm_streams::ReadableStream> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        todo!();
    }
}

impl AsyncWrite for Wrapper<wasm_streams::WritableStream> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Error>> {
        todo!();
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        todo!();
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        todo!()
    }
}

// No difference in wasm
pub type OwnedComms = Comms;