use std::collections::HashMap;

use color_eyre::eyre::{eyre, Result};
use wasmtime::component::{Component, Linker};
use wasmtime::{Engine, Store};

use carton_runner_interface::types::Tensor as CartonTensor;

use crate::component::{HostImpl, Model, Tensor};

mod component;
mod types;

pub struct WASMModelInstance {
    store: Store<HostImpl>,
    model: Model,
}

impl WASMModelInstance {
    pub fn from_bytes(engine: &Engine, bytes: &[u8]) -> Result<Self> {
        /*
        see https://docs.wasmtime.dev/api/wasmtime/component/macro.bindgen.html
        Some of the names may be confusing, here is the general idea from my
        understanding:
        - HostImpl is the host side implementation of what a interface imports
          since our current interface does not import anything, this is an empty
          struct
        - Model is the loaded and linked interface, i.e. the API we expect the
          user to implement. (Non stateful)
          TODO: rename to ModelInterface
         */
        let comp = Component::from_binary(&engine, bytes).unwrap();
        let mut linker = Linker::<HostImpl>::new(&engine);
        Model::add_to_linker(&mut linker, |state: &mut HostImpl| state).unwrap();
        let mut store = Store::new(&engine, HostImpl);
        let (model, _) = Model::instantiate(&mut store, &comp, &linker).unwrap();
        Ok(Self { store, model })
    }

    pub fn infer(
        &mut self,
        inputs: HashMap<String, CartonTensor>,
    ) -> Result<HashMap<String, CartonTensor>> {
        let inputs = inputs
            .into_iter()
            .map(|(k, v)| Ok((k, v.try_into()?)))
            .collect::<Result<Vec<(String, Tensor)>>>()?;
        let outputs = self
            .model
            .call_infer(&mut self.store, inputs.as_ref())
            .map_err(|e| eyre!(e))?;
        let mut ret = HashMap::new();
        for (k, v) in outputs.into_iter() {
            ret.insert(k, v.into());
        }
        Ok(ret)
    }
}
