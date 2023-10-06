use std::collections::HashMap;

use color_eyre::eyre::{eyre, Result};
use wasmtime::{Engine, Store};
use wasmtime::component::*;

use carton_runner_interface::types::Tensor as CartonTensor;

use crate::component::{DummyState, Model, Tensor};

mod component;
mod types;

pub struct WASMModelInstance {
    instance: Instance,
    store: Store<DummyState>,
    model: Model,
}

impl WASMModelInstance {
    pub fn from_bytes(engine: &Engine, bytes: &[u8]) -> Result<Self> {
        let comp = Component::from_binary(&engine, bytes).unwrap();
        let mut linker = Linker::<DummyState>::new(&engine);
        Model::add_to_linker(&mut linker, |state: &mut DummyState| state).unwrap();
        let mut store = Store::new(&engine, DummyState);
        let (model, instance) = Model::instantiate(&mut store, &comp, &linker).unwrap();
        Ok(Self {
            instance,
            store,
            model,
        })
    }

    pub fn infer(&mut self, inputs: HashMap<String, CartonTensor>) -> Result<HashMap<String, CartonTensor>> {
        let inputs = inputs.into_iter()
            .map(|(k, v)| Ok((k, v.try_into()?)))
            .collect::<Result<Vec<(String, Tensor)>>>()?;
        let outputs = self.model.call_infer(&mut self.store, inputs.as_ref())
            .map_err(|e| eyre!(e))?;
        let mut ret = HashMap::new();
        for (k, v) in outputs.into_iter() {
            ret.insert(k, v.into());
        }
        Ok(ret)
    }
}
