//! General configuration for carton loaded from a config.toml file and the env
//! `Env var` overrides `config.toml` overrides `default`
//!
//! The path to the config file defaults to `~/.carton/config.toml` and can be overridden by
//! the `CARTON_CONFIG_PATH` env var.
//!
//! See `docs/config.md` for more info.

use lazy_static::lazy_static;
use serde::Deserialize;
use std::path::PathBuf;

lazy_static! {
    pub static ref CONFIG: CartonConfig = CartonConfig::load();
}

#[derive(Deserialize)]
#[serde(default)]
pub struct CartonConfig {
    /// The directory where runners are stored on disk
    /// Defaults to `~/.carton/runners/`
    /// Env: CARTON_RUNNER_DIR
    pub runner_dir: PathBuf,

    /// Runners can store data in `{runner_data_dir}/{runner_name}`
    /// Defaults to `~/.carton/runner_data/`
    /// Env: CARTON_RUNNER_DATA_DIR
    pub runner_data_dir: PathBuf,

    /// A directory where carton can cache downloads
    /// Defaults to `~/.carton/cache/`
    /// Env: CARTON_CACHE_DIR
    pub cache_dir: PathBuf,
}

impl Default for CartonConfig {
    fn default() -> Self {
        Self {
            runner_dir: shellexpand::tilde("~/.carton/runners/").to_string().into(),
            runner_data_dir: shellexpand::tilde("~/.carton/runner_data/")
                .to_string()
                .into(),
            cache_dir: shellexpand::tilde("~/.carton/cache/").to_string().into(),
        }
    }
}

impl CartonConfig {
    fn load() -> CartonConfig {
        // Load the config
        let mut config = match std::env::var("CARTON_CONFIG_PATH") {
            Ok(p) => {
                let config_path: PathBuf = shellexpand::tilde(&p).to_string().into();
                if !config_path.exists() {
                    panic!("CARTON_CONFIG_PATH was set to `{p}` which does not exist");
                }

                toml::from_slice(&std::fs::read(config_path).unwrap()).unwrap()
            }

            Err(_) => {
                let config_path: PathBuf = shellexpand::tilde("~/.carton/config.toml")
                    .to_string()
                    .into();

                if config_path.exists() {
                    toml::from_slice(&std::fs::read(config_path).unwrap()).unwrap()
                } else {
                    CartonConfig::default()
                }
            }
        };

        // Override with env
        if let Ok(v) = std::env::var("CARTON_RUNNER_DIR") {
            config.runner_dir = shellexpand::tilde(&v).to_string().into();
        }

        if let Ok(v) = std::env::var("CARTON_RUNNER_DATA_DIR") {
            config.runner_data_dir = shellexpand::tilde(&v).to_string().into();
        }

        if let Ok(v) = std::env::var("CARTON_CACHE_DIR") {
            config.cache_dir = shellexpand::tilde(&v).to_string().into();
        }

        config
    }
}
