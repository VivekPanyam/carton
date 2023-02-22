Carton can be configured via a configuration file or environment variables (or a mix of both).

Environment variables override the values in the configuration file which override the default values.

The location of the configuration file defaults to `~/.carton/config.toml` and can be overridden with the `CARTON_CONFIG_PATH` environment variable.

The table below shows configuration options and their default values:

Env var | config file key | default value | description
--- | --- | --- | ---
`CARTON_RUNNER_DIR` | `runner_dir` | `~/.carton/runners/` | The directory where runners are stored on disk
`CARTON_RUNNER_DATA_DIR` | `runner_data_dir` | `~/.carton/runner_data/` | Runners can store caches or local data in `{runner_data_dir}/{runner_name}`