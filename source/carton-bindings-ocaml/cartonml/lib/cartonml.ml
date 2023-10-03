open Option

type f_f_i_carton
type f_f_i_carton_error

type tensor =
  | Float of float array
  | Double of float array
  | I8 of Int64.t array
  | I16 of Int64.t array
  | I32 of Int64.t array
  | I64 of Int64.t array
  | U8 of Int64.t array
  | U16 of Int64.t array
  | U32 of Int64.t array
  | U64 of Int64.t array
  | String of string array
[@@boxed]

external _load
  :  string
  -> string option
  -> string option
  -> string option
  -> (f_f_i_carton, f_f_i_carton_error) Result.t
  = "__ocaml_ffi_load"

external infer
  :  f_f_i_carton
  -> (string * tensor) array
  -> (string * tensor) array
  = "__ocaml_ffi_infer"

let load
  ?(visible_device = none)
  ?(override_runner_name = none)
  ?(override_required_framework_version = none)
  path
  =
  _load
    path
    visible_device
    override_runner_name
    override_required_framework_version
;;
