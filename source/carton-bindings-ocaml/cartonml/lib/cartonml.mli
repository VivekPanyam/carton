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

external infer
  :  f_f_i_carton
  -> (string * tensor) array
  -> (string * tensor) array
  = "__ocaml_ffi_infer"

val load
  :  ?visible_device:string option
  -> ?override_runner_name:string option
  -> ?override_required_framework_version:string option
  -> string
  -> (f_f_i_carton, f_f_i_carton_error) Result.t
