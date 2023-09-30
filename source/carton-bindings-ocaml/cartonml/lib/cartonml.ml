open Option

type f_f_i_carton;;
type f_f_i_carton_error;;
external _load
  : string -> string option -> string option -> string option -> (f_f_i_carton, f_f_i_carton_error) Result.t
  = "__ocaml_ffi_load"
;;

let load ?(visible_device = none) ?(override_runner_name = none) ?(override_required_framework_version = none) path =
    _load path visible_device override_runner_name override_required_framework_version 
