{
  pkgs,
  config,
  inputs,
  ...
}: {
  imports = [
    inputs.devkit.devenvModule
  ];

  # Not devkit's `modules.rust`: it pulls the whole Tauri/GTK/WebKit stack.
  languages.rust.enable = true;

  env.CARGO_HOME = "${config.env.DEVENV_STATE}/cargo";

  # ort is built with `load-dynamic` against API 24, so any onnxruntime >= 1.24 works.
  env.ORT_DYLIB_PATH = "${pkgs.onnxruntime}/lib/libonnxruntime.so";

  packages = with pkgs; [
    pkg-config
    flac
    sox
  ];
}
