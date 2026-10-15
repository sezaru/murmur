{
  lib,
  rustPlatform,
  makeWrapper,
  onnxruntime,
  pipewire,
  wtype,
  wl-clipboard,
  playerctl,
  libnotify,
  systemd,
}:
rustPlatform.buildRustPackage {
  pname = "murmur";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ../.;
    filter = path: _: let
      rel = lib.removePrefix (toString ../. + "/") (toString path);
    in
      !(lib.any (p: lib.hasPrefix p rel) ["target" ".devenv" ".direnv" ".agent-tmp" "result"]);
  };

  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [makeWrapper];

  # ort is built with `load-dynamic`, so onnxruntime is found through ORT_DYLIB_PATH on every
  # launch path, and the daemon shells out to the rest.
  postInstall = ''
    wrapProgram $out/bin/murmur \
      --set ORT_DYLIB_PATH ${onnxruntime}/lib/libonnxruntime.so \
      --prefix PATH : ${lib.makeBinPath [pipewire wtype wl-clipboard playerctl libnotify systemd]}
  '';

  meta = {
    description = "Local dictation and side-by-side speech model comparison";
    mainProgram = "murmur";
    license = lib.licenses.mit;
    platforms = lib.platforms.linux;
  };
}
