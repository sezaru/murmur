# murmur

Local speech-to-text in Rust: a dictation daemon that types what you say into the focused
window, and a test bench that feeds one audio stream to several models at once so their
transcripts and latencies can be compared side by side.

## Models

| preset | kind | how it produces text |
|---|---|---|
| `unified` | cache-aware Parakeet (`parakeet-unified-en-0.6b` int8, what voxtype runs) | true streaming: appends text every 0.56s chunk, ignores pauses |
| `phonon2` | offline Parakeet TDT ([Phonon-2](https://huggingface.co/FermionResearch/Phonon-2), int8 ONNX export) | transcribes each phrase once you pause |

Any other model of these kinds: `--model name=kind:/path/to/dir` with kind `tdt` or `unified`.

Pauses come from [earshot](https://crates.io/crates/earshot), a voice activity detector shared by
every model: a phrase ends after `--silence-ms` (default 500) of silence, or is cut at
`--max-segment` seconds (default 25).

## Setup

```sh
scripts/fetch-models          # phonon2 into ~/.local/share/murmur/models; `unified` needs MURMUR_UNIFIED_URL
scripts/fetch-corpus [name]   # librispeech-dummy (default), ami, earnings22 clips for `bench`
cargo build --release
```

`unified` reuses voxtype's copy under `~/.local/share/voxtype/models` when it exists.

## Dictation

```sh
murmur daemon -m unified --replace MX=Emacs    # usually a systemd user service
murmur toggle                                  # bind this to a key
```

`toggle` starts recording, or stops it; if no daemon answers it starts `murmur.service` first
(`--unit` to name another). The mic opens at the keypress and audio is buffered while the model
loads, so you can talk straight away. Text is typed with `wtype` a whole word at a time (a
streaming model can end a chunk mid-word); anything that fails to type is put on the clipboard
at the end. Media players that were playing are paused for the duration.

Without `--resident` the daemon exits after each dictation, freeing the model's ~1.1GB.

State for an on-screen indicator is in `$XDG_RUNTIME_DIR/murmur/state`: `initializing` (written by
`toggle` before the daemon is up), `streaming`, `transcribing` (draining after a stop), `idle`;
the file is gone when no daemon is running. `murmur status` prints it.

## Compare live

```sh
murmur compare -m unified -m phonon2
```

One column per model. `q` stops; transcripts and the recording are saved under
`~/.local/share/murmur/sessions/<timestamp>/`. Replay a recording through the same pipeline, in
real time as if it were the mic, or with `--fast`:

```sh
murmur compare -m phonon2 --replay ~/.local/share/murmur/sessions/<timestamp>/audio.wav
```

**lag** is the time from when you stopped speaking to when a model's final text appeared (for
`unified`, from the end of the audio chunk that produced the text). Models running together
compete for the same cores, so replay a recording with one model for clean timings. The footer
shows process RSS, system-wide zswapped memory and major page faults, so a run that spilled to
the swapfile is obvious.

## Bench

```sh
murmur bench --manifest ~/.local/share/murmur/corpus/librispeech-dummy/manifest.tsv \
  -m unified -m phonon2
```

Each model runs in its own process over the same pipeline, unpaced. Columns:

- **WER**: word error rate, ignoring case and punctuation.
- **short**: WER on clips of four words or fewer.
- **empty**: clips that produced no text.
- **RTFx**: audio seconds per compute second.
- **final**: time to produce a phrase's final text after the pause is detected.
- **peak**: peak RSS.

`--show` prints every clip that has errors.
