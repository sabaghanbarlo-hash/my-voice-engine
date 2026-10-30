# my-voice-engine

Paste a script, get it read aloud in **my own cloned voice**. Engine for the page in `my-voice-studio`.

- `tts.py` reads a script with [Voice Studio](https://github.com/debpalash/VoiceStudio) (OmniVoice), cloning the voice in `voice/ref.m4a`. Free, CPU only.
- `.github/workflows/speak.yml` runs it on GitHub Actions (input: a file in `scripts/`), saves the MP3 in `results/`.
- `voice/ref.m4a` is the voice sample (copied from the `yuki-video` repo). This repo is private on purpose.
- Free CPU is slow: expect a few minutes per sentence on the first run (finished sentences are cached).
