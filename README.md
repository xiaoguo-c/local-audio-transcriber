# Local Audio Transcription Web App

A local drag-and-drop web app for Chinese audio transcription. It supports:

- faster-whisper model selection
- FunASR Paraformer Chinese model selection
- upload progress and transcription status
- dialogue-style output
- copy and TXT download

## Start

```powershell
.\whisper_web_app\start.ps1
```

Then open:

```text
http://127.0.0.1:8787
```

## Notes

FunASR dependencies are expected at:

```text
D:\codex_python_deps\funasr
```

FunASR / ModelScope model cache is expected at:

```text
D:\modelscope_cache
```

The app keeps uploaded audio and generated outputs out of Git.
