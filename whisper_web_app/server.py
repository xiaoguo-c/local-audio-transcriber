from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
FUNASR_DEPS = Path("D:/codex_python_deps/funasr")
FFMPEG_BIN = FUNASR_DEPS / "imageio_ffmpeg" / "binaries"
os.environ.setdefault("MODELSCOPE_CACHE", "D:/modelscope_cache")
os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
if FUNASR_DEPS.exists():
    sys.path.insert(0, str(FUNASR_DEPS))
if (FFMPEG_BIN / "ffmpeg.exe").exists():
    os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
MODEL_LOCK = threading.Lock()
MODELS: dict[str, WhisperModel] = {}
FUNASR_MODELS: dict[str, object] = {}
ALLOWED_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v3"}
ALLOWED_FUNASR_MODELS = {"paraformer-zh"}
JOB_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}


def set_job(job_id: str, **updates) -> None:
    with JOB_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)


def get_job(job_id: str) -> dict | None:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def fmt_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def likely_question(text: str) -> bool:
    return bool(
        re.search(
            r"(吗|么|吧|呢|对不对|是不是|有没有|可以不可以|什么|哪里|哪儿|怎么|为什么|多少)[？?]?$",
            text,
        )
    )


def assign_speakers(segments: list[dict]) -> list[dict]:
    current = "说话人A"
    previous_end = None
    previous_text = ""
    for item in segments:
        pause = 0 if previous_end is None else item["start"] - previous_end
        switch = False
        if previous_end is not None:
            if pause >= 1.2:
                switch = True
            if likely_question(previous_text) and pause >= 0.25:
                switch = True
            if len(item["text"]) <= 4 and pause < 0.8:
                switch = False
        if switch:
            current = "说话人B" if current == "说话人A" else "说话人A"
        item["speaker"] = current
        previous_end = item["end"]
        previous_text = item["text"]
    return segments


def merge_dialogue(segments: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for item in segments:
        if (
            merged
            and merged[-1]["speaker"] == item["speaker"]
            and item["start"] - merged[-1]["end"] < 1.0
            and len(merged[-1]["text"]) < 140
        ):
            merged[-1]["end"] = item["end"]
            merged[-1]["text"] += item["text"]
        else:
            merged.append(dict(item))
    return merged


def dialogue_text(dialogue: list[dict]) -> str:
    lines = []
    for item in dialogue:
        lines.append(
            f"[{fmt_time(item['start'])}-{fmt_time(item['end'])}] "
            f"{item['speaker']}：{item['text']}"
        )
    return "\n".join(lines)


def split_text_by_punctuation(text: str, max_chars: int = 80) -> list[str]:
    parts = []
    current = ""
    for char in text:
        current += char
        if char in "。！？!?；;，,吗呢吧" or len(current) >= max_chars:
            piece = current.strip("，,。！？!?；; ")
            if piece:
                parts.append(piece)
            current = ""
    if current.strip():
        parts.append(current.strip("，,。！？!?；; "))
    return parts


def segments_from_text(text: str) -> list[dict]:
    parts = split_text_by_punctuation(text)
    segments = []
    cursor = 0.0
    for part in parts:
        # Approximate timings keep the dialogue formatter useful when FunASR
        # returns text without sentence timestamps.
        duration = max(1.0, min(8.0, len(part) / 6))
        segments.append({"start": cursor, "end": cursor + duration, "text": part})
        cursor += duration + 0.35
    return segments


def segments_from_funasr_timestamps(text: str, timestamps: list) -> list[dict]:
    parts = split_text_by_punctuation(text)
    if not parts or not timestamps:
        return []

    segments = []
    char_cursor = 0
    for part in parts:
        start_index = char_cursor
        end_index = min(len(timestamps) - 1, char_cursor + len(part) - 1)
        if start_index >= len(timestamps):
            break
        start_ts = timestamps[start_index]
        end_ts = timestamps[end_index]
        if isinstance(start_ts, (list, tuple)):
            start = float(start_ts[0]) / 1000
        else:
            start = float(start_ts) / 1000
        if isinstance(end_ts, (list, tuple)):
            end = float(end_ts[-1]) / 1000
        else:
            end = float(end_ts) / 1000
        segments.append({"start": start, "end": max(start, end), "text": part})
        char_cursor += len(part)
    return segments


def get_model(name: str) -> WhisperModel:
    with MODEL_LOCK:
        if name not in MODELS:
            MODELS[name] = WhisperModel(name, device="cpu", compute_type="int8")
        return MODELS[name]


def parse_engine_model(value: str) -> tuple[str, str]:
    if ":" not in value:
        return "whisper", value
    engine, model_name = value.split(":", 1)
    return engine.strip(), model_name.strip()


def get_funasr_model(name: str):
    if not FUNASR_DEPS.exists():
        raise RuntimeError("FunASR dependencies are not installed at D:/codex_python_deps/funasr")
    with MODEL_LOCK:
        if name not in FUNASR_MODELS:
            from funasr import AutoModel

            if name == "paraformer-zh":
                FUNASR_MODELS[name] = AutoModel(
                    model="paraformer-zh",
                    vad_model="fsmn-vad",
                    disable_update=True,
                    device="cpu",
                )
            else:
                raise ValueError("Unsupported FunASR model")
        return FUNASR_MODELS[name]


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        raise ValueError("Missing multipart boundary")

    boundary = match.group("boundary").strip('"').encode()
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    marker = b"--" + boundary

    for part in body.split(marker):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        header_blob, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue
        payload = payload.rstrip(b"\r\n")
        headers = header_blob.decode("utf-8", errors="replace")
        disposition = re.search(r'name="([^"]+)"(?:;\s*filename="([^"]*)")?', headers)
        if not disposition:
            continue
        name = disposition.group(1)
        filename = disposition.group(2)
        if filename is None:
            fields[name] = payload.decode("utf-8", errors="replace")
        else:
            files[name] = (Path(filename).name or "audio", payload)
    return fields, files


def transcribe_audio(audio_path: Path, model_name: str, job_id: str | None = None) -> dict:
    started = time.time()
    if job_id:
        set_job(job_id, status="loading", message="正在加载模型", progress=0.03)
    model = get_model(model_name)
    if job_id:
        set_job(job_id, status="transcribing", message="正在识别音频", progress=0.08)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
    )

    segments = []
    for seg in segments_iter:
        text = clean_text(seg.text)
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})
        if job_id:
            progress = 0.08
            if info.duration:
                progress = min(0.98, max(0.08, seg.end / info.duration))
            set_job(
                job_id,
                status="transcribing",
                message=f"已识别到 {fmt_time(seg.end)} / {fmt_time(info.duration)}",
                progress=progress,
                duration=info.duration,
                current_time=seg.end,
                partial_text=dialogue_text(merge_dialogue(assign_speakers([dict(item) for item in segments]))),
            )

    segments = assign_speakers(segments)
    dialogue = merge_dialogue(segments)
    text = dialogue_text(dialogue)
    result = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "elapsed": time.time() - started,
        "model": model_name,
        "segments": segments,
        "dialogue": dialogue,
        "text": text,
    }

    output_path = OUTPUTS / f"{audio_path.stem}_{int(time.time())}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_file"] = str(output_path)
    if job_id:
        set_job(
            job_id,
            status="done",
            message="转写完成",
            progress=1,
            current_time=info.duration,
            duration=info.duration,
            result=result,
            partial_text=text,
        )
    return result


def normalize_funasr_result(raw_result: list[dict], elapsed: float, model_name: str) -> dict:
    text = ""
    segments: list[dict] = []
    if raw_result:
        text = clean_text(raw_result[0].get("text", ""))
        sentence_info = raw_result[0].get("sentence_info") or []
        for sentence in sentence_info:
            start = float(sentence.get("start", 0)) / 1000
            end = float(sentence.get("end", 0)) / 1000
            item_text = clean_text(sentence.get("text", ""))
            if item_text:
                segments.append({"start": start, "end": end, "text": item_text})

    if not segments and text:
        timestamps = raw_result[0].get("timestamp") or [] if raw_result else []
        segments = segments_from_funasr_timestamps(text, timestamps)

    if not segments and text:
        segments = segments_from_text(text)

    segments = assign_speakers(segments)
    dialogue = merge_dialogue(segments)
    final_text = dialogue_text(dialogue) if dialogue else text
    duration = max((item["end"] for item in segments), default=0.0)
    return {
        "language": "zh",
        "language_probability": None,
        "duration": duration,
        "elapsed": elapsed,
        "model": f"funasr:{model_name}",
        "segments": segments,
        "dialogue": dialogue,
        "text": final_text,
        "raw": raw_result,
    }


def transcribe_funasr(audio_path: Path, model_name: str, job_id: str | None = None) -> dict:
    started = time.time()
    if job_id:
        set_job(job_id, status="loading", message="正在加载 FunASR 模型", progress=0.05)
    model = get_funasr_model(model_name)
    if job_id:
        set_job(job_id, status="transcribing", message="FunASR 正在识别音频", progress=0.3)

    raw_result = model.generate(
        input=str(audio_path),
        batch_size_s=300,
        sentence_timestamp=True,
    )
    if job_id:
        set_job(job_id, status="formatting", message="正在整理 FunASR 结果", progress=0.9)

    result = normalize_funasr_result(raw_result, time.time() - started, model_name)
    output_path = OUTPUTS / f"{audio_path.stem}_{int(time.time())}_funasr.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_file"] = str(output_path)
    if job_id:
        set_job(
            job_id,
            status="done",
            message="转写完成",
            progress=1,
            current_time=result["duration"],
            duration=result["duration"],
            result=result,
            partial_text=result["text"],
        )
    return result


def run_job(job_id: str, audio_path: Path, engine: str, model_name: str) -> None:
    try:
        if engine == "funasr":
            transcribe_funasr(audio_path, model_name, job_id=job_id)
        else:
            transcribe_audio(audio_path, model_name, job_id=job_id)
    except Exception as exc:
        set_job(job_id, status="error", message=str(exc), progress=0)


class Handler(SimpleHTTPRequestHandler):
    server_version = "LocalWhisperWeb/1.0"

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean_path = unquote(parsed.path).lstrip("/")
        if not clean_path:
            return str(STATIC / "index.html")
        requested = (STATIC / clean_path).resolve()
        if not requested.is_relative_to(STATIC.resolve()):
            return str(STATIC / "index.html")
        return str(requested)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "cached_whisper_models": sorted(MODELS),
                    "cached_funasr_models": sorted(FUNASR_MODELS),
                    "funasr_available": FUNASR_DEPS.exists(),
                }
            )
            return
        job_match = re.fullmatch(r"/api/jobs/([0-9a-f]+)", parsed.path)
        if job_match:
            job = get_job(job_match.group(1))
            if not job:
                self.send_json({"error": "Job not found"}, status=404)
                return
            self.send_json(job)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/transcribe":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", "0"))
            fields, files = parse_multipart(content_type, self.rfile.read(length))
            engine, model_name = parse_engine_model(fields.get("model", "whisper:small"))
            if engine == "whisper" and model_name not in ALLOWED_WHISPER_MODELS:
                raise ValueError("Unsupported Whisper model")
            if engine == "funasr" and model_name not in ALLOWED_FUNASR_MODELS:
                raise ValueError("Unsupported FunASR model")
            if engine not in {"whisper", "funasr"}:
                raise ValueError("Unsupported engine")
            if "audio" not in files:
                raise ValueError("No audio file uploaded")

            filename, payload = files["audio"]
            suffix = Path(filename).suffix or ".audio"
            audio_path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
            audio_path.write_bytes(payload)
            job_id = uuid.uuid4().hex
            with JOB_LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "queued",
                    "message": "已上传，等待开始识别",
                    "progress": 0.01,
                    "filename": filename,
                    "engine": engine,
                    "model": f"{engine}:{model_name}",
                    "created_at": time.time(),
                    "partial_text": "",
                }
            worker = threading.Thread(
                target=run_job,
                args=(job_id, audio_path, engine, model_name),
                daemon=True,
            )
            worker.start()
            self.send_json({"job_id": job_id})
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def copyfile(self, source, outputfile) -> None:
        shutil.copyfileobj(source, outputfile)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    UPLOADS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer(("127.0.0.1", 8787), Handler)
    print("Local Whisper web app running at http://127.0.0.1:8787")
    server.serve_forever()


if __name__ == "__main__":
    main()
