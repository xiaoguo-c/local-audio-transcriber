from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from faster_whisper import WhisperModel


def fmt_time(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", "", text.strip())
    return text


def likely_question(text: str) -> bool:
    return bool(re.search(r"(吗|么|吧|呢|对不对|是不是|有没有|可以不可以|什么|哪里|哪儿|怎么|为什么|多少)[？?]?$", text))


def assign_speakers(segments: list[dict]) -> list[dict]:
    # This is a conservative dialogue formatting heuristic, not true diarization.
    # It alternates on long pauses and question-answer turns, while keeping short
    # continuation fragments with the previous speaker.
    current = "说话人A"
    previous_end = None
    previous_text = ""
    for item in segments:
        text = item["text"]
        pause = 0 if previous_end is None else item["start"] - previous_end
        switch = False
        if previous_end is not None:
            if pause >= 1.2:
                switch = True
            if likely_question(previous_text) and pause >= 0.25:
                switch = True
            if len(text) <= 4 and pause < 0.8:
                switch = False
        if switch:
            current = "说话人B" if current == "说话人A" else "说话人A"
        item["speaker"] = current
        previous_end = item["end"]
        previous_text = text
    return segments


def merge_dialogue(segments: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for item in segments:
        if (
            merged
            and merged[-1]["speaker"] == item["speaker"]
            and item["start"] - merged[-1]["end"] < 1.0
            and len(merged[-1]["text"]) < 120
        ):
            merged[-1]["end"] = item["end"]
            merged[-1]["text"] += item["text"]
        else:
            merged.append(dict(item))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", default="small")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    audio = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(
        str(audio),
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
    )

    segments = []
    for seg in segments_iter:
        text = clean_text(seg.text)
        if not text:
            continue
        segments.append({"start": seg.start, "end": seg.end, "text": text})

    segments = assign_speakers(segments)
    dialogue = merge_dialogue(segments)

    stem = audio.stem
    raw_path = outdir / f"{stem}_raw_segments.json"
    md_path = outdir / f"{stem}_dialogue.md"
    txt_path = outdir / f"{stem}_dialogue.txt"

    raw_path.write_text(
        json.dumps(
            {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# {stem} 转文字对话稿",
        "",
        f"- 识别语言：{info.language}（置信度 {info.language_probability:.2f}）",
        f"- 音频时长：{fmt_time(info.duration)}",
        "- 说明：说话人标签为按停顿和问答关系自动整理，非声纹级别分离。",
        "",
    ]
    for item in dialogue:
        lines.append(
            f"[{fmt_time(item['start'])}-{fmt_time(item['end'])}] {item['speaker']}：{item['text']}"
        )

    content = "\n".join(lines) + "\n"
    md_path.write_text(content, encoding="utf-8")
    txt_path.write_text("\n".join(lines[6:]) + "\n", encoding="utf-8")
    print(md_path)
    print(txt_path)
    print(raw_path)


if __name__ == "__main__":
    main()
