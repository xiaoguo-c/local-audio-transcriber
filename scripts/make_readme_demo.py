from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "app-screenshot.png"
OUTPUT = ROOT / "docs" / "demo.gif"


def font(size: int):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_frame(base: Image.Image, title: str, subtitle: str, accent: str) -> Image.Image:
    frame = base.copy().convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, _ = frame.size
    box = (40, 34, width - 40, 150)
    draw.rounded_rectangle(box, radius=14, fill=(255, 255, 255, 235), outline=accent, width=4)
    draw.text((64, 52), title, fill=(23, 32, 39, 255), font=font(34))
    draw.text((64, 100), subtitle, fill=(82, 94, 105, 255), font=font(22))
    return Image.alpha_composite(frame, overlay).convert("P", palette=Image.Palette.ADAPTIVE)


def main() -> None:
    base = Image.open(SOURCE)
    base.thumbnail((1100, 760))
    frames = [
        make_frame(base, "1. Drag in an audio file", "Drop m4a/mp3/wav recordings into the local web app.", "#0f766e"),
        make_frame(base, "2. Choose ASR + organizer", "Use Whisper or FunASR, then optionally clean up with an LLM.", "#2563eb"),
        make_frame(base, "3. Copy a polished transcript", "Get dialogue-style notes for recruiting calls and meetings.", "#7c3aed"),
    ]
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=1250,
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
