import base64
import os
from openai import OpenAI

print("OPENAI_API_KEY IN OCR:", repr(os.getenv("OPENAI_API_KEY")))

def _get(obj, key, default=None):
    # dict면 key로, 객체면 attribute로
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

# ocr.py (추가)
import io
from PIL import Image

def _split_image_bytes(
    image_bytes: bytes,
    *,
    target_min_width: int = 900,
    chunk_height: int = 1100,
    overlap: int = 140,
) -> list[bytes]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1) 너무 좁으면 업스케일 (OCR 성능에 직결)
    if img.width < target_min_width:
        scale = target_min_width / img.width
        new_w = target_min_width
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # 2) 세로로 자르기 (오버랩 넣어서 경계 글자 깨짐 방지)
    chunks: list[bytes] = []
    y = 0
    while y < img.height:
        y2 = min(img.height, y + chunk_height)
        tile = img.crop((0, y, img.width, y2))

        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        chunks.append(buf.getvalue())

        if y2 >= img.height:
            break
        y = max(0, y2 - overlap)

    return chunks


def _dedup_adjacent_lines(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    out = []
    prev = None
    for ln in lines:
        if not ln:
            continue
        if ln == prev:
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out)


def run_ocr_tiled(image_bytes: bytes, content_type: str | None) -> str:
    tiles = _split_image_bytes(image_bytes)

    parts: list[str] = []
    for tb in tiles:
        t = run_ocr(tb, "image/png")  # 타일은 PNG로 통일
        if t and t.strip():
            parts.append(t.strip())

    merged = "\n".join(parts)
    return _dedup_adjacent_lines(merged).strip()


def run_ocr(image_bytes: bytes, content_type: str | None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    if not content_type or not content_type.startswith("image/"):
        content_type = "image/png"

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{image_b64}"

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "이미지에 있는 모든 텍스트만 그대로 추출해라.\n"
                            "- 추측 금지\n"
                            "- 보이는 순서 유지\n"
                            "- 설명 금지\n"
                            "- 텍스트 없으면 빈 문자열"
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url
                    }
                ]
            }
        ]
    )

    # 1️⃣ output_text 우선
    if getattr(response, "output_text", None):
        return response.output_text.strip()

    # 2️⃣ 구조 파싱
    texts = []
    for block in _get(response, "output", []) or []:
        for item in _get(block, "content", []) or []:
            t = _get(item, "type", None)
            if t in ("output_text", "text"):
                text = _get(item, "text", "")
                if text:
                    texts.append(text)

    return "\n".join(texts).strip()
