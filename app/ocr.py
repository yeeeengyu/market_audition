import base64
import os
from openai import OpenAI

print("OPENAI_API_KEY IN OCR:", repr(os.getenv("OPENAI_API_KEY")))

def _get(obj, key, default=None):
    # dict면 key로, 객체면 attribute로
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

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
