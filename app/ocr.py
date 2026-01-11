import base64
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_ocr(image_bytes: bytes, content_type: str) -> str:
    image_b64 = base64.b64encode(image_bytes).decode()
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
                            "이미지에 있는 텍스트만 추출해라.\n"
                            "- 추측 금지\n"
                            "- 보이는 순서 유지\n"
                            "- 마크다운 불필요\n"
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

    return response.output_text or ""
