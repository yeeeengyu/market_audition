from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentCreateResponse(BaseModel):
    document_id: str
    status: str


class Document(BaseModel):
    input_type: str            # "text" | "image"
    raw_text: str

    image_filename: Optional[str] = None

    ocr_text: Optional[str] = None
    ocr_engine: Optional[str] = None
    ocr_status: Optional[str] = None

    status: str                # READY
    created_at: datetime
