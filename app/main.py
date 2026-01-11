from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from uuid import uuid4

from db import documents
from ocr import run_ocr
from models import DocumentCreateResponse

app = FastAPI()


# ---------- 텍스트 입력 ----------

class TextInput(BaseModel):
    text: str

@app.get('/')
def health():
    return {"status": "ok"}

@app.post("/documents/text", response_model=DocumentCreateResponse)
def create_document_text(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    doc_id = str(uuid4())

    documents.insert_one({
        "_id": doc_id,
        "input_type": "text",
        "raw_text": payload.text,
        "status": "READY",
        "created_at": datetime.now(timezone.utc)
    })

    return DocumentCreateResponse(
        document_id=doc_id,
        status="READY"
    )


# ---------- 이미지 입력 + OCR ----------

@app.post("/documents/image", response_model=DocumentCreateResponse)
async def create_document_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    doc_id = str(uuid4())

    ocr_text = ""
    ocr_status = "FAIL"

    try:
        ocr_text = run_ocr(image_bytes, file.content_type)
        ocr_status = "EMPTY" if ocr_text.strip() else "SUCCESS"   # ⭐ 호출만 성공하면 SUCCES

    except Exception as e:
        import traceback
        print("OCR ERROR TYPE:", type(e))
        print("OCR ERROR:", e)
        traceback.print_exc()


    print("OCR LEN:", len(ocr_text))
    print("OCR TEXT:", repr(ocr_text))

    documents.insert_one({
        "_id": doc_id,
        "input_type": "image",
        "raw_text": ocr_text,          # ⭐ 평가 기준은 raw_text
        "image_filename": file.filename,
        "ocr_text": ocr_text,
        "ocr_engine": "gpt-4o-mini",
        "ocr_status": ocr_status,
        "status": "READY",
        "created_at": datetime.now(timezone.utc)
    })

    return DocumentCreateResponse(
        document_id=doc_id,
        status="READY"
    )

@app.get("/documents/{document_id}")
def get_document(document_id: str):
    doc = documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    doc["document_id"] = doc.pop("_id")
    return doc