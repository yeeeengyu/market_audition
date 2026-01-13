from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Document(BaseModel):
    input_type: str            # "text" | "image"
    raw_text: str

    image_filename: Optional[str] = None

    ocr_text: Optional[str] = None
    ocr_engine: Optional[str] = None
    ocr_status: Optional[str] = None

    status: str                # READY
    created_at: datetime

class DocumentCreateResponse(BaseModel):
    document_id: str
    status: str

class DocumentResponse(BaseModel):
    document_id: str
    input_type: str
    raw_text: str
    status: str
    created_at: datetime

    image_filename: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_engine: Optional[str] = None
    ocr_status: Optional[str] = None

class TextInput(BaseModel):
    text: str

class SentenceItem(BaseModel):
    idx: int
    sentence: str
    kind: str  # FACT_CLAIM | OPINION | OTHER
    verification_status: str  # VERIFIED | UNVERIFIED | NOT_APPLICABLE
    reason: Optional[str] = None

class SentencesResponse(BaseModel):
    document_id: str
    sentences: List[SentenceItem]

class EvidenceItem(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    snippet: str
    score: float

class IssueItem(BaseModel):
    idx: int
    sentence: str
    issue_type: str        # LEGAL_RISK | FACT_UNVERIFIED | TRUST_OVERCLAIM | MARKETING_GAP
    severity: str          # LOW | MEDIUM | HIGH
    reason: str
    suggestion: Optional[str] = None
    evidence: Optional[List[EvidenceItem]] = None

from typing import Dict
class EvaluateResponse(BaseModel):
    document_id: str
    scores: Dict[str, int]     # {"legal":..,"marketing":..,"fact":..,"trust":..,"overall":..}
    risk_level: str            # LOW | MEDIUM | HIGH
    summary: str
    issues: List[IssueItem]

class EmbedRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = None


# 문장분해에 사용됨
FACT_TRIGGERS = [
    "임상", "인증", "허가", "승인", "특허", "검증", "입증", "테스트", "시험",
    "효과", "성분", "함유", "함량", "개선", "치료", "완화"
]
SOURCE_MARKERS = ["식약처", "FDA", "ISO", "NCT", "특허", "KC", "CE"]


# 평가에 사용됨
MEDICAL_RISK_TERMS = [
    "치료", "완치", "부작용 없음", "부작용없음", "약", "의약품", "항암", "염증", "통증",
    "질병", "병원", "처방", "진단", "효능", "효과 확실", "즉시 효과"
]

CERT_TERMS = ["식약처", "FDA", "임상", "승인", "인증", "특허"]
GUARANTEE_TERMS = ["100%", "무조건", "절대", "확실", "완벽", "즉시", "영구", "단기간", "바로"]
OVERHYPE_TERMS = ["최고", "역대급", "대박", "미쳤", "압도", "완벽", "프리미엄", "강력 추천"]

DISCLAIMERS = ["개인차", "의약품이 아닙니다", "질병의 예방", "치료를 위한", "효과는 개인", "사용 전"]