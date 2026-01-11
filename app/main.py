from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import re
import traceback
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from db import documents, sentence_analysis
from ocr import run_ocr
from models import (
    DocumentCreateResponse,
    DocumentResponse,
    TextInput,
    SentencesResponse,
    SentenceItem,
    EvaluateResponse,
    IssueItem,
    # sentence analysis heuristics
    FACT_TRIGGERS,
    SOURCE_MARKERS,
    # evaluation heuristics
    MEDICAL_RISK_TERMS,
    CERT_TERMS,
    GUARANTEE_TERMS,
    OVERHYPE_TERMS,
    DISCLAIMERS,
)

load_dotenv()

app = FastAPI(title="Marketing Document Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 공통 유틸
# ---------------------------

def now_utc():
    return datetime.now(timezone.utc)


def split_sentences(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out: List[str] = []

    for ln in lines:
        parts = re.split(r"(?<=[\.\?\!])\s+", ln)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)

    return out


def analyze_sentence(s: str):
    """
    returns: (kind, verification_status, reason)
    kind: FACT_CLAIM | OPINION | OTHER
    verification_status: VERIFIED | UNVERIFIED | NOT_APPLICABLE
    """
    s = (s or "").strip()

    has_number = bool(re.search(r"\d", s))
    has_fact_kw = any(k in s for k in FACT_TRIGGERS)
    is_fact_claim = has_number or has_fact_kw

    is_opinion = bool(re.search(r"(최고|완벽|확실|대박|강력|추천|만족|혁신|프리미엄)", s))

    if is_fact_claim:
        kind = "FACT_CLAIM"
        has_source = any(m in s for m in SOURCE_MARKERS)
        has_id_like = bool(re.search(r"(NCT\d+|ISO\s?\d+|제?\s?\d{4,})", s))

        if has_source and has_id_like:
            return kind, "VERIFIED", None

        return kind, "UNVERIFIED", "검증 가능한 주장이나 출처/증빙이 문장에 포함되지 않음"

    if is_opinion:
        return "OPINION", "NOT_APPLICABLE", None

    return "OTHER", "NOT_APPLICABLE", None


def upsert_sentence_analysis(document_id: str, raw_text: str) -> List[dict]:
    """
    - sentence_analysis 컬렉션을 항상 최신으로 재생성
    - 반환: insert한 rows (dict list)
    """
    sents = split_sentences(raw_text)
    if not sents:
        raise HTTPException(status_code=400, detail="No sentences extracted")

    sentence_analysis.delete_many({"document_id": document_id})

    now = now_utc()
    docs_to_insert = []
    for idx, s in enumerate(sents):
        kind, vstat, reason = analyze_sentence(s)
        docs_to_insert.append(
            {
                "document_id": document_id,
                "idx": idx,
                "sentence": s,
                "kind": kind,
                "verification_status": vstat,
                "reason": reason,
                "created_at": now,
            }
        )

    sentence_analysis.insert_many(docs_to_insert)
    return docs_to_insert


# ---------------------------
# Health
# ---------------------------

@app.get("/")
def health():
    """헬스.체크.~^^    """
    return {"status": "ok"}


# ---------------------------
# Document Create
# ---------------------------

@app.post("/document/text", response_model=DocumentCreateResponse)
def create_document_text(payload: TextInput):
    """문서 입력 ( 텍스트 )"""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    doc_id = str(uuid4())
    documents.insert_one(
        {
            "_id": doc_id,
            "input_type": "text",
            "raw_text": payload.text,
            "status": "READY",
            "created_at": now_utc(),
        }
    )

    return DocumentCreateResponse(document_id=doc_id, status="READY")


@app.post("/document/image", response_model=DocumentCreateResponse)
async def create_document_image(file: UploadFile = File(...)):
    """이미지로 문서 입력 ( OCR 같이 )"""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    doc_id = str(uuid4())

    ocr_text = ""
    ocr_status = "FAIL"

    try:
        ocr_text = run_ocr(image_bytes, file.content_type)
        # ✅ 정상 로직: 텍스트 있으면 SUCCESS, 없으면 EMPTY
        ocr_status = "SUCCESS" if ocr_text.strip() else "EMPTY"
    except Exception as e:
        print("OCR ERROR:", type(e).__name__, str(e))
        traceback.print_exc()

    documents.insert_one(
        {
            "_id": doc_id,
            "input_type": "image",
            "raw_text": ocr_text,
            "image_filename": file.filename,
            "ocr_text": ocr_text,
            "ocr_engine": "gpt-4o-mini",
            "ocr_status": ocr_status,
            "status": "READY",
            "created_at": now_utc(),
        }
    )

    return DocumentCreateResponse(document_id=doc_id, status="READY")


# ---------------------------
# Document Read
# ---------------------------

@app.get("/document/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str):
    """설명서 조회"""
    doc = documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    doc["document_id"] = doc.pop("_id")
    return doc


# ---------------------------
# Sentences (store / read)
# ---------------------------

@app.post("/document/{document_id}/sentences/analyze", response_model=SentencesResponse)
def analyze_and_store_sentences(document_id: str):
    """문장 나눈걸로 설명서 평가"""
    doc = documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = (doc.get("raw_text") or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="No text to analyze (raw_text is empty)")

    rows = upsert_sentence_analysis(document_id, raw_text)

    items = [
        SentenceItem(
            idx=r["idx"],
            sentence=r["sentence"],
            kind=r["kind"],
            verification_status=r["verification_status"],
            reason=r.get("reason"),
        )
        for r in rows
    ]
    return SentencesResponse(document_id=document_id, sentences=items)


@app.get("/document/{document_id}/sentences", response_model=SentencesResponse)
def get_sentences(document_id: str):
    """문서를 문장 단위로 뜯고 진위여부 평가"""
    if not documents.find_one({"_id": document_id}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="Document not found")

    rows = list(sentence_analysis.find({"document_id": document_id}).sort("idx", 1))
    items = [
        SentenceItem(
            idx=r["idx"],
            sentence=r["sentence"],
            kind=r["kind"],
            verification_status=r["verification_status"],
            reason=r.get("reason"),
        )
        for r in rows
    ]
    return SentencesResponse(document_id=document_id, sentences=items)


# ---------------------------
# Evaluate (RAG optional)
# ---------------------------

def _contains_any(text: str, terms: list[str]) -> bool:
    return any(t in text for t in terms)


def _has_id_like(text: str) -> bool:
    return bool(
        re.search(
            r"(NCT\d+|ISO\s?\d+|제\s?\d{3,}|등록\s?번호|보고서\s?번호|시험\s?번호|특허\s?\d+)",
            text,
        )
    )


def compute_marketing_score(raw_text: str) -> int:
    score = 40

    has_benefit = bool(re.search(r"(효과|개선|완화|도움|해결|업그레이드|강화)", raw_text))
    has_feature = bool(re.search(r"(성분|함유|함량|기술|포뮬러|원료|특징)", raw_text))
    has_target = bool(re.search(r"(추천 대상|이런 분|누구|고민|피부|모발|체형|직장인|학생)", raw_text))
    has_usage = bool(re.search(r"(사용법|섭취법|방법|주의사항|권장)", raw_text))
    has_cta = bool(re.search(r"(지금|구매|문의|신청|한정|할인|이벤트)", raw_text))
    has_social_proof = bool(re.search(r"(후기|리뷰|만족|누적|판매|1위)", raw_text))
    has_structure = ("\n" in raw_text) or ("-" in raw_text) or ("•" in raw_text) or ("✅" in raw_text)

    score += 10 if has_benefit else 0
    score += 10 if has_feature else 0
    score += 10 if has_target else 0
    score += 10 if has_usage else 0
    score += 10 if has_cta else 0
    score += 5 if has_social_proof else 0
    score += 5 if has_structure else 0

    return max(0, min(100, score))


# rag.retrieve_evidence 가 있으면 근거 붙임 (없어도 평가 자체는 돌아감)
try:
    from rag import retrieve_evidence  # type: ignore
except Exception:
    retrieve_evidence = None  # type: ignore


def _evidence_types_for_issue(issue_type: str) -> Optional[list[str]]:
    if issue_type == "LEGAL_RISK":
        return ["policy", "legal"]
    if issue_type == "FACT_UNVERIFIED":
        return ["policy", "fact"]
    if issue_type == "TRUST_OVERCLAIM":
        return ["approved", "policy"]
    return None


def _attach_evidence(issue_type: str, sentence: str):
    if retrieve_evidence is None:
        return []
    try:
        types = _evidence_types_for_issue(issue_type)
        return retrieve_evidence(query=sentence, types=types, top_k=3)
    except Exception:
        return []


@app.post("/document/{document_id}/evaluate", response_model=EvaluateResponse)
def evaluate_document(document_id: str):
    """문장단위로 평가한 정보 종합한 평가 API"""
    doc = documents.find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = (doc.get("raw_text") or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="No text to evaluate (raw_text is empty)")

    rows = list(sentence_analysis.find({"document_id": document_id}).sort("idx", 1))
    if not rows:
        rows = upsert_sentence_analysis(document_id, raw_text)

    issues: list[IssueItem] = []

    for r in rows:
        idx = r["idx"]
        s = r["sentence"]
        kind = r.get("kind")
        vstat = r.get("verification_status")

        if _contains_any(s, MEDICAL_RISK_TERMS):
            issues.append(
                IssueItem(
                    idx=idx,
                    sentence=s,
                    issue_type="LEGAL_RISK",
                    severity="HIGH",
                    reason="의료/치료로 오인될 수 있는 표현이 포함됨",
                    suggestion="질병/치료 암시 표현 제거, 필요 시 고지문/표현 완화",
                    evidence=_attach_evidence("LEGAL_RISK", s),
                )
            )

        if _contains_any(s, GUARANTEE_TERMS):
            issues.append(
                IssueItem(
                    idx=idx,
                    sentence=s,
                    issue_type="LEGAL_RISK",
                    severity="MEDIUM",
                    reason="과도한 보장/확정 표현이 포함됨(절대/100%/즉시 등)",
                    suggestion="확정 표현을 완화하고 조건/범위/근거를 명시",
                    evidence=_attach_evidence("LEGAL_RISK", s),
                )
            )

        if _contains_any(s, CERT_TERMS) and not _has_id_like(s):
            issues.append(
                IssueItem(
                    idx=idx,
                    sentence=s,
                    issue_type="LEGAL_RISK",
                    severity="HIGH",
                    reason="임상/인증/승인 등 권위 표현이 있으나 식별자(번호/출처) 단서가 없음",
                    suggestion="시험/보고서/등록/특허 번호 또는 공식 출처를 명시하거나 표현 제거",
                    evidence=_attach_evidence("LEGAL_RISK", s),
                )
            )

        if kind == "FACT_CLAIM" and vstat == "UNVERIFIED":
            issues.append(
                IssueItem(
                    idx=idx,
                    sentence=s,
                    issue_type="FACT_UNVERIFIED",
                    severity="MEDIUM",
                    reason=r.get("reason") or "검증 가능한 주장이나 증빙 단서가 부족함",
                    suggestion="수치/시험조건/기관/기간/출처(링크·번호) 중 최소 1개를 문장에 포함",
                    evidence=_attach_evidence("FACT_UNVERIFIED", s),
                )
            )

        if _contains_any(s, OVERHYPE_TERMS):
            issues.append(
                IssueItem(
                    idx=idx,
                    sentence=s,
                    issue_type="TRUST_OVERCLAIM",
                    severity="LOW",
                    reason="과장/홍보성 표현이 포함되어 신뢰도가 떨어질 수 있음",
                    suggestion="정량/구체 근거 표현으로 대체하거나 톤 다운",
                    evidence=_attach_evidence("TRUST_OVERCLAIM", s),
                )
            )

    legal_score = 100
    for it in issues:
        if it.issue_type == "LEGAL_RISK":
            legal_score -= 25 if it.severity == "HIGH" else 12 if it.severity == "MEDIUM" else 6
    legal_score = max(0, min(100, legal_score))

    fact_claims = [r for r in rows if r.get("kind") == "FACT_CLAIM"]
    unverified = [r for r in fact_claims if r.get("verification_status") == "UNVERIFIED"]
    if not fact_claims:
        fact_score = 90
    else:
        ratio = len(unverified) / max(1, len(fact_claims))
        fact_score = int(100 - ratio * 60)
    fact_score = max(0, min(100, fact_score))

    marketing_score = compute_marketing_score(raw_text)

    trust_score = int((legal_score * 0.45) + (fact_score * 0.45) + 10)
    if not _contains_any(raw_text, DISCLAIMERS) and any(i.issue_type == "LEGAL_RISK" for i in issues):
        trust_score -= 10
    hype_cnt = sum(1 for i in issues if i.issue_type == "TRUST_OVERCLAIM")
    trust_score -= min(10, hype_cnt * 3)
    trust_score = max(0, min(100, trust_score))

    overall = int(legal_score * 0.35 + fact_score * 0.25 + trust_score * 0.25 + marketing_score * 0.15)
    overall = max(0, min(100, overall))

    high_legal = any(i.issue_type == "LEGAL_RISK" and i.severity == "HIGH" for i in issues)
    if high_legal or legal_score < 60:
        risk_level = "HIGH"
    elif legal_score < 80 or fact_score < 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return EvaluateResponse(
        document_id=document_id,
        overall=overall,
        overall_score=overall,  # 모델이 둘 중 뭐 쓰든 안전하게
        legal_score=legal_score,
        fact_score=fact_score,
        trust_score=trust_score,
        marketing_score=marketing_score,
        risk_level=risk_level,
        issues=issues,
    )
