from pathlib import Path
from typing import Dict, List, Tuple

from app.config import settings
from app.services.chat_nlp import predict_chat_risk
from app.services.rag_index import search

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def retrieve(query: str, top_k: int = 3) -> List[Tuple[str, str]]:
    vector_hits = search(query, top_k=top_k)
    if vector_hits:
        return [(src, txt) for src, _, txt in vector_hits]

    if not DOCS_DIR.exists():
        return []

    results: List[Tuple[str, int, str]] = []
    q = query.lower()
    for path in DOCS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        score = text.lower().count(q) if q else 0
        results.append((path.name, score, text))

    results.sort(key=lambda x: x[1], reverse=True)
    return [(name, text) for name, _, text in results[:top_k]]


def generate_answer(query: str, risk_drivers: List[str]) -> Tuple[str, List[str]]:
    vector_hits = search(query, top_k=3)
    if vector_hits:
        docs = [(name, text) for name, _, text in vector_hits]
        sources = [f"{name} (score={score:.3f})" for name, score, _ in vector_hits]
    else:
        docs = retrieve(query)
        sources = [name for name, _ in docs]

    if not docs:
        return (
            "No guideline documents found. Please add trusted clinical guidance to the docs folder.",
            [],
        )

    snippet = docs[0][1][:300].strip().replace("\n", " ")
    drivers = ", ".join(risk_drivers) if risk_drivers else "the reported symptoms"
    answer = (
        f"Based on the available guidelines and {drivers}, consider the following: "
        f"{snippet}"
    )

    return answer, sources


RISK_KEYWORDS: Dict[str, Tuple[List[str], float]] = {
    "sleep_disturbance": (
        ["cannot sleep", "insomnia", "no sleep", "sleep deprived", "sleeping badly"],
        0.17,
    ),
    "low_mood": (
        ["sad", "hopeless", "empty", "crying", "depressed", "down all day"],
        0.2,
    ),
    "anxiety": (
        ["anxious", "panic", "constant worry", "overthinking", "afraid"],
        0.12,
    ),
    "fatigue": (
        ["exhausted", "tired all day", "no energy", "drained"],
        0.1,
    ),
    "appetite_change": (
        ["no appetite", "overeating", "not eating", "lost appetite"],
        0.08,
    ),
    "bonding_difficulty": (
        ["cannot bond", "detached from baby", "not connected to baby"],
        0.18,
    ),
    "guilt_or_worthlessness": (
        ["bad mother", "guilty", "worthless", "failure as a mother"],
        0.15,
    ),
    "financial_stress": (
        ["money stress", "financial stress", "cannot afford", "debt"],
        0.07,
    ),
}

RISK_KEYWORDS_MULTILINGUAL: Dict[str, List[str]] = {
    "sleep_disturbance": ["neend nahi", "neend kam", "sone mein dikkat", "so nahi pa rahi"],
    "low_mood": ["udaas", "dukhi", "mann kharab", "bechain aur udaasi"],
    "anxiety": ["chinta", "ghabrahat", "tension", "dar lagta hai"],
    "fatigue": ["thakan", "bohot thak gayi", "energy nahi"],
    "appetite_change": ["bhook nahi", "khana nahi kha pa rahi", "zyada khana"],
    "guilt_or_worthlessness": ["main buri maa hoon", "main bekaar hoon", "guilt"],
    "self_harm_ideation": ["khud ko nuksan", "mar jana", "jeene ka mann nahi"],
}

PROTECTIVE_KEYWORDS: Dict[str, List[str]] = {
    "social_support": ["family supports me", "partner supports me", "help at home"],
    "care_engagement": ["seeing therapist", "talking to doctor", "counseling"],
    "coping_behaviors": ["walking daily", "breathing exercises", "journaling"],
}

PROTECTIVE_KEYWORDS_MULTILINGUAL: Dict[str, List[str]] = {
    "social_support": ["parivar support", "ghar wale saath", "partner support"],
    "care_engagement": ["doctor se baat", "therapy", "counseling"],
    "coping_behaviors": ["walk karti hoon", "saans ki exercise", "dhyan"],
}

CRISIS_KEYWORDS = [
    "hurt myself",
    "harm myself",
    "kill myself",
    "suicidal",
    "better off dead",
    "end my life",
]

CRISIS_KEYWORDS_MULTILINGUAL = [
    "khud ko nuksan",
    "marna chahti hoon",
    "jeene ka mann nahi",
    "apne aap ko chot",
]

POSTPARTUM_KEYWORDS = [
    "gave birth",
    "after birth",
    "postpartum",
    "after delivery",
    "newborn",
    "my baby is",
    "weeks after birth",
]

ANTENATAL_KEYWORDS = [
    "pregnant",
    "pregnancy",
    "gestational",
    "trimester",
]


def _collect_hits(text: str, keyword_map: Dict[str, List[str]]) -> List[str]:
    hits: List[str] = []
    for label, phrases in keyword_map.items():
        if any(phrase in text for phrase in phrases):
            hits.append(label)
    return hits


def _collect_weighted_risk_hits(text: str) -> Tuple[List[str], float]:
    factors: List[str] = []
    score = 0.0
    for label, (phrases, weight) in RISK_KEYWORDS.items():
        if any(phrase in text for phrase in phrases):
            factors.append(label)
            score += weight
    return factors, score


def assess_chat_message(message: str) -> Dict[str, object]:
    text = message.lower()
    risk_factors, base_score = _collect_weighted_risk_hits(text)
    protective_factors = _collect_hits(text, PROTECTIVE_KEYWORDS)

    if settings.enable_multilingual_chat:
        for factor, phrases in RISK_KEYWORDS_MULTILINGUAL.items():
            if any(phrase in text for phrase in phrases) and factor not in risk_factors:
                risk_factors.append(factor)
                base_score += 0.1
        for factor, phrases in PROTECTIVE_KEYWORDS_MULTILINGUAL.items():
            if any(phrase in text for phrase in phrases) and factor not in protective_factors:
                protective_factors.append(factor)

    crisis_mode = any(term in text for term in CRISIS_KEYWORDS)
    if settings.enable_multilingual_chat:
        crisis_mode = crisis_mode or any(term in text for term in CRISIS_KEYWORDS_MULTILINGUAL)
    if crisis_mode:
        base_score = 1.0
        if "self_harm_ideation" not in risk_factors:
            risk_factors.append("self_harm_ideation")

    base_score -= 0.08 * len(protective_factors)
    risk_percent = round(max(0.0, min(100.0, base_score * 100.0)), 1)

    if risk_percent >= 75:
        risk_level = "high"
    elif risk_percent >= 45:
        risk_level = "moderate"
    else:
        risk_level = "low"

    postpartum_hits = sum(1 for kw in POSTPARTUM_KEYWORDS if kw in text)
    antenatal_hits = sum(1 for kw in ANTENATAL_KEYWORDS if kw in text)
    if postpartum_hits > antenatal_hits:
        likely_context = "postpartum"
    elif antenatal_hits > postpartum_hits:
        likely_context = "antenatal"
    else:
        likely_context = "unclear"

    language = "non-english" if any(ord(ch) > 127 for ch in message) else "english_or_romanized"

    model_used = "rules_only"
    ml_pred = predict_chat_risk(message)
    if ml_pred is not None:
        ml_risk_percent, ml_risk_level, ml_factors, ml_model_used = ml_pred
        model_used = f"hybrid_{ml_model_used}"
        if not crisis_mode:
            risk_percent = round((risk_percent * 0.5) + (ml_risk_percent * 0.5), 1)
            if risk_percent >= 75:
                risk_level = "high"
            elif risk_percent >= 45:
                risk_level = "moderate"
            else:
                risk_level = "low"
        for factor in ml_factors:
            if factor not in risk_factors:
                risk_factors.append(factor)

    if crisis_mode:
        risk_percent = 100.0
        risk_level = "high"

    return {
        "risk_percent": risk_percent,
        "risk_level": risk_level,
        "likely_context": likely_context,
        "language": language,
        "model_used": model_used,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "crisis_mode": crisis_mode,
    }


def generate_chat_guidance(
    message: str, risk_factors: List[str], likely_context: str, crisis_mode: bool
) -> Tuple[str, List[str]]:
    if crisis_mode:
        return (
            "Crisis indicators were detected. Immediate emergency and clinician support is required. "
            "Call local emergency services now and contact a trusted support person immediately. "
            "Action window: now (0-1 hour). Psychiatric referral: immediate.",
            [],
        )

    query = (
        f"{likely_context} maternal mental health guidance for "
        f"{', '.join(risk_factors) if risk_factors else 'mood symptoms'}"
    )
    answer, sources = generate_answer(query, risk_factors)
    followups = []
    if "sleep_disturbance" in risk_factors:
        followups.append("Sleep hygiene intervention: start today, reassess in 72 hours.")
    if "low_mood" in risk_factors or "anxiety" in risk_factors:
        followups.append("Psychological screening follow-up: schedule within 7 days.")
    if "financial_stress" in risk_factors:
        followups.append("Social worker referral: within 14 days.")
    if not followups:
        followups.append("Routine follow-up check: within 14 days.")

    actionable = " Actionable next steps: " + " ".join(followups)
    return answer + actionable, sources


def get_source_content(source_name: str) -> Dict[str, str] | None:
    safe_name = Path(source_name).name
    path = DOCS_DIR / safe_name
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return {"name": safe_name, "content": text}
