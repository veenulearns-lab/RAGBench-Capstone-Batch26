"""Scoring functions with NFKC normalisation and all-parts integration scoring.
Extracted from RGB_Evaluation_v2_CP4_Batch26.ipynb, cell 8.
"""

# CELL 8: RGB Scoring Functions — v2 FIXES:
#   (1) info integration now requires ALL answer parts (paper definition) — v1 any-match inflated scores
#   (2) counterfactual adds gold_present (for Acc_doc) alongside detection/correction
#   (3) CR reported both ways in the summary: paper-style (among detected) and overall
import unicodedata

def _norm(text):
    """NFKC-fold Unicode (non-breaking/narrow spaces -> plain space etc.), lowercase, collapse whitespace.
    gpt-oss models emit U+00A0/U+202F inside dates and times, which silently breaks ASCII substring matching."""
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.lower().split())

REJECTION_PHRASE = "i can not answer the question because of the insufficient information in documents"
DETECTION_PHRASE = "there are factual errors in the provided documents"


def _is_error(response):
    return response.startswith("[ERROR")


def score_accuracy_any(response, answer_raw):
    # single-answer testbeds (noise robustness, counterfactual): any variant of the answer counts
    if _is_error(response):
        return 0
    resp = _norm(response)
    flat = []
    for item in answer_raw if isinstance(answer_raw, list) else [answer_raw]:
        if isinstance(item, list):
            flat.extend(_norm(a) for a in item)
        else:
            flat.append(_norm(item))
    return int(any(a in resp for a in flat))


def score_accuracy_all_parts(response, answer_raw):
    # info integration (official checkanswer logic): EVERY part must be present.
    # Each part may itself be a list of acceptable variants (any variant satisfies that part).
    if _is_error(response):
        return 0
    resp = _norm(response)
    parts = answer_raw if isinstance(answer_raw, list) else [answer_raw]
    for part in parts:
        if isinstance(part, list):
            if not any(_norm(v) in resp for v in part):
                return 0
        else:
            if _norm(part) not in resp:
                return 0
    return 1


def score_rejection(response):
    if _is_error(response):
        return 0
    return int(REJECTION_PHRASE in _norm(response))


def score_error_detection(response):
    if _is_error(response):
        return 0
    return int(DETECTION_PHRASE in _norm(response))


def score_error_correction(response, answer_raw):
    # correction = detected AND true answer present (harness definition; summary also
    # reports paper-style CR = corrected / detected)
    if _is_error(response):
        return 0
    return int(score_error_detection(response) and score_accuracy_any(response, answer_raw))


print("Scoring functions ready (all-parts integration + Acc_doc support).")
