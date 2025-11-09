"""
unified_name_extractor.py
-------------------------------------
Frame-based name extractor for Aadhaar / PAN / eAadhaar documents
Uses both EasyOCR and Tesseract for maximum accuracy.

Usage (from newmain.py):
    from unified_name_extractor import UnifiedNameExtractor

    extractor = UnifiedNameExtractor()
    first, full, debug = extractor.extract_name_from_image(frame)
"""

import cv2
import numpy as np
import re
from collections import defaultdict, Counter

# OCR imports
try:
    import easyocr
    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

# ------------------------------
# CONFIG / CONSTANTS
# ------------------------------
IGNORE_KEYWORDS = {
    "GOVERNMENT", "INDIA", "GOVT", "INCOME", "TAX", "PERMANENT", "ACCOUNT",
    "NUMBER", "PAN", "AADHAAR", "UIDAI", "AADHAR", "IDENTIFICATION", "AUTHORITY",
    "ENROLMENT", "ENROLLMENT", "DOB", "DATE", "BIRTH", "ADDRESS", "SIGNATURE",
    "MOBILE", "PHOTO", "CARD", "GENDER", "MALE", "FEMALE", "YEAR"
}

ADDRESS_HINTS = {"ROAD", "STREET", "NAGAR", "VILLAGE", "COLONY", "WARD", "POST", "DIST", "HOUSE", "PIN"}
AADHAAR_REL = ["S/O", "D/O", "W/O", "C/O", "S O", "D O"]
PAN_NAME_LABELS = ["NAME", "HOLDER'S NAME", "HOLDERS NAME"]
PAN_FATHER_LABELS = ["FATHER", "FATHER'S NAME", "FATHER NAME"]

MULTISPACE = re.compile(r"\s+")
ALPHA_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.'-]{0,40}$")
HAS_DIGIT = re.compile(r"\d")

# ------------------------------
# UTILS
# ------------------------------
def preprocess_for_ocr(frame):
    """Convert to grayscale and enhance for OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # adaptive threshold + contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=10)
    return gray

def image_variants(gray):
    """Generate enhanced versions for better OCR."""
    imgs = [("orig", gray)]
    # CLAHE
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imgs.append(("clahe", clahe.apply(gray)))
    except Exception:
        pass
    # Adaptive threshold
    try:
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 15, 3)
        imgs.append(("adaptive", thr))
    except Exception:
        pass
    # Sharpen
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    imgs.append(("sharpen", sharp))
    return imgs

def tesseract_text(img):
    """Run Tesseract OCR."""
    try:
        txt = pytesseract.image_to_string(img, lang="eng")
        return [MULTISPACE.sub(" ", l).strip() for l in txt.splitlines() if l.strip()]
    except Exception:
        return []

def easyocr_text(reader, img):
    """Run EasyOCR."""
    try:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = reader.readtext(bgr, detail=0, paragraph=False)
        return [MULTISPACE.sub(" ", s).strip() for s in out if s.strip()]
    except Exception:
        return []

# ------------------------------
# CLEANING & NAME FILTERING
# ------------------------------
def is_address_line(line):
    up = line.upper()
    if len(re.findall(r"\d", up)) >= 3:
        return True
    if any(h in up for h in ADDRESS_HINTS):
        return True
    if "," in line and len(line.split()) > 4:
        return True
    return False

def clean_line(line):
    """Clean OCR line, return as possible name candidate."""
    if not line or len(line.strip()) < 2:
        return None
    if is_address_line(line):
        return None

    tokens = [t.strip(" ,;:()[]\"") for t in line.split() if t.strip()]
    valid = []
    for t in tokens:
        if re.match(r"^[A-Za-z]\.?$", t):
            valid.append(t.upper()); continue
        if ALPHA_TOKEN.match(t):
            valid.append(t)
        elif re.search(r"[A-Za-z]", t) and not HAS_DIGIT.search(t):
            valid.append(t)
    if not valid or len(valid) > 6:
        return None

    def fix(t):
        return t.upper() if len(t) == 1 or t.endswith(".") else t.capitalize()
    name = " ".join(fix(v) for v in valid)
    if len(name) < 3:
        return None
    if any(kw in name.upper() for kw in IGNORE_KEYWORDS):
        return None
    return name

def gather_candidates(ocr_results):
    """Extract probable names with scoring."""
    candidates = []
    for res in ocr_results:
        lines = res["lines"]
        tag = res["variant"]
        for i, ln in enumerate(lines):
            lup = ln.upper()
            # Aadhaar before DOB/relation
            if any(r in lup for r in AADHAAR_REL) or "DOB" in lup:
                for j in range(max(0, i-3), i):
                    cand = clean_line(lines[j])
                    if cand:
                        candidates.append((cand, 250, "AADHAAR_BEFORE_REL", tag))
            # PAN after name label
            if any(lbl in lup for lbl in PAN_NAME_LABELS):
                for j in range(i+1, min(i+4, len(lines))):
                    if any(stop in lines[j].upper() for stop in PAN_FATHER_LABELS):
                        break
                    cand = clean_line(lines[j])
                    if cand:
                        candidates.append((cand, 200, "PAN_AFTER_LABEL", tag))
            # PAN before father label
            if any(lbl in lup for lbl in PAN_FATHER_LABELS):
                for j in range(max(0, i-3), i):
                    cand = clean_line(lines[j])
                    if cand:
                        candidates.append((cand, 180, "PAN_BEFORE_FATHER", tag))
            # Generic
            cand = clean_line(ln)
            if cand:
                wc = len(cand.split())
                candidates.append((cand, 100 + wc*8, "GENERIC", tag))
    return candidates

def select_best(candidates):
    if not candidates:
        return None, None, {}
    group = defaultdict(lambda: {"count":0,"score":0,"tags":Counter()})
    for c, s, st, tag in candidates:
        k = c.upper()
        group[k]["count"] += 1
        group[k]["score"] += s
        group[k]["tags"][st] += 1
    ranked = []
    for k, v in group.items():
        avg = v["score"]/v["count"]
        total = avg + v["count"]*10
        ranked.append((k, total, v))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top = ranked[0]
    full = top[0].title()
    first = full.split()[0].title() if full else None
    debug = {"picked":full,"score":float(top[1]),"occurrences":top[2]["count"],"tags":dict(top[2]["tags"])}
    return first, full, debug

# ------------------------------
# MAIN CLASS
# ------------------------------
class UnifiedNameExtractor:
    def __init__(self):
        self.reader = None
        if EASY_AVAILABLE:
            try:
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                print("[INFO] EasyOCR initialized")
            except Exception as e:
                print("[WARN] EasyOCR init failed:", e)
                self.reader = None
        if not EASY_AVAILABLE and not TESS_AVAILABLE:
            raise RuntimeError("No OCR engine available (install easyocr or pytesseract)")

    def extract_name_from_image(self, frame):
        """Takes OpenCV frame, returns (first_name, full_name, debug_info)."""
        gray = preprocess_for_ocr(frame)
        variants = image_variants(gray)
        results = []
        for tag, img in variants:
            lines = []
            if self.reader:
                lines += easyocr_text(self.reader, img)
            if TESS_AVAILABLE:
                lines += tesseract_text(img)
            clean = []
            seen = set()
            for l in lines:
                if l and l not in seen:
                    seen.add(l)
                    clean.append(l)
            if clean:
                results.append({"variant":tag, "lines":clean})
        # collect candidates
        candidates = gather_candidates(results)
        return select_best(candidates)
