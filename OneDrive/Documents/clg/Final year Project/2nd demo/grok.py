#!/usr/bin/env python3
"""
Drishya AI — Refined single-file voice-activated continuous camera registration system.
Refinements:
- beep + "Give your response after the beep" -> listens once -> registers only if user says "register this person"
- During registration: beep -> listen once for name -> "Shall I remember you as {name}? Please confirm." -> listen once.
- If confirm -> "Ok, I will remember you after your verification." -> accepts ONLY Aadhaar or PAN -> OCR first valid read used.
- Captures 120 frames (best frames used), saves photo and encoding when card name matches spoken name.
- Continuous overlay: red box + "Unknown" and green box + person name.
- Continuous camera display.

Dependencies:
pip install opencv-python face-recognition pyttsx3 SpeechRecognition pyaudio easyocr rapidfuzz numpy Pillow simpleaudio
(If pyaudio install fails on Linux: sudo apt-get install portaudio19-dev then pip install pyaudio)

Run:
python drishya_ai_refined_final.py
"""

import os
import sys
import time
import json
import re
import threading
from queue import Queue
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

# Optional libs
try:
    import face_recognition
except Exception:
    print("ERROR: face_recognition is required. pip install face-recognition")
    raise

try:
    import pyttsx3
except Exception:
    print("ERROR: pyttsx3 is required. pip install pyttsx3")
    raise

try:
    import speech_recognition as sr
except Exception:
    print("ERROR: SpeechRecognition (and pyaudio) required. pip install SpeechRecognition pyaudio")
    raise

try:
    import easyocr
    OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
except Exception:
    OCR_READER = None
    print("Warning: easyocr not available. Card OCR will be limited. pip install easyocr")

try:
    from rapidfuzz import fuzz
    FUZZY = True
except Exception:
    FUZZY = False

# try to import simpleaudio for cross platform beep fallback
try:
    import simpleaudio as sa
    SIMPLEAUDIO = True
except:
    SIMPLEAUDIO = False

# ---------------- CONFIG ----------------
JSON_FILE = "faces.json"
MIN_CAPTURE_FRAMES = 120
BEST_FRAME_COUNT = 20
FACE_TOLERANCE = 0.45
GREETING_COOLDOWN = 10.0
SCENARIO_ANNOUNCE_INTERVAL = 6.0
CARD_CAPTURE_MAX_SECONDS = 30
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------- GLOBALS ----------------
engine_lock = threading.Lock()
camera_lock = threading.Lock()
data_lock = threading.Lock()

cap = None
active = True

known_encodings = []
known_names = []
known_meta = {}

last_greet_time = {}
last_scenario_time = 0

command_queue = Queue()

# ---------------- UTILS ----------------
def speak(text, block=True):
    print("TTS:", text)
    def _tts():
        try:
            with engine_lock:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print("TTS error:", e)
    t = threading.Thread(target=_tts, daemon=True)
    t.start()
    if block:
        t.join(timeout=5)

def beep(duration_ms=200, freq=1000):
    """Cross-platform beep fallback: winsound -> simpleaudio generated tone -> system bell."""
    try:
        import winsound
        winsound.Beep(freq, duration_ms)
        return
    except Exception:
        pass
    if SIMPLEAUDIO:
        try:
            fs = 44100
            t = np.linspace(0, duration_ms/1000.0, int(fs * (duration_ms/1000.0)), False)
            tone = np.sin(freq * 2 * np.pi * t)
            audio = (tone * 32767).astype(np.int16)
            try:
                play_obj = sa.play_buffer(audio, 1, 2, fs)
                play_obj.wait_done()
                return
            except Exception:
                pass
        except Exception:
            pass
    # fallback bell
    sys.stdout.write('\a'); sys.stdout.flush()

def normalize_name(s):
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r'[^A-Z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def names_match(a, b, threshold=80):
    na = normalize_name(a)
    nb = normalize_name(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    if FUZZY:
        score = fuzz.token_sort_ratio(na, nb)
        return score >= threshold
    return na.split()[0] == nb.split()[0]

# ---------------- DATABASE ----------------
def load_faces():
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
            json.dump([], f)
        return [], [], {}
    
    try:
        if os.path.getsize(JSON_FILE) == 0:
            return [], [], {}
        
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        
        names = []
        encodings = []
        verification_map = {}
        
        for entry in data:
            name = entry.get('name', '')
            if not name:
                continue
                
            names.append(name)
            encodings.append(np.array(entry['encoding']))
            
            card_type = entry.get('card_type')
            full_name = entry.get('full_name')
            is_verified = bool(card_type and full_name)
            verification_map[name] = is_verified
        
        print(f"[DATABASE] Loaded {len(names)} faces, {sum(verification_map.values())} verified")
        return encodings, names, verification_map
        
    except Exception as e:
        print(f"[LOAD ERROR]: {e}")
        return [], [], {}

def save_face(name, encoding, card_type, full_name, photo_path):
    try:
        data = []
        if os.path.exists(JSON_FILE) and os.path.getsize(JSON_FILE) > 0:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
        
        data.append({
            'name': name,
            'encoding': encoding.tolist(),
            'card_type': card_type,
            'full_name': full_name,
            'timestamp': datetime.now().isoformat(),
            'verified': True,
            'photo': photo_path
        })
        
        temp_file = JSON_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        os.replace(temp_file, JSON_FILE)
        print(f"[SAVE SUCCESS] {name} - Verified with {card_type}")
        return True
    except Exception as e:
        print(f"[SAVE ERROR]: {e}")
        return False

# ---------------- FACE HELPERS ----------------
def detect_faces_and_encodings(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)
        locs_small = face_recognition.face_locations(small, model='hog')
        locs = [(t*2, r*2, b*2, l*2) for (t,r,b,l) in locs_small]
        encs = []
        if locs:
            encs = face_recognition.face_encodings(rgb, locs)
        return locs, encs
    except Exception:
        return [], []

def match_encoding(enc):
    if not known_encodings:
        return None
    dists = face_recognition.face_distance(known_encodings, enc)
    idx = int(np.argmin(dists))
    if dists[idx] <= FACE_TOLERANCE:
        return idx
    return None

def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ---------------- ID VERIFIER ----------------
class IDVerifier:
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[ID VERIFY] EasyOCR initialized")
        except Exception as e:
            print(f"[ID VERIFY INIT ERROR]: {e}")
            raise
        
        self.verified_cards = []
    
    def calculate_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def preprocess_for_mobile_display(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(binary)
        contrast = cv2.convertScaleAbs(denoised, alpha=1.8, beta=20)
        return contrast
    
    def detect_screen_glare(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        glare_ratio = bright_pixels / total_pixels
        return glare_ratio > 0.15
    
    def enhance_mobile_id(self, image):
        kernel = np.ones((2,2), np.uint8)
        denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
        
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def capture_id_card(self, cap, card_prompt="Please show your ID card."):
        speak(card_prompt, block=True)
        speak("You can show physical card, or display it on your mobile phone screen, or show a printed document.", block=True)
        time.sleep(1)
        
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < CARD_CAPTURE_MAX_SECONDS and len(frames) < MIN_CAPTURE_FRAMES:
            ret, frame = cap.read()
            if not ret:
                continue
            
            sharpness = self.calculate_sharpness(frame)
            has_glare = self.detect_screen_glare(frame)
            
            if sharpness > 80 or (sharpness > 50 and not has_glare):
                frames.append((frame.copy(), sharpness))
                print(f"[ID CAPTURE] Frame {len(frames)}/{MIN_CAPTURE_FRAMES} (sharpness={sharpness:.1f})")
            
            time.sleep(0.25)
        
        if not frames:
            return None
        
        best_frame = max(frames, key=lambda x: x[1])[0]
        print(f"[ID CAPTURE] Selected best frame")
        return best_frame
    
    def extract_text_multiple_methods(self, image):
        all_text = []
        
        try:
            results = self.reader.readtext(image, paragraph=False)
            text1 = " ".join([r[1] for r in results])
            all_text.append(text1)
            print(f"[OCR-Original]: {text1[:150]}")
        except Exception as e:
            print(f"[OCR-Original Error]: {e}")
        
        try:
            enhanced = self.enhance_mobile_id(image)
            results = self.reader.readtext(enhanced, paragraph=False)
            text2 = " ".join([r[1] for r in results])
            all_text.append(text2)
            print(f"[OCR-Enhanced]: {text2[:150]}")
        except Exception as e:
            print(f"[OCR-Enhanced Error]: {e}")
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
            results = self.reader.readtext(contrast, paragraph=False)
            text4 = " ".join([r[1] for r in results])
            all_text.append(text4)
            print(f"[OCR-Contrast]: {text4[:150]}")
        except Exception as e:
            print(f"[OCR-Contrast Error]: {e}")
        
        combined_text = " ".join(all_text).upper()
        return combined_text
    
    def detect_card_type(self, image):
        try:
            text = self.extract_text_multiple_methods(image)
            print(f"[FULL OCR TEXT]: {text[:500]}")
            
            aadhaar_score = 0
            pan_score = 0
            
            aadhaar_critical = ["UIDAI", "UNIQUE IDENTIFICATION AUTHORITY OF INDIA"]
            aadhaar_strong = ["AADHAAR", "AADHAR", "ADHAAR"]
            aadhaar_medium = ["GOVERNMENT OF INDIA", "GOI", "UNIQUE IDENTIFICATION", 
                             "ENROLLMENT", "ENROLMENT"]
            
            pan_critical = ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER"]
            pan_strong = ["INCOME TAX", "PAN CARD", "PANCARD"]
            pan_medium = ["PERMANENT ACCOUNT", "GOVT OF INDIA", "FATHER NAME", 
                         "FATHERS NAME", "SIGNATURE"]
            
            invalid_strong = ["DRIVING LICENCE", "DRIVING LICENSE", "DL NO", "TRANSPORT",
                             "PASSPORT", "VOTER", "ELECTION COMMISSION", "RATION CARD"]
            
            for keyword in invalid_strong:
                if keyword in text:
                    print(f"[INVALID CARD] Detected: {keyword}")
                    return "Invalid"
            
            for keyword in aadhaar_critical:
                if keyword in text:
                    aadhaar_score += 20
            
            for keyword in aadhaar_strong:
                if keyword in text:
                    aadhaar_score += 12
            
            for keyword in aadhaar_medium:
                if keyword in text:
                    aadhaar_score += 5
            
            for keyword in pan_critical:
                if keyword in text:
                    pan_score += 20
            
            for keyword in pan_strong:
                if keyword in text:
                    pan_score += 12
            
            for keyword in pan_medium:
                if keyword in text:
                    pan_score += 5
            
            if re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text):
                pan_score += 18
            
            if re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text):
                aadhaar_score += 18
            
            print(f"[FINAL SCORE] Aadhaar={aadhaar_score}, PAN={pan_score}")
            
            if aadhaar_score >= 15 and aadhaar_score > pan_score + 5:
                return "Aadhaar"
            elif pan_score >= 15 and pan_score > aadhaar_score + 5:
                return "PAN"
            elif aadhaar_score > pan_score and aadhaar_score >= 10:
                return "Aadhaar"
            elif pan_score > aadhaar_score and pan_score >= 10:
                return "PAN"
            else:
                return "Unknown"
                
        except Exception as e:
            print(f"[CARD DETECT ERROR]: {e}")
            return "Unknown"
    
    def extract_first_name(self, image):
        first_name, full_name, debug = extractor.extract_name_from_image(image)
        
        if first_name and full_name:
            print("\n" + "=" * 60)
            print(f"✅ NAME EXTRACTED SUCCESSFULLY")
            print(f"   First Name: {first_name}")
            print(f"   Full Name:  {full_name}")
            print("=" * 60 + "\n")
            
            return first_name, full_name
        else:
            print("[NAME EXTRACTION] Failed")
            return None, None
    
    def verify_with_spoken_name(self, cap, spoken_name):
        speak(f"Hello {spoken_name}. Let's verify your identity.", block=True)
        
        max_attempts = 1
        
        for attempt in range(max_attempts):
            if attempt > 0:
                speak(f"Attempt {attempt + 1}. Please show your ID card again.", block=True)
            
            id_image = self.capture_id_card(cap)
            if id_image is None:
                speak("Could not capture image. Try again.", block=True)
                continue
            
            card_type = self.detect_card_type(id_image)
            
            if card_type == "Invalid":
                speak("Invalid ID. Please show Aadhaar or PAN.", block=True)
                continue
            elif card_type == "Unknown":
                speak("Could not identify card. Show more clearly.", block=True)
                continue
            
            speak(f"{card_type} card detected. Extracting name.", block=True)
            
            first_name, full_name = self.extract_first_name(id_image)
            
            if not first_name:
                speak("Could not extract name. Try again.", block=True)
                continue
            
            if names_match(spoken_name, first_name):
                print(f"[VERIFICATION SUCCESS] '{spoken_name}' matches '{first_name}'")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                id_path = os.path.join("verified_ids", f"{first_name}_{card_type}_{timestamp}.jpg")
                cv2.imwrite(id_path, id_image)
                
                speak(f"Perfect! Your name matches. Verification completed.", block=True)
                return card_type, first_name, full_name
            else:
                print(f"[VERIFICATION FAILED] '{spoken_name}' != '{first_name}'")
                speak(f"Name mismatch. You said {spoken_name}, but ID shows {first_name}.", block=True)
        
        speak("Verification failed. Name does not match ID card.", block=True)
        return None, None, None

# ---------------- NAME EXTRACTOR ----------------
class UnifiedNameExtractor:
    def __init__(self):
        self.reader = None
        if 'easyocr' in globals():
            try:
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                print("[INFO] EasyOCR initialized")
            except Exception as e:
                print("[WARN] EasyOCR init failed:", e)
                self.reader = None

    def extract_name_from_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=10)
        
        imgs = [("orig", gray)]
        
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            imgs.append(("clahe", clahe.apply(gray)))
        except:
            pass
        
        try:
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 3)
            imgs.append(("adaptive", thr))
        except:
            pass
        
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        imgs.append(("sharpen", sharp))
        
        results = []
        for tag, img in imgs:
            lines = []
            if self.reader:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                out = self.reader.readtext(bgr, detail=0, paragraph=False)
                lines += [re.sub(r"\s+", " ", s).strip() for s in out if s.strip()]
            
            clean = []
            seen = set()
            for l in lines:
                if l and l not in seen:
                    seen.add(l)
                    clean.append(l)
            if clean:
                results.append({"variant":tag, "lines":clean})
        
        candidates = []
        for res in results:
            lines = res["lines"]
            tag = res["variant"]
            for i, ln in enumerate(lines):
                lup = ln.upper()
                if any(r in lup for r in ["S/O", "D/O", "W/O", "C/O", "S O", "D O"]) or "DOB" in lup:
                    for j in range(max(0, i-3), i):
                        cand = self.clean_line(lines[j])
                        if cand:
                            candidates.append((cand, 250, "AADHAAR_BEFORE_REL", tag))
                if any(lbl in lup for lbl in ["NAME", "HOLDER'S NAME", "HOLDERS NAME"]):
                    for j in range(i+1, min(i+4, len(lines))):
                        if any(stop in lines[j].upper() for stop in ["FATHER", "FATHER'S NAME", "FATHER NAME"]):
                            break
                        cand = self.clean_line(lines[j])
                        if cand:
                            candidates.append((cand, 200, "PAN_AFTER_LABEL", tag))
                if any(lbl in lup for lbl in ["FATHER", "FATHER'S NAME", "FATHER NAME"]):
                    for j in range(max(0, i-3), i):
                        cand = self.clean_line(lines[j])
                        if cand:
                            candidates.append((cand, 180, "PAN_BEFORE_FATHER", tag))
                cand = self.clean_line(ln)
                if cand:
                    wc = len(cand.split())
                    candidates.append((cand, 100 + wc*8, "GENERIC", tag))
        
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
    
    def clean_line(self, line):
        if not line or len(line.strip()) < 2:
            return None
        if len(re.findall(r"\d", line.upper())) >= 3:
            return None
        if any(h in line.upper() for h in ["ROAD", "STREET", "NAGAR", "VILLAGE", "COLONY", "WARD", "POST", "DIST", "HOUSE", "PIN"]):
            return None
        if "," in line and len(line.split()) > 4:
            return None

        tokens = [t.strip(" ,;:()[]\"") for t in line.split() if t.strip()]
        valid = []
        for t in tokens:
            if re.match(r"^[A-Za-z]\.?$", t):
                valid.append(t.upper()); continue
            if re.match(r"^[A-Za-z][A-Za-z.'-]{0,40}$", t):
                valid.append(t)
            elif re.search(r"[A-Za-z]", t) and not re.search(r"\d", t):
                valid.append(t)
        if not valid or len(valid) > 6:
            return None

        def fix(t):
            return t.upper() if len(t) == 1 or t.endswith(".") else t.capitalize()
        name = " ".join(fix(v) for v in valid)
        if len(name) < 3:
            return None
        if any(kw in name.upper() for kw in ["GOVERNMENT", "INDIA", "GOVT", "INCOME", "TAX", "PERMANENT", "ACCOUNT",
    "NUMBER", "PAN", "AADHAAR", "UIDAI", "AADHAR", "IDENTIFICATION", "AUTHORITY",
    "ENROLMENT", "ENROLLMENT", "DOB", "DATE", "BIRTH", "ADDRESS", "SIGNATURE",
    "MOBILE", "PHOTO", "CARD", "GENDER", "MALE", "FEMALE", "YEAR"]):
            return None
        return name

# ---------------- Background Listener ----------------
class BackgroundListener(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.microphone = None
        try:
            self.microphone = sr.Microphone()
        except Exception as e:
            print("Microphone init error:", e)
            raise
        self.running = True

    def run(self):
        speak("Voice listener started in background.", block=False)
        while self.running and active:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=6)
                try:
                    text = self.recognizer.recognize_google(audio, language='en-IN')
                except:
                    try:
                        text = self.recognizer.recognize_google(audio, language='en-US')
                    except:
                        text = ""
                if text:
                    txt = text.strip().lower()
                    print("Voice heard (bg):", txt)
                    if "register this person" in txt and "not" not in txt:
                        command_queue.put(("register", txt))
                    print(f"Processed command: {txt} within 2 seconds")
            except Exception as e:
                print("Background listen error:", e)

# ---------------- Single Listen ----------------
def single_listen(timeout=5, phrase_time_limit=4):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[LISTENING]")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        
        try:
            text = recognizer.recognize_google(audio, language='en-IN')
            print(f"[GOOGLE en-IN]: '{text}'")
            return text.strip()
        except:
            try:
                text = recognizer.recognize_google(audio, language='en-US')
                print(f"[GOOGLE en-US]: '{text}'")
                return text.strip()
            except:
                pass
    except sr.WaitTimeoutError:
        print("[TIMEOUT]")
    except Exception as e:
        print(f"[LISTEN ERROR]: {e}")
    
    return ""

# ---------------- Registration Flow ----------------
def registration_flow(frame=None):
    global known_encodings, known_names, known_meta
    
    speak("Beginning registration.", block=True)
    
    # 1) Listen for name
    speak("Please introduce yourself with your name. Give your response after the beep.", block=True)
    beep()
    name_heard = single_listen(timeout=5, phrase_time_limit=4)
    
    if not name_heard:
        speak("I did not hear your name. Registration cancelled.", block=True)
        return False
    
    name_cap = " ".join(w.capitalize() for w in name_heard.split())
    
    # 2) Confirm name
    speak(f"Shall I remember you as {name_cap}? Please confirm.", block=True)
    beep()
    confirm_heard = single_listen(timeout=5, phrase_time_limit=4)
    
    if not confirm_heard:
        speak("No confirmation. Registration cancelled.", block=True)
        return False
    
    confirm_lower = confirm_heard.lower()
    if "yes" in confirm_lower or name_cap.lower() in confirm_lower:
        final_name = name_cap
    else:
        final_name = " ".join(w.capitalize() for w in confirm_heard.split())
    
    speak("Ok. I will remember you after your verification.", block=True)

    # Verify with ID
    verifier = IDVerifier()
    card_type, extracted_first, extracted_full = verifier.verify_with_spoken_name(cap, final_name)
    
    if not card_type:
        speak("ID verification failed. Registration cancelled.", block=True)
        return False
    
    # Capture frames for face
    speak(f"Look at camera. Capturing {MIN_CAPTURE_FRAMES} frames for best quality.", block=True)
    frames = []
    encodings = []
    startcap = time.time()
    while len(frames) < MIN_CAPTURE_FRAMES and (time.time() - startcap) < 60:
        with camera_lock:
            if cap is None:
                break
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue
        locs, encs = detect_faces_and_encodings(frame)
        if encs:
            frames.append((frame.copy(), sharpness_score(frame)))
            encodings.append(encs[0])
        time.sleep(0.02)
    
    if not encodings:
        speak("Could not capture face clearly. Registration cancelled.", block=True)
        return False
    
    frames_sorted = sorted(frames, key=lambda x: x[1], reverse=True)[:BEST_FRAME_COUNT]
    best_images = [f[0] for f in frames_sorted]
    try:
        avg_enc = np.mean(encodings, axis=0)
    except:
        avg_enc = encodings[0]

    photos_dir = "registered_photos"
    os.makedirs(photos_dir, exist_ok=True)
    photo_name = f"{final_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    photo_path = os.path.join(photos_dir, photo_name)
    cv2.imwrite(photo_path, best_images[0])

    # Save
    saved = save_face(final_name, avg_enc, card_type, extracted_full, photo_path)
    if saved:
        speak(f"The registration of the person is completed! Welcome {final_name}. Verified with {card_type}.", block=True)
        return True
    else:
        speak("Failed to save registration. Please try again later.", block=True)
        return False

# ---------------- Overlay and announce ----------------
def draw_overlay(frame, locs, names_list):
    disp = frame.copy()
    for (t,r,b,l), nm in zip(locs, names_list):
        color = (0,255,0) if nm != "Unknown" else (0,0,255)
        cv2.rectangle(disp, (l,t), (r,b), color, 2)
        label = f"{nm}" if nm != "Unknown" else "Unknown"
        cv2.putText(disp, label, (l, b+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return disp

def announce_counts(known_count, unknown_count):
    global last_scenario_time
    now = time.time()
    if now - last_scenario_time < SCENARIO_ANNOUNCE_INTERVAL:
        return
    last_scenario_time = now
    total = known_count + unknown_count
    if total == 0:
        speak("No persons detected in front of you.", block=False)
        return
    # Grammar handling for singular/plural
    if unknown_count > 0 and known_count == 0:
        if unknown_count == 1:
            speak("There is one unknown person in front of you.", block=True)
        else:
            speak(f"There are {unknown_count} unknown persons in front of you.", block=True)
        speak("Give your response after the beep.", block=True)
        beep()
        # Listen once after beep for user response
        try:
            with sr.Microphone() as source:
                r = sr.Recognizer()
                r.adjust_for_ambient_noise(source, duration=0.7)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    text = r.recognize_google(audio, language='en-IN').lower()
                except:
                    text = r.recognize_google(audio, language='en-US').lower()
                print("User said:", text)
                if "register this person" in text and "not" not in text:
                    speak("Registration command detected.", block=False)
                    command_queue.put(("register", text))
                else:
                    speak("Continuing monitoring.", block=False)
        except Exception as e:
            print("Listening error after beep:", e)
            speak("I did not catch any response.", block=False)
    elif known_count > 0 and unknown_count == 0:
        if known_count == 1:
            speak("There is one known person in front of you.", block=False)
        else:
            speak(f"There are {known_count} known persons in front of you.", block=False)
    else:
        speak(f"There are {total} persons: {known_count} known and {unknown_count} unknown.", block=False)

# ---------------- Main loop ----------------
def main_loop():
    global cap, active
    last_known = -1
    last_unknown = -1
    while active:
        with camera_lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.2)
                continue
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        locs, encs = detect_faces_and_encodings(frame)
        names_list = []
        known_count = 0
        unknown_count = 0
        for loc, enc in zip(locs, encs):
            idx = match_encoding(enc)
            if idx is not None:
                name = known_names[idx]
                names_list.append(name)
                known_count += 1
                now = time.time()
                last = last_greet_time.get(name, 0)
                if now - last > GREETING_COOLDOWN:
                    last_greet_time[name] = now
                    speak(f"Welcome back {name}.", block=False)
            else:
                names_list.append("Unknown")
                unknown_count += 1

        if known_count != last_known or unknown_count != last_unknown:
            announce_counts(known_count, unknown_count)
            last_known = known_count
            last_unknown = unknown_count

        out = draw_overlay(frame, locs, names_list)
        cv2.imshow("Drishya AI", out)

        try:
            cmd, raw = command_queue.get_nowait()
            if cmd == "register":
                speak("Register command received. Preparing to register the person now.", block=False)
                registration_flow()
                known_encodings, known_names, known_meta = load_faces()
        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("User requested quit. Shutting down.", block=True)
            active = False
            break
        time.sleep(0.01)

    with camera_lock:
        if cap:
            cap.release()
    cv2.destroyAllWindows()

# ==================== FACE DETECTION ====================
def detect_faces(frame):
    """Detect faces"""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        locs = face_recognition.face_locations(small, model="hog")
        locs = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locs]
        encs = face_recognition.face_encodings(rgb, locs)
        return locs, encs
    except:
        return [], []

def match_face(enc):
    """Match face"""
    if not known_encodings:
        return None
    dists = face_recognition.face_distance(known_encodings, enc)
    idx = np.argmin(dists)
    return idx if dists[idx] < FACE_TOL else None

def calculate_sharpness(image):
    """Calculate sharpness"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0

# ==================== NAME MATCHING ====================
def normalize(name):
    """Normalize name"""
    if not name:
        return ""
    return ' '.join(re.sub(r'[^a-z\s]', '', name.lower()).split())

def names_match(n1, n2):
    """Check if names match"""
    norm1, norm2 = normalize(n1), normalize(n2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2 or norm1 in norm2 or norm2 in norm1:
        return True
    if FUZZY:
        try:
            return fuzz.ratio(norm1, norm2) >= 70
        except:
            pass
    return False

# ==================== OCR ====================
def extract_card_info(frame):
    """Extract card info"""
    if not OCR:
        return None, None, False, "OCR not available"
    
    try:
        sharpness = calculate_sharpness(frame)
        if sharpness < 100:
            return None, None, True, "Blurry"
        
        results = OCR.readtext(frame, paragraph=False)
        if not results:
            return None, None, False, "No text"
        
        text = " ".join([r[1] for r in results]).upper()
        
        card_type = None
        if any(k in text for k in ["AADHAAR", "AADHAR", "UID"]):
            card_type = "AADHAAR"
        elif any(k in text for k in ["INCOME TAX", "PAN"]):
            card_type = "PAN"
        
        if not card_type:
            if any(k in text for k in ["VOTER", "ELECTION", "DRIVING", "LICENSE"]):
                return None, None, False, "Not accepted"
            return None, None, False, "Not recognized"
        
        # Extract name
        lines = text.split('\n')
        for line in lines:
            words = line.strip().split()
            if 2 <= len(words) <= 4 and not any(c.isdigit() for c in line):
                clean = re.sub(r'[^A-Z\s]', '', line.strip())
                if 5 <= len(clean) <= 50:
                    return clean.title(), card_type, False, "Success"
        
        return None, card_type, False, "Name unclear"
    except:
        return None, None, False, "Error"
    
    # ==================== REGISTRATION ====================
def do_registration():
    """Registration process"""
    global is_registering, registration_in_progress, scenario_announced
    
    if is_registering or registration_in_progress:
        return
    
    is_registering = True
    registration_in_progress = True
    scenario_announced = True
    
    try:
        speak("Starting registration process.")
        time.sleep(0.5)
        
        # Get name
        speak("Please introduce yourself. Tell me your full name after the beep.")
        beep()
        time.sleep(0.5)
        
        name_heard = mic.listen_once_ultra_accurate(timeout=20)
        
        if not name_heard:
            speak("I did not hear your name. Registration cancelled.")
            return
        
        name_heard = ' '.join(word.capitalize() for word in name_heard.split())
        
        speak(f"I heard your name as {name_heard}.")
        time.sleep(0.5)
        speak("Is this correct? Say yes to confirm or repeat your name after the beep.")
        beep()
        time.sleep(0.5)
        
        confirmation = mic.listen_once_ultra_accurate(timeout=20)
        
        if not confirmation:
            speak("No confirmation. Registration cancelled.")
            return
        
        conf_lower = confirmation.lower()
        yes_words = ['yes', 'yeah', 'yep', 'correct', 'right', 'ok', 'okay', 'sure', 'yah', 'ya', 'yup']
        
        if any(word in conf_lower for word in yes_words):
            print("✅ Name confirmed")
            speak(f"Name confirmed as {name_heard}.")
        else:
            print(f"🔄 Name repeated: {confirmation}")
            name_heard = ' '.join(word.capitalize() for word in confirmation.split())
            speak(f"Using {name_heard} as your name.")
        
        time.sleep(0.5)
        
        # Capture face
        speak("Now I will capture your face.")
        time.sleep(0.3)
        speak("Please look at the camera and stay still.")
        time.sleep(0.5)
        speak(f"Capturing {MIN_FRAMES_CAPTURE} frames for best quality.")
        time.sleep(1)
        speak("Starting capture. Please remain still.")
        time.sleep(0.5)
        
        frames_data = []
        frame_count = 0
        start_time = time.time()
        
        while frame_count < MIN_FRAMES_CAPTURE and (time.time() - start_time) < 30:
            with cam_lock:
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        locs, encs = detect_faces(frame)
                        if locs and encs:
                            sharpness = calculate_sharpness(frame)
                            frames_data.append({
                                'frame': frame.copy(),
                                'encoding': encs[0],
                                'sharpness': sharpness
                            })
                            frame_count += 1
                            
                            if frame_count % 30 == 0:
                                print(f"📸 {frame_count} frames captured")
            
            time.sleep(0.05)
        
        if len(frames_data) < MIN_FRAMES_CAPTURE:
            speak(f"Only captured {len(frames_data)} frames. Please try again.")
            return
        
        speak(f"Successfully captured {len(frames_data)} frames.")
        time.sleep(0.5)
        speak("Selecting best quality image.")
        
        best_frame_data = max(frames_data, key=lambda x: x['sharpness'])
        best_frame = best_frame_data['frame']
        
        print(f"✅ Best sharpness: {best_frame_data['sharpness']:.2f}")
        
        # Save photo
        photos_dir = "registered_photos"
        os.makedirs(photos_dir, exist_ok=True)
        photo_filename = f"{name_heard.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        photo_path = os.path.join(photos_dir, photo_filename)
        cv2.imwrite(photo_path, best_frame)
        
        speak("Best quality frame saved.")
        
        # Average encoding
        all_encodings = [fd['encoding'] for fd in frames_data]
        avg_encoding = np.mean(all_encodings, axis=0)
        
        time.sleep(0.5)
        
        # ID Card
        speak("Now I need to verify your identity with your ID card.")
        time.sleep(0.3)
        speak("Please show your Aadhaar or PAN card to the camera.")
        time.sleep(0.3)
        speak("Note: Only Aadhaar or PAN cards are accepted.")
        time.sleep(0.5)
        speak("Hold the card clearly with good lighting.")
        time.sleep(1)
        speak("Starting card capture. Hold steady.")
        time.sleep(0.5)
        
        card_name = None
        card_type = None
        attempts = 0
        max_attempts = 50
        last_feedback = 0
        
        while attempts < max_attempts:
            with cam_lock:
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        name_extracted, type_extracted, is_blurry, message = extract_card_info(frame)
                        
                        if is_blurry:
                            if (time.time() - last_feedback) > 3:
                                speak("Card is blurry. Please hold it steady.")
                                last_feedback = time.time()
                        elif type_extracted and not name_extracted:
                            if (time.time() - last_feedback) > 3:
                                speak(f"{type_extracted} detected but name not clear.")
                                last_feedback = time.time()
                        elif name_extracted and type_extracted:
                            card_name = name_extracted
                            card_type = type_extracted
                            speak(f"{card_type} card detected successfully.")
                            break
                        elif "not accepted" in message.lower():
                            if (time.time() - last_feedback) > 3:
                                speak("This card type is not accepted. Please show Aadhaar or PAN card only.")
                                last_feedback = time.time()
            
            attempts += 1
            time.sleep(0.2)
        
        if not card_name or not card_type:
            speak("Could not read card clearly. Please try again with better lighting.")
            return
        
        time.sleep(0.5)
        speak(f"Card read successfully. Card type is {card_type}.")
        time.sleep(0.5)
        speak(f"Name on card is {card_name}.")
        time.sleep(0.5)
        
        # Verify match
        if not names_match(name_heard, card_name):
            speak(f"Name mismatch. You said {name_heard} but card shows {card_name}.")
            speak("Registration cancelled.")
            return
        
        speak("Name verified successfully. Names match perfectly.")
        beep()
        time.sleep(0.5)
        
        # Save
        speak("Saving your information to the database.")
        
        if save_face(name_heard, avg_encoding, card_type, card_name, photo_path):
            load_db()
            speak(f"Registration completed successfully for {name_heard}.")
            beep()
            time.sleep(0.5)
            speak(f"Your {card_type} card has been verified and stored securely.")
            time.sleep(0.5)
            speak("Your face has been registered with best quality photo.")
            time.sleep(0.5)
            speak(f"{name_heard} is now fully verified and authenticated.")
            beep()
            time.sleep(0.5)
            speak("Registration complete. Welcome to the system.")
        else:
            speak("Failed to save. Please try again.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        speak("An error occurred. Please try again.")
    
    finally:
        is_registering = False
        registration_in_progress = False
        scenario_announced = False

# ==================== DISPLAY ====================
def show_frame(frame, locs, names):
    """Display frame"""
    try:
        disp = frame.copy()
        
        for (t, r, b, l), name in zip(locs, names):
            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(disp, (l, t), (r, b), color, 3)
            
            lbl = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
            cv2.rectangle(disp, (l, b), (l + lbl[0] + 20, b + lbl[1] + 15), color, -1)
            cv2.putText(disp, name, (l + 10, b + lbl[1] + 8), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Drishya AI', disp)
        cv2.waitKey(1)
    except:
        pass
# ---------------- STARTUP ----------------
def start_system():
    global cap
    known_encodings, known_names, known_meta = load_faces()
    speak("Drishya AI system activated.", block=False)
    time.sleep(0.2)
    speak("Opening camera for continuous monitoring.", block=False)

    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG, cv2.CAP_ANY]:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap and cap.isOpened():
                break
        except:
            cap = None
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap or not cap.isOpened():
        speak("Cannot open camera. Exiting.", block=True)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    listener = BackgroundListener()
    listener.start()

    time.sleep(0.5)
    cv2.namedWindow("Drishya AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drishya AI", 800, 600)

    speak("System is now active and monitoring continuously. Say register this person to begin registration.", block=False)
    try:
        main_loop()
    except KeyboardInterrupt:
        speak("Keyboard interrupt received. Shutting down.", block=True)
    listener.running = False
    speak("System shutting down. Goodbye.", block=False)

if __name__ == "__main__":
    extractor = UnifiedNameExtractor()
    start_system()