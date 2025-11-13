#!/usr/bin/env python3
"""
================================================================================
DRISHYA AI - COMPLETE SYSTEM FOR BLIND USERS
================================================================================

Run Instructions:
1. Install dependencies:
   pip install opencv-python numpy face_recognition dlib deepface pyttsx3 google-cloud-speech speechrecognition pyaudio easyocr pytesseract webrtcvad noisereduce rapidfuzz jellyfish boto3 pillow tkinter vosk
   Note: For dlib if issues: install cmake, then pip install dlib
   For Tesseract: install system package (e.g., sudo apt install tesseract-ocr)
   For Vosk: download model from https://alphacephei.com/vosk/models and set path in code
2. Set environment variables:
   - GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_credentials.json
   - AWS_ACCESS_KEY_ID=your_key
   - AWS_SECRET_ACCESS_KEY=your_secret
3. Configure in code: S3_BUCKET, VOSK_MODEL_PATH, etc.
4. Run: python main.py
   - For help: python main.py --help

Privacy Warning: This system handles PII (faces, names, cards). Use local-only mode if no cloud credentials provided.
================================================================================
"""

import os
import sys
import time
import json
import re
import threading
from queue import Queue
from datetime import datetime
from collections import defaultdict, Counter
import signal
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

# Configuration
S3_BUCKET = ""  # Set to your bucket or leave empty to disable sync
VOSK_MODEL_PATH = "vosk-model-small-en-in-0.4"  # Download from https://alphacephei.com/vosk/models
TTS_VOICE_RATE = 150
STT_TIMEOUT_SECONDS = 7
FACE_MATCH_TOLERANCE = 0.38
AUDIO_DEVICE_INDEX = None  # None for default
NAME_MATCH_SIMILARITY = 85
NAME_EDIT_DISTANCE = 0.20
MIN_FRAMES_CAPTURE = 60
CARD_CAPTURE_FRAMES = 120
BEST_FRAMES_SELECT = 20
GREETING_COOLDOWN = 10.0
SCENARIO_DEBOUNCE = 2.0
JSON_FILE = "faces.json"

# Imports with fallbacks
FACE_LIB = None
try:
    import face_recognition
    FACE_LIB = "face_recognition"
except ImportError:
    print("face_recognition not available, falling back")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

TTS_LIB = None
try:
    import pyttsx3
    TTS_LIB = "pyttsx3"
except ImportError:
    print("pyttsx3 not available")

GOOGLE_STT_AVAILABLE = 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ
if GOOGLE_STT_AVAILABLE:
    from google.cloud import speech_v1p1beta1 as speech

try:
    import vosk
    VOSK_AVAILABLE = os.path.exists(VOSK_MODEL_PATH)
except ImportError:
    VOSK_AVAILABLE = False

try:
    import speech_recognition as sr
    STT_FALLBACK = "sr"
except ImportError:
    STT_FALLBACK = None

try:
    import webrtcvad
    import noisereduce as nr
    NOISE_SUPPRESS = True
except ImportError:
    NOISE_SUPPRESS = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESS_AVAILABLE = True
except ImportError:
    TESS_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

if not (FACE_LIB or DEEPFACE_AVAILABLE):
    print("No face detection library available")
    sys.exit(1)

# Globals
known_encodings = []
known_names = []
known_cards = {}
known_photos = {}

cap = None
active = True
is_registering = False
last_greet = {}
last_scenario_state = (0, 0)
last_scenario_time = 0

command_queue = Queue()
tts_queue = Queue()
status_text = ""

root = None
video_label = None
status_label = None

face_lock = threading.Lock()
cam_lock = threading.Lock()
tts_lock = threading.Lock()

haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Unified Name Extractor (advanced from claude.py, but adapted for single attempt with multi-variants)
IGNORE_KEYWORDS = {
    "GOVERNMENT", "INDIA", "GOVT", "INCOME", "TAX", "PERMANENT", "ACCOUNT",
    "NUMBER", "PAN", "AADHAAR", "UIDAI", "AADHAR", "IDENTIFICATION", "AUTHORITY",
    "ENROLMENT", "ENROLLMENT", "DOB", "DATE", "BIRTH", "ADDRESS", "SIGNATURE",
    "MOBILE", "PHOTO", "CARD", "GENDER", "MALE", "FEMALE", "YEAR", "TO", "THE"
}

ADDRESS_HINTS = {"ROAD", "STREET", "NAGAR", "VILLAGE", "COLONY", "WARD", "POST", "DIST", "HOUSE", "PIN"}
AADHAAR_REL = ["S/O", "D/O", "W/O", "C/O", "S O", "D O"]
PAN_NAME_LABELS = ["NAME", "HOLDER'S NAME", "HOLDERS NAME", "NAM"]
PAN_FATHER_LABELS = ["FATHER", "FATHER'S NAME", "FATHER NAME"]

MULTISPACE = re.compile(r"\s+")
ALPHA_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.'-]{0,40}$")
HAS_DIGIT = re.compile(r"\d")

def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=10)
    return gray

def image_variants(gray):
    imgs = [("orig", gray)]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    imgs.append(("clahe", clahe.apply(gray)))
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    imgs.append(("adaptive", thr))
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    imgs.append(("sharpen", sharp))
    high_contrast = cv2.convertScaleAbs(gray, alpha=2.2, beta=30)
    imgs.append(("high_contrast", high_contrast))
    return imgs

def ocr_text(img, reader=None):
    lines = []
    if EASYOCR_AVAILABLE and reader:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = reader.readtext(bgr, detail=0, paragraph=False)
        lines += [MULTISPACE.sub(" ", s).strip() for s in out if s.strip()]
    if TESS_AVAILABLE:
        txt = pytesseract.image_to_string(img, lang="eng")
        lines += [MULTISPACE.sub(" ", l).strip() for l in txt.splitlines() if l.strip()]
    seen = set()
    clean = [l for l in lines if l and l not in seen and not seen.add(l)]
    return clean

def clean_line(line):
    if not line or len(line.strip()) < 2 or is_address_line(line):
        return None
    tokens = [t.strip(" ,;:()[]\"") for t in line.split() if t.strip()]
    valid = []
    for t in tokens:
        if re.match(r"^[A-Za-z]\.?$", t):
            valid.append(t.upper())
        elif ALPHA_TOKEN.match(t):
            valid.append(t)
        elif re.search(r"[A-Za-z]", t) and not HAS_DIGIT.search(t):
            valid.append(t)
    if not valid or len(valid) > 6:
        return None
    fix = lambda t: t.upper() if len(t) == 1 or t.endswith(".") else t.capitalize()
    name = " ".join(fix(v) for v in valid)
    if len(name) < 3 or any(kw in name.upper() for kw in IGNORE_KEYWORDS):
        return None
    return name

def gather_candidates(ocr_results):
    candidates = []
    for res in ocr_results:
        lines = res["lines"]
        tag = res["variant"]
        for i, ln in enumerate(lines):
            lup = ln.upper()
            if lup.strip() in ['TO', 'ಗೆ']:
                for j in range(i+1, min(i+4, len(lines))):
                    cand_line = lines[j]
                    if not is_address_line(cand_line) and not any(x in cand_line.upper() for x in ['D/O', 'S/O', 'W/O', 'C/O']):
                        cand = clean_line(cand_line)
                        if cand and len(cand.split()) >= 2:
                            candidates.append((cand, 250, "AADHAAR_AFTER_TO", tag))
            if any(r in lup for r in AADHAAR_REL) or "DOB" in lup:
                for j in range(max(0, i-3), i):
                    cand = clean_line(lines[j])
                    if cand and len(cand.split()) >= 2:
                        candidates.append((cand, 240, "AADHAAR_BEFORE_REL", tag))
            if any(lbl in lup for lbl in PAN_NAME_LABELS):
                for j in range(i+1, min(i+4, len(lines))):
                    if any(stop in lines[j].upper() for stop in PAN_FATHER_LABELS):
                        break
                    cand = clean_line(lines[j])
                    if cand and len(cand.split()) >= 2:
                        candidates.append((cand, 230, "PAN_AFTER_LABEL", tag))
            if any(lbl in lup for lbl in PAN_FATHER_LABELS):
                for j in range(max(0, i-3), i):
                    cand = clean_line(lines[j])
                    if cand and len(cand.split()) >= 2:
                        candidates.append((cand, 220, "PAN_BEFORE_FATHER", tag))
            cand = clean_line(ln)
            if cand and len(cand.split()) >= 2:
                wc = len(cand.split())
                candidates.append((cand, 100 + wc*8, "GENERIC", tag))
    return candidates

def select_best(candidates):
    if not candidates:
        return None, None
    group = defaultdict(lambda: {"count":0,"score":0,"tags":Counter()})
    for c, s, st, tag in candidates:
        k = c.upper()
        group[k]["count"] += 1
        group[k]["score"] += s
        group[k]["tags"][st] += 1
    ranked = sorted(group.items(), key=lambda x: x[1]["score"]/x[1]["count"] + x[1]["count"]*10, reverse=True)
    top = ranked[0]
    full = top[0].title()
    first = full.split()[0].title() if full else None
    return first, full

def extract_name_from_card(frame):
    if not (EASYOCR_AVAILABLE or TESS_AVAILABLE):
        return None, None
    reader = easyocr.Reader(['en'], gpu=False, verbose=False) if EASYOCR_AVAILABLE else None
    gray = preprocess_for_ocr(frame)
    variants = image_variants(gray)
    results = []
    for tag, img in variants:
        lines = ocr_text(img, reader)
        if lines:
            results.append({"variant": tag, "lines": lines})
    candidates = gather_candidates(results)
    return select_best(candidates)

# Audio Preprocessing
def process_audio(audio_data, sample_rate=16000):
    if NOISE_SUPPRESS:
        vad = webrtcvad.Vad(3)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        reduced = nr.reduce_noise(y=audio_np, sr=sample_rate)
        return reduced.astype(np.int16).tobytes()
    return audio_data

# STT Functions
def google_stt(audio_data):
    if not GOOGLE_STT_AVAILABLE:
        return ""
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-IN",
    )
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        return result.alternatives[0].transcript.strip()
    return ""

def vosk_stt(audio_data):
    if not VOSK_AVAILABLE:
        return ""
    model = vosk.Model(VOSK_MODEL_PATH)
    rec = vosk.KaldiRecognizer(model, 16000)
    if rec.AcceptWaveform(audio_data):
        result = json.loads(rec.Result())
        return result.get("text", "").strip()
    return ""

def fallback_stt(audio_data):
    if STT_FALLBACK:
        r = sr.Recognizer()
        audio = sr.AudioData(audio_data, 16000, 2)
        try:
            return r.recognize_sphinx(audio, language="en-IN")
        except:
            return ""
    return ""

def listen_once(timeout=STT_TIMEOUT_SECONDS, phrase_time_limit=None):
    r = sr.Recognizer()
    with sr.Microphone(device_index=AUDIO_DEVICE_INDEX, sample_rate=16000) as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    raw_data = audio.get_raw_data()
    processed = process_audio(raw_data)
    text = google_stt(processed)
    if not text and VOSK_AVAILABLE:
        text = vosk_stt(processed)
    if not text and STT_FALLBACK:
        text = fallback_stt(processed)
    return text.lower().strip() if text else ""

# TTS
def tts_worker():
    while active:
        text = tts_queue.get()
        if text:
            with tts_lock:
                if TTS_LIB == "pyttsx3":
                    engine = pyttsx3.init()
                    engine.setProperty('rate', TTS_VOICE_RATE)
                    engine.say(text)
                    engine.runAndWait()

def speak(text, update_gui=True):
    print(f"🔊 {text}")
    tts_queue.put(text)
    if update_gui and root:
        status_text = text
        status_label.config(text=status_text)

# Face Detection and Matching
def detect_faces(frame):
    locs = []
    encs = []
    if FACE_LIB == "face_recognition":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
    elif DEEPFACE_AVAILABLE:
        try:
            faces = DeepFace.extract_faces(frame, enforce_detection=False)
            locs = [(f['facial_area']['y'], f['facial_area']['x'] + f['facial_area']['w'], f['facial_area']['y'] + f['facial_area']['h'], f['facial_area']['x']) for f in faces]
            encs = [DeepFace.represent(f['face'], model_name='Facenet', enforce_detection=False)[0]['embedding'] for f in faces]
        except:
            pass
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray, 1.3, 5)
        locs = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if DEEPFACE_AVAILABLE:
                enc = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
                encs.append(enc)
            else:
                encs.append(None)
    return locs, encs

def match_face(enc):
    if not known_encodings or enc is None:
        return None
    if FACE_LIB == "face_recognition":
        dists = face_recognition.face_distance(known_encodings, enc)
    elif DEEPFACE_AVAILABLE:
        dists = [np.linalg.norm(np.array(ke) - np.array(enc)) for ke in known_encodings]
    else:
        return None
    min_dist = min(dists)
    if min_dist < FACE_MATCH_TOLERANCE:
        return dists.index(min_dist)
    return None

def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def names_match(spoken, card):
    norm_spoken = re.sub(r'[^a-z\s]', '', spoken.lower()).strip()
    norm_card = re.sub(r'[^a-z\s]', '', card.lower()).strip()
    if not norm_spoken or not norm_card:
        return False
    if norm_spoken in norm_card or norm_card in norm_spoken:
        return True
    if RAPIDFUZZ_AVAILABLE:
        if fuzz.ratio(norm_spoken, norm_card) >= NAME_MATCH_SIMILARITY:
            return True
    if JELLYFISH_AVAILABLE:
        if jellyfish.metaphone(norm_spoken) == jellyfish.metaphone(norm_card):
            return True
        dist = jellyfish.levenshtein_distance(norm_spoken, norm_card) / max(len(norm_spoken), len(norm_card))
        if dist <= NAME_EDIT_DISTANCE:
            return True
    return False

# Database
def load_db():
    global known_encodings, known_names, known_cards, known_photos
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
            json.dump([], f)
        return
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    known_encodings = [np.array(d['encoding']) for d in data]
    known_names = [d['name'] for d in data]
    known_cards = {d['name']: d['card_name'] for d in data}
    known_photos = {d['name']: d['photo'] for d in data}

def save_db(name, encoding, card_name, photo_path):
    data = json.load(open(JSON_FILE)) if os.path.exists(JSON_FILE) else []
    data.append({
        'name': name,
        'encoding': encoding.tolist(),
        'card_name': card_name,
        'photo': photo_path
    })
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f)

# AWS Sync
def aws_sync_thread():
    if not AWS_AVAILABLE or not S3_BUCKET:
        return
    while active:
        time.sleep(60)
        try:
            s3 = boto3.client('s3')
            # Upload
            s3.upload_file(JSON_FILE, S3_BUCKET, JSON_FILE)
            speak("Uploaded database to cloud")
            # Download for merge
            s3.download_file(S3_BUCKET, JSON_FILE, 'temp.json')
            remote = json.load(open('temp.json'))
            local = json.load(open(JSON_FILE))
            # Merge, avoid duplicates by name
            merged = {d['name']: d for d in local}
            for r in remote:
                if r['name'] not in merged:
                    merged[r['name']] = r
            with open(JSON_FILE, 'w') as f:
                json.dump(list(merged.values()), f)
            speak("Synchronized with cloud")
            os.remove('temp.json')
        except (NoCredentialsError, ClientError):
            print("AWS error")
            break

# Registration
def registration():
    global is_registering
    is_registering = True
    speak("Okay, I will begin with registration.")
    speak("Please tell your name after the beep.")
    beep()
    name = listen_once(phrase_time_limit=5)
    if not name:
        speak("I could not understand clearly, registration cancelled.")
        is_registering = False
        return
    speak(f"I heard {name}. Please confirm your name again after the beep.")
    beep()
    confirm = listen_once(phrase_time_limit=5)
    if not names_match(name, confirm):
        speak("Name mismatch detected. Please try again.")
        is_registering = False
        return
    spoken_name = name
    speak("Okay, let’s verify your identity. Please show your Aadhaar card and PAN card.")
    # Capture card frames for single attempt extraction with multi-frames
    card_frames = []
    start = time.time()
    while len(card_frames) < CARD_CAPTURE_FRAMES and time.time() - start < 15:
        with cam_lock:
            ret, frame = cap.read()
        if ret and sharpness_score(frame) > 50:
            card_frames.append(frame)
        time.sleep(0.05)
    first, full = extract_name_from_card(card_frames[0]) if card_frames else (None, None)  # Single attempt on best frame, but captured multiple for selection
    if not full:
        speak("Unable to read the card clearly. Please hold it steady and try again.")
        is_registering = False
        return
    if not names_match(spoken_name, full):
        speak("Name mismatch detected. Please try again.")
        is_registering = False
        return
    # Capture face
    face_frames = []
    encodings = []
    start = time.time()
    while len(face_frames) < MIN_FRAMES_CAPTURE and time.time() - start < 15:
        with cam_lock:
            ret, frame = cap.read()
        if ret:
            locs, encs = detect_faces(frame)
            if encs:
                sharpness = sharpness_score(frame)
                face_frames.append((frame, sharpness))
                encodings.append(encs[0])
        time.sleep(0.05)
    if not encodings:
        speak("No face detected.")
        is_registering = False
        return
    face_frames.sort(key=lambda x: x[1], reverse=True)
    best_photo = face_frames[0][0]
    avg_encoding = np.mean(encodings, axis=0)
    photo_path = f"photos/{spoken_name.replace(' ', '_')}.jpg"
    os.makedirs('photos', exist_ok=True)
    cv2.imwrite(photo_path, best_photo)
    save_db(spoken_name, avg_encoding, full, photo_path)
    load_db()
    speak("Your registration is done. Thank you for registering.")
    is_registering = False

# Background Listener
def background_listener():
    while active:
        text = listen_once(timeout=5)
        if "register this unknown person" in text:
            if not is_registering:
                threading.Thread(target=registration, daemon=True).start()

# Video Loop
def video_loop():
    while active:
        with cam_lock:
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        locs, encs = detect_faces(frame)
        known_count = 0
        unknown_count = 0
        names = []
        for enc in encs:
            idx = match_face(enc)
            if idx is not None:
                name = known_names[idx]
                known_count += 1
                if name not in last_greet or time.time() - last_greet[name] > GREETING_COOLDOWN:
                    last_greet[name] = time.time()
                    speak(f"Welcome back, {name}.")
            else:
                unknown_count += 1
                name = "Unknown"
            names.append(name)
        current_state = (known_count, unknown_count)
        if current_state != last_scenario_state and time.time() - last_scenario_time > SCENARIO_DEBOUNCE:
            last_scenario_state = current_state
            last_scenario_time = time.time()
            total = known_count + unknown_count
            if total == 0:
                scenario = "No persons detected."
            elif known_count == 1 and unknown_count == 1:
                scenario = "There is 1 known person and 1 unknown person."
            elif known_count == 1:
                scenario = f"There is 1 known person and {unknown_count} unknown persons."
            elif unknown_count == 1:
                scenario = f"There are {known_count} known persons and 1 unknown person."
            else:
                scenario = f"There are {known_count} known persons and {unknown_count} unknown persons."
            speak(scenario)
        # Draw
        for (top, right, bottom, left), name in zip(locs, names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        # GUI update
        if root:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk)
            video_label.image = imgtk
        time.sleep(0.03)

# Beep
def beep():
    frequency = 1000
    duration = 200
    try:
        import winsound
        winsound.Beep(frequency, duration)
    except:
        print('\a')

# Shutdown
def shutdown(sig, frame):
    global active
    active = False
    speak("System shutting down. Thank you for using Drishya AI.")
    if cap:
        cap.release()
    if root:
        root.destroy()
    sys.exit(0)

# Main
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python main.py\nSee top for instructions.")
        sys.exit(0)

    load_db()
    speak("Drishya AI system activated.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        sys.exit(1)

    root = tk.Tk()
    root.title("Drishya AI")
    video_label = ttk.Label(root)
    video_label.pack()
    status_label = ttk.Label(root, text="Status: Active")
    status_label.pack()

    signal.signal(signal.SIGINT, shutdown)

    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=background_listener, daemon=True).start()
    threading.Thread(target=video_loop, daemon=True).start()
    if AWS_AVAILABLE and S3_BUCKET:
        threading.Thread(target=aws_sync_thread, daemon=True).start()

    root.mainloop()