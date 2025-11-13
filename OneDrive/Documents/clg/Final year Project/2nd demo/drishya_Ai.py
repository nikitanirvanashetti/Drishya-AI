#!/usr/bin/env python3
"""
Drishya AI — Refined single-file voice-activated continuous camera registration system.
Refinements:
- beep + "Give your response after the beep" -> listens once -> registers only if user says "register"
- During registration: beep -> listen once for name -> "Shall I remember you as {name}? Please confirm." -> listen once.
- If confirm -> "Ok, I will remember you after your verification." -> accepts ONLY Aadhaar or PAN -> OCR first valid read used.
- Captures 120 frames (best frames used), saves photo and encoding when card name matches spoken name.
- Continuous overlay: red box + "Unknown" and green box + person name.

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
CARD_CAPTURE_MAX_SECONDS = 20
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
def load_db():
    global known_encodings, known_names, known_meta
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
            json.dump([], f)
        return
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        with data_lock:
            known_encodings = [np.array(entry['encoding']) for entry in data if 'encoding' in entry]
            known_names = [entry['name'] for entry in data if 'name' in entry]
            known_meta = {entry['name']: {'card_type': entry.get('card_type',''),
                                         'card_name': entry.get('card_name',''),
                                         'photo': entry.get('photo','')}
                          for entry in data if 'name' in entry}
        print(f"Loaded {len(known_names)} known persons.")
    except Exception as e:
        print("DB load error:", e)

def save_person(name, encoding, card_type, card_name, photo_path):
    entry = {
        'name': name,
        'encoding': encoding.tolist(),
        'card_type': card_type,
        'card_name': card_name,
        'photo': photo_path,
        'timestamp': datetime.now().isoformat()
    }
    try:
        data = []
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    data = []
        data.append(entry)
        with open(JSON_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        with data_lock:
            known_names.append(name)
            known_encodings.append(np.array(encoding))
            known_meta[name] = {'card_type': card_type, 'card_name': card_name, 'photo': photo_path}
        return True
    except Exception as e:
        print("Save error:", e)
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

# ---------------- OCR ----------------
def extract_name_from_card(frame):
    """
    Uses EasyOCR to read text and returns (name, card_type, message).
    Card must be detected as 'AADHAAR' or 'PAN' to be accepted.
    """
    if OCR_READER is None:
        return None, None, "OCR not available"
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = OCR_READER.readtext(np.array(pil), detail=0, paragraph=False)
        if not results:
            return None, None, "No text detected"
        text = " ".join(results).upper()
        card_type = None
        if any(k in text for k in ["AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION", "GOVT OF INDIA UID"]):
            card_type = "AADHAAR"
        elif any(k in text for k in ["PAN", "PERMANENT ACCOUNT", "INCOME TAX"]):
            card_type = "PAN"
        # extract candidate names heuristically
        tokens = re.split(r'[\n,;]', text)
        candidates = []
        for t in tokens:
            t_clean = re.sub(r'[^A-Z\s]', ' ', t).strip()
            if not t_clean:
                continue
            if any(ch.isdigit() for ch in t_clean):
                continue
            words = t_clean.split()
            if 2 <= len(words) <= 5:
                cand = " ".join(words)
                if len(cand) >= 4:
                    candidates.append(cand.title())
        if not card_type:
            return None, None, "Card not Aadhaar or PAN"
        if candidates:
            return candidates[0], card_type, "OK"
        return None, card_type, "No name candidate"
    except Exception as e:
        return None, None, f"OCR error: {e}"

# ---------------- Background Listener (keeps listening for general commands) ----------------
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
                    if any(kw in txt for kw in ["register this person", "register this unknown", "register this", "register person", "register"]):
                        command_queue.put(("register", txt))
                    elif any(kw in txt for kw in ["who is this", "identify", "who is there"]):
                        command_queue.put(("identify", txt))
                    elif any(kw in txt for kw in ["stop", "exit", "shutdown", "quit"]):
                        command_queue.put(("shutdown", txt))
                    else:
                        command_queue.put(("text", txt))
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print("Listener loop error:", e)
                time.sleep(0.5)

    def stop(self):
        self.running = False

# ---------------- Registration Flow ----------------
def single_listen(timeout=6, phrase_time_limit=5, lang_pref='en-IN'):
    """Listen once and return recognized text (lowercased) or ''."""
    try:
        with sr.Microphone() as source:
            r = sr.Recognizer()
            r.adjust_for_ambient_noise(source, duration=0.6)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            try:
                text = r.recognize_google(audio, language=lang_pref)
            except:
                try:
                    text = r.recognize_google(audio, language='en-US')
                except:
                    text = ""
            return text.strip()
    except Exception as e:
        print("single_listen error:", e)
        return ""

def registration_flow(passed_frame):
    """
    Flow:
    - beep, listen once for name
    - ask "Shall I remember you as {name}? Please confirm." -> listen once
    - if confirm -> speak "Ok, I will remember you after your verification." -> proceed
    - request Aadhaar or PAN -> OCR first valid found used
    - compare card name and spoken name -> if match, capture 120 frames, save, announce success
    """
    # 1) ask for name
    speak("Understood. After the beep, please say your full name clearly.", block=True)
    beep()
    name_heard = single_listen(timeout=7, phrase_time_limit=6)
    if not name_heard:
        speak("I did not hear your name. Registration cancelled.", block=True)
        return False
    # normalize capitalization for TTS
    name_cap = " ".join(w.capitalize() for w in name_heard.split())
    # 2) ask for confirmation wording asked by user
    speak(f"Shall I remember you as {name_cap}? Please confirm.", block=True)
    beep()
    confirm_heard = single_listen(timeout=5, phrase_time_limit=4)
    if not confirm_heard:
        speak("No confirmation detected. Registration cancelled.", block=True)
        return False
    conf_low = confirm_heard.lower()
    if any(w in conf_low for w in ["yes", "yep", "correct", "confirm", "yup", "okay", "ok"]):
        final_name = name_cap
    else:
        # if user repeated name, accept that as corrected name
        # use the spoken phrase as corrected name
        final_name = " ".join(w.capitalize() for w in confirm_heard.split())
    speak("Ok. I will remember you after your verification.", block=True)

    # 3) Request Aadhaar or PAN and perform OCR (first valid attempt)
    speak("Now please show your Aadhaar card or PAN card clearly to the camera. I will read the name from the card.", block=True)
    card_name = None
    card_type = None
    start = time.time()
    attempt_once = False
    while (time.time() - start) < CARD_CAPTURE_MAX_SECONDS:
        with camera_lock:
            if cap is None:
                break
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        # quick sharpness check
        if sharpness_score(frame) < 30:
            # advise user if too blurry occasionally
            if not attempt_once or (time.time()-start) > 3:
                speak("Card image too blurry. Hold the card steady and ensure good lighting.", block=False)
                attempt_once = True
            time.sleep(0.2)
            continue
        # OCR once per good frame
        name_extracted, ctype, msg = extract_name_from_card(frame)
        if msg.startswith("OCR error"):
            time.sleep(0.15)
            continue
        if msg == "Card not Aadhaar or PAN":
            speak("Only Aadhaar or PAN accepted. Please show Aadhaar or PAN card.", block=False)
            time.sleep(0.4)
            continue
        if name_extracted and ctype in ("AADHAAR", "PAN"):
            card_name = name_extracted
            card_type = ctype
            speak(f"{card_type} read. Name on card is {card_name}.", block=False)
            break
        else:
            # if no name candidate yet, ask to adjust
            time.sleep(0.15)
            continue
    if not card_name:
        speak("Could not read Aadhaar or PAN. Registration cancelled.", block=True)
        return False

    # 4) Compare final_name with card_name
    if not names_match(final_name, card_name):
        speak(f"Name mismatch: you said {final_name}, but the card name is {card_name}. Registration cancelled.", block=True)
        return False

    # 5) Capture MIN_CAPTURE_FRAMES frames and average encoding
    speak(f"Name matched. Now please look at the camera and stay still. I will capture {MIN_CAPTURE_FRAMES} frames for registration.", block=True)
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

    # choose best frames and average encoding
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

    # save person
    saved = save_person(final_name, avg_enc, card_type, card_name, photo_path)
    if saved:
        speak(f"Registration of {final_name} completed successfully. Thank you.", block=True)
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
    # prefer unknown-only announcement that triggers the beep+single-listen for 'register'
    if unknown_count > 0 and known_count == 0:
        if unknown_count == 1:
            speak("There is one unknown person in front of you.", block=True)
        else:
            speak(f"There are {unknown_count} unknown persons in front of you.", block=True)
        speak("Give your response after the beep.", block=True)
        beep()
        # Listen once after beep for register command
        text = single_listen(timeout=5, phrase_time_limit=4)
        if not text:
            speak("No response detected. Continuing monitoring.", block=False)
            return
        txt = text.lower()
        print("Post-announcement heard:", txt)
        if "register" in txt:
            # queue register command (registration_flow will handle name prompt)
            speak("Registration command detected. Beginning registration flow.", block=False)
            command_queue.put(("register", txt))
        else:
            speak("Continuing monitoring.", block=False)
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

        # announce only when counts change
        if known_count != last_known or unknown_count != last_unknown:
            announce_counts(known_count, unknown_count)
            last_known = known_count
            last_unknown = unknown_count

        out = draw_overlay(frame, locs, names_list)
        cv2.imshow("Drishya AI", out)

        # handle queued commands
        try:
            cmd, raw = command_queue.get_nowait()
            if cmd == "register":
                speak("Register command received. Preparing to register the person now.", block=False)
                registration_flow(frame)
                load_db()
            elif cmd == "identify":
                if locs and encs:
                    idx = match_encoding(encs[0])
                    if idx is not None:
                        speak(f"This person is {known_names[idx]}.", block=False)
                    else:
                        speak("I do not recognize this person.", block=False)
                else:
                    speak("No face detected to identify.", block=False)
            elif cmd == "shutdown":
                speak("Shutdown command received. Shutting down system.", block=True)
                active = False
                break
            elif cmd == "text":
                pass
        except Exception:
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

# ---------------- STARTUP ----------------
def start_system():
    global cap
    load_db()
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
    listener.stop()
    speak("System shutting down. Goodbye.", block=False)

if __name__ == "__main__":
    start_system()
