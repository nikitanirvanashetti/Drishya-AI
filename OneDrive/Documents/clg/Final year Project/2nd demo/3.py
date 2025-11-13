#!/usr/bin/env python3
"""
Drishya AI — registration flow updated per user spec.

Key behaviors:
- Only registers when explicit "register" command heard (no negation).
- Single-attempt listening for register command and name confirmation.
- Card OCR uses 120-frame capture, chooses sharpest frame, runs OCR once.
- Accepts only Aadhaar or PAN.
- Face capture & encoding: capture 120 frames, choose best frames, average encoding.
- Announces registration completed only after verification.
- Greet/name announcement rate-limited to once per 10 seconds.
- Beep + "Give your response after the beep" before listening.
"""

import os, sys, time, re, json, threading
from queue import Queue
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

# Libraries
try:
    import face_recognition
except Exception:
    raise RuntimeError("Install face_recognition: pip install face-recognition")

try:
    import pyttsx3
except Exception:
    raise RuntimeError("Install pyttsx3: pip install pyttsx3")

try:
    import speech_recognition as sr
except Exception:
    raise RuntimeError("Install SpeechRecognition and pyaudio: pip install SpeechRecognition pyaudio")

try:
    import easyocr
    OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
except Exception:
    OCR_READER = None
    print("Warning: easyocr not available. Card OCR will be limited. pip install easyocr")

try:
    from rapidfuzz import fuzz
    FUZZY = True
except:
    FUZZY = False

# Optional beep fallback
try:
    import winsound
    HAVE_WINSOUND = True
except:
    HAVE_WINSOUND = False

# ---------- CONFIG ----------
JSON_FILE = "faces.json"
REG_FACE_FRAMES = 120
REG_CARD_FRAMES = 120
BEST_FRAME_COUNT = 20
FACE_TOLERANCE = 0.45
ANNOUNCE_INTERVAL = 6.0
GREET_COOLDOWN = 10.0
CAM_W = 640
CAM_H = 480

# ---------- GLOBALS ----------
engine_lock = threading.Lock()
camera_lock = threading.Lock()
data_lock = threading.Lock()
cap = None
active = True

known_encodings = []
known_names = []
known_meta = {}
command_queue = Queue()

last_scenario_time = 0
last_greet = {}  # name -> timestamp of last greeting

# ---------- UTILS ----------
def speak(text, block=True):
    print("TTS:", text)
    def _tts():
        try:
            with engine_lock:
                engine = pyttsx3.init()
                engine.setProperty('rate', 145)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print("TTS error:", e)
    t = threading.Thread(target=_tts, daemon=True)
    t.start()
    if block:
        t.join(timeout=6)

def beep(ms=180, freq=1000):
    # try winsound on Windows
    if HAVE_WINSOUND:
        try:
            winsound.Beep(freq, ms)
            return
        except:
            pass
    # fallback: terminal bell
    sys.stdout.write('\a'); sys.stdout.flush()
    time.sleep(ms/1000.0)

def normalize_name(s):
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r'[^A-Z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def names_match(a, b, thresh=85):
    na = normalize_name(a)
    nb = normalize_name(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    if FUZZY:
        score = fuzz.token_sort_ratio(na, nb)
        return score >= thresh
    return na.split()[0] == nb.split()[0]

def save_person(name, encoding, card_type, card_name, photo_path):
    entry = {
        "name": name,
        "encoding": encoding.tolist(),
        "card_type": card_type,
        "card_name": card_name,
        "photo": photo_path,
        "timestamp": datetime.now().isoformat()
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
            known_meta[name] = {"card_type": card_type, "card_name": card_name, "photo": photo_path}
        return True
    except Exception as e:
        print("Save error:", e)
        return False

def load_db():
    global known_encodings, known_names, known_meta
    known_encodings, known_names, known_meta = [], [], {}
    if not os.path.exists(JSON_FILE):
        return
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        with data_lock:
            for entry in data:
                if 'name' in entry and 'encoding' in entry:
                    known_names.append(entry['name'])
                    known_encodings.append(np.array(entry['encoding']))
                    known_meta[entry['name']] = {'card_type': entry.get('card_type',''),
                                                 'card_name': entry.get('card_name',''),
                                                 'photo': entry.get('photo','')}
        print(f"Loaded {len(known_names)} persons.")
    except Exception as e:
        print("DB load error:", e)

# ---------- FACE / OCR helpers ----------
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
    except Exception as e:
        # on error return empty
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

def extract_name_from_image(frame):
    """
    Run easyocr on one image and heuristically extract name candidate and card type.
    Return (name, card_type, message).
    """
    if OCR_READER is None:
        return None, None, "OCR not available"
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = OCR_READER.readtext(np.array(pil), detail=0, paragraph=False)
        if not results:
            return None, None, "No text"
        text = " ".join(results).upper()
        card_type = None
        if any(k in text for k in ["AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION", "UNIQUE ID"]):
            card_type = "AADHAAR"
        elif any(k in text for k in ["PAN", "PERMANENT ACCOUNT", "INCOME TAX"]):
            card_type = "PAN"
        if not card_type:
            return None, None, "Not Aadhaar/PAN"
        # find candidate name tokens (2-5 words, no digits)
        tokens = re.split(r'[\n,;]', text)
        candidates = []
        for t in tokens:
            t_clean = re.sub(r'[^A-Z\s]', ' ', t).strip()
            if not t_clean: continue
            if any(ch.isdigit() for ch in t_clean): continue
            words = t_clean.split()
            if 2 <= len(words) <= 5:
                candidates.append(" ".join(words).title())
        if candidates:
            return candidates[0], card_type, "OK"
        return None, card_type, "No name"
    except Exception as e:
        return None, None, f"OCR err: {e}"

# ---------- Speech single-listen helper ----------
def single_listen(timeout=6, phrase_limit=6, lang='en-IN'):
    """Listen once and return recognized text (str) or '' on failure."""
    try:
        with sr.Microphone() as source:
            r = sr.Recognizer()
            r.adjust_for_ambient_noise(source, duration=0.6)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            try:
                text = r.recognize_google(audio, language=lang)
            except:
                try:
                    text = r.recognize_google(audio, language='en-US')
                except:
                    text = ""
            return text.strip()
    except Exception as e:
        print("single_listen error:", e)
        return ""

# ---------- Registration flow ----------
def perform_registration(current_frame):
    """
    1) beep -> single listen for name (one attempt)
    2) ask "Shall I remember you as {name}? Please confirm." -> single listen
    3) if confirmed -> "Ok, I will remember you after your verification."
    4) ask to show Aadhaar/PAN -> capture REG_CARD_FRAMES frames, pick best, run OCR once on best image
    5) if OCR reads Aadhaar/PAN and name matches spoken name -> capture REG_FACE_FRAMES frames for face -> save -> announce
    """
    # (1) ask for name
    speak("After the beep, please say your full name clearly.", block=True)
    beep()
    name_heard = single_listen(timeout=7, phrase_limit=6)
    if not name_heard:
        speak("I could not hear your name. Registration cancelled.", block=True)
        return False
    name_cap = " ".join(w.capitalize() for w in name_heard.split())

    # (2) confirmation
    speak(f"Shall I remember you as {name_cap}? Please confirm.", block=True)
    beep()
    conf = single_listen(timeout=5, phrase_limit=4)
    if not conf:
        speak("No confirmation detected. Registration cancelled.", block=True)
        return False
    conf_low = conf.lower()
    if any(w in conf_low for w in ["no", "don't", "do not", "not"]):
        speak("Confirmation denied. Registration cancelled.", block=True)
        return False
    if any(w in conf_low for w in ["yes", "yep", "correct", "confirm", "ok", "okay"]):
        final_name = name_cap
    else:
        # assume user repeated corrected name
        final_name = " ".join(w.capitalize() for w in conf.split())

    speak("Ok. I will remember you after your verification.", block=True)

    # (3) card OCR - capture multiple frames, pick the sharpest
    speak("Now please show your Aadhaar card or PAN card to the camera. Hold it steady. I will capture a few frames and read the name.", block=True)
    captured = []
    start = time.time()
    while len(captured) < REG_CARD_FRAMES and (time.time() - start) < 30:
        with camera_lock:
            if cap is None: break
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        captured.append((frame.copy(), sharpness_score(frame)))
        time.sleep(0.02)
    if not captured:
        speak("Could not capture card frames. Registration cancelled.", block=True)
        return False

    # pick best sharp frame
    best_frame = max(captured, key=lambda x: x[1])[0]
    # run OCR once on best_frame
    card_name, card_type, msg = extract_name_from_image(best_frame)
    if msg.startswith("OCR err") or msg == "No text":
        speak("Could not read text from the card. Registration cancelled.", block=True)
        return False
    if msg == "Not Aadhaar/PAN":
        speak("Only Aadhaar or PAN cards are accepted. Registration cancelled.", block=True)
        return False
    if not card_name:
        speak("Could not find a name on the card. Registration cancelled.", block=True)
        return False

    speak(f"{card_type} read. Name on card: {card_name}.", block=False)

    # (4) compare names
    if not names_match(final_name, card_name):
        speak(f"Name mismatch: you said {final_name} but card shows {card_name}. Registration cancelled.", block=True)
        return False

    # (5) capture face frames for encoding
    speak(f"Name matched. Please look at the camera and stay still. Capturing {REG_FACE_FRAMES} face frames now.", block=True)
    face_frames = []
    enc_list = []
    startf = time.time()
    while len(face_frames) < REG_FACE_FRAMES and (time.time() - startf) < 60:
        with camera_lock:
            if cap is None: break
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        locs, encs = detect_faces_and_encodings(frame)
        if encs:
            face_frames.append((frame.copy(), sharpness_score(frame)))
            enc_list.append(encs[0])
        time.sleep(0.02)
    if not enc_list:
        speak("Could not detect face clearly for registration. Registration cancelled.", block=True)
        return False

    # make best photo and average encoding
    best_face = sorted(face_frames, key=lambda x: x[1], reverse=True)[0][0]
    try:
        avg_enc = np.mean(enc_list, axis=0)
    except:
        avg_enc = enc_list[0]

    photos_dir = "registered_photos"
    os.makedirs(photos_dir, exist_ok=True)
    photo_name = f"{final_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    photo_path = os.path.join(photos_dir, photo_name)
    cv2.imwrite(photo_path, best_face)

    saved = save_person(final_name, avg_enc, card_type, card_name, photo_path)
    if saved:
        speak(f"Registration completed for {final_name}. I will remember you.", block=True)
        # mark last greet so greeting won't repeat immediately
        last_greet[final_name] = time.time()
        return True
    else:
        speak("Failed to save registration. Please try again.", block=True)
        return False

# ---------- Announce counts (with strict register detection) ----------
def announce_counts(known_count, unknown_count):
    global last_scenario_time
    now = time.time()
    if now - last_scenario_time < ANNOUNCE_INTERVAL:
        return
    last_scenario_time = now
    total = known_count + unknown_count
    if total == 0:
        speak("No persons detected in front of you.", block=False)
        return

    if unknown_count > 0 and known_count == 0:
        # unknown-only announcement
        if unknown_count == 1:
            speak("There is one unknown person in front of you.", block=True)
        else:
            speak(f"There are {unknown_count} unknown persons in front of you.", block=True)

        # beep and single listen for explicit "register" without negation
        speak("Give your response after the beep.", block=True)
        beep()
        text = single_listen(timeout=5, phrase_limit=4)
        if not text:
            speak("No response detected. Continuing monitoring.", block=False)
            return
        txt = text.lower().strip()
        print("After-beep heard:", txt)
        # Accept only if contains 'register' and does NOT contain explicit negation
        negations = ["do not", "don't", "dont", "no", "not"]
        if "register" in txt and not any(neg in txt for neg in negations):
            speak("Register command detected. Beginning registration.", block=False)
            # queue registration (will do single-listen steps inside)
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

# ---------- Overlay drawing ----------
def draw_overlay(frame, locs, names):
    out = frame.copy()
    for (t,r,b,l), nm in zip(locs, names):
        color = (0,255,0) if nm != "Unknown" else (0,0,255)
        cv2.rectangle(out, (l,t), (r,b), color, 2)
        label = nm
        cv2.putText(out, label, (l, b+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return out

# ---------- Main loop ----------
def main_loop():
    global cap, active
    last_known = -1
    last_unknown = -1
    while active:
        with camera_lock:
            if cap is None or not cap.isOpened():
                time.sleep(0.2); continue
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue

        locs, encs = detect_faces_and_encodings(frame)
        names = []
        known_count = 0
        unknown_count = 0
        for loc, enc in zip(locs, encs):
            idx = match_encoding(enc)
            if idx is not None:
                name = known_names[idx]
                names.append(name)
                known_count += 1
                # greet rate-limited per person (10 sec)
                now = time.time()
                last = last_greet.get(name, 0)
                if now - last > GREET_COOLDOWN:
                    speak(f"Welcome {name}.", block=False)
                    last_greet[name] = now
            else:
                names.append("Unknown")
                unknown_count += 1

        # announce changes
        if known_count != last_known or unknown_count != last_unknown:
            announce_counts(known_count, unknown_count)
            last_known = known_count
            last_unknown = unknown_count

        out = draw_overlay(frame, locs, names)
        cv2.imshow("Drishya AI", out)

        # handle queued commands
        try:
            cmd, raw = command_queue.get_nowait()
            if cmd == "register":
                # run registration in blocking manner to ensure single-attempt flows
                perform_registration(frame)
                load_db()
            elif cmd == "shutdown":
                speak("Shutting down.", block=True)
                active = False
                break
        except Exception:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Quit requested. Exiting.", block=True)
            active = False
            break
        time.sleep(0.01)

    with camera_lock:
        if cap:
            cap.release()
    cv2.destroyAllWindows()

# ---------- Startup ----------
def start_system():
    global cap
    load_db()
    speak("Drishya AI system activated.", block=False)
    speak("Opening camera for continuous monitoring.", block=False)

    # open camera
    cap = None
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # background listener (for general commands like shutdown)
    listener = threading.Thread(target=background_listener_loop, daemon=True)
    listener.start()

    cv2.namedWindow("Drishya AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drishya AI", 800, 600)
    speak("System is now active. Say register this person after the beep to begin registration.", block=False)
    try:
        main_loop()
    except KeyboardInterrupt:
        speak("Interrupted. Shutting down.", block=True)

def background_listener_loop():
    """Non-blocking background listener for general commands like shutdown"""
    r = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception as e:
        print("Mic init failed:", e)
        return
    while active:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.6)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio, language='en-IN')
            except:
                try:
                    text = r.recognize_google(audio, language='en-US')
                except:
                    text = ""
            if text:
                txt = text.lower().strip()
                print("BG heard:", txt)
                if "shutdown" in txt or "stop" in txt or "exit" in txt or "quit" in txt:
                    command_queue.put(("shutdown", txt))
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            print("BG listener err:", e)
            time.sleep(0.5)

if __name__ == "__main__":
    start_system()
