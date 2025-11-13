"""
================================================================================
DRISHYA AI - COMPLETE FIXED CONTINUOUS CAMERA & VOICE SYSTEM
================================================================================
✅ Camera continuously visible from start
✅ Dynamic person count updates in real-time
✅ Fixed voice recognition with proper error handling
✅ Background listening that doesn't interfere with registration
✅ Smooth concurrent operation

INSTALLATION:
pip install opencv-python face-recognition pyttsx3 SpeechRecognition pyaudio rapidfuzz easyocr numpy scipy Pillow

RUN:
python 2.py
"""

import sys
import cv2
import numpy as np
import json
import os
import time
import threading
from datetime import datetime
import re
from queue import Queue

print("\n" + "="*70)
print("🎙️  DRISHYA AI - FIXED SYSTEM")
print("="*70 + "\n")

# ==================== IMPORTS ====================
try:
    import pyttsx3
    print("✅ Text-to-Speech")
except:
    print("❌ Install: pip install pyttsx3")
    sys.exit(1)

try:
    import face_recognition
    print("✅ Face Recognition")
except:
    print("❌ Install: pip install face-recognition")
    sys.exit(1)

try:
    import speech_recognition as sr
    print("✅ Speech Recognition")
except:
    print("❌ Install: pip install SpeechRecognition pyaudio")
    sys.exit(1)

try:
    from rapidfuzz import fuzz
    FUZZY = True
    print("✅ Fuzzy Matching")
except:
    FUZZY = False
    print("⚠️  Fuzzy matching unavailable")

try:
    import easyocr
    print("✅ OCR (EasyOCR)")
    OCR = easyocr.Reader(['en'], gpu=False, verbose=False)
except:
    print("⚠️  OCR unavailable")
    OCR = None

try:
    from PIL import Image
    print("✅ Image Processing (Pillow)")
except:
    print("❌ Install: pip install Pillow")
    sys.exit(1)

print("="*70 + "\n")

# ==================== CONFIG ====================
JSON_FILE = "faces.json"
FACE_TOL = 0.40
COOLDOWN = 15.0
SCENARIO_ANNOUNCE_INTERVAL = 10.0
MIN_FRAMES_CAPTURE = 120

# ==================== GLOBALS ====================
known_encodings = []
known_names = []
known_cards = {}
known_photos = {}

cap = None
cam_lock = threading.Lock()
face_lock = threading.Lock()
tts_lock = threading.Lock()

active = True
is_registering = False
registration_in_progress = False
last_greet = {}
last_scenario_time = 0

command_queue = Queue()

# Current frame for display (updated continuously)
current_frame = None
current_locs = []
current_names = []
frame_lock = threading.Lock()

# ==================== TTS ====================
def speak(text, block=True):
    """Thread-safe TTS with blocking control"""
    print(f"🔊 {text}")
    
    def _speak():
        try:
            with tts_lock:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                engine.setProperty('volume', 1.0)
                voices = engine.getProperty('voices')
                if voices:
                    engine.setProperty('voice', voices[0].id)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
        except Exception as e:
            print(f"⚠️ TTS error: {e}")
    
    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    if block:
        t.join()

def beep():
    """Beep sound"""
    try:
        import winsound
        winsound.Beep(1000, 200)
    except:
        print("*BEEP*")

# ==================== DATABASE ====================
def load_db():
    """Load face database"""
    global known_encodings, known_names, known_cards, known_photos
    
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
            json.dump([], f)
        return
    
    try:
        with face_lock:
            with open(JSON_FILE, 'r') as f:
                data = json.load(f)
            
            known_encodings = []
            known_names = []
            known_cards = {}
            known_photos = {}
            
            for e in data:
                if e.get('name'):
                    known_names.append(e['name'])
                    known_encodings.append(np.array(e['encoding']))
                    known_cards[e['name']] = {
                        'card_type': e.get('card_type', ''),
                        'card_name': e.get('card_name', '')
                    }
                    if e.get('photo'):
                        known_photos[e['name']] = e['photo']
            
            if known_names:
                speak(f"Loaded {len(known_names)} registered persons.", block=False)
    except Exception as e:
        print(f"⚠️ Load error: {e}")

def save_face(name, enc, card_type, card_name, photo_path):
    """Save face to database"""
    try:
        with face_lock:
            data = []
            if os.path.exists(JSON_FILE):
                with open(JSON_FILE, 'r') as f:
                    data = json.load(f)
            
            data.append({
                'name': name,
                'encoding': enc.tolist(),
                'card_type': card_type,
                'card_name': card_name,
                'photo': photo_path,
                'timestamp': datetime.now().isoformat()
            })
            
            with open(JSON_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
    except:
        return False

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

# ==================== VOICE RECOGNITION - FIXED ====================
class EnhancedVoiceRecognition:
    """Fixed voice recognition with proper error handling"""
    
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = 300
        self.rec.dynamic_energy_threshold = True
        self.rec.pause_threshold = 0.8
    
    def listen_once_ultra_accurate(self, prompt="", timeout=20):
        """Ultra-accurate listening with fixed error handling"""
        try:
            with sr.Microphone(sample_rate=16000) as source:
                if prompt:
                    speak(prompt, block=True)
                
                print("\n🎤 LISTENING NOW...")
                
                # Adjust for ambient noise
                self.rec.adjust_for_ambient_noise(source, duration=1.0)
                
                # Listen
                audio = self.rec.listen(source, timeout=timeout, phrase_time_limit=15)
                
                print("🔄 Processing audio...")
                
                # Try multiple engines
                results = []
                
                # Try India
                try:
                    text = self.rec.recognize_google(audio, language='en-IN')
                    if text and len(text.strip()) > 0:
                        results.append(text.strip())
                        print(f"✅ India: '{text.strip()}'")
                except sr.UnknownValueError:
                    print("⚠️ India: Could not understand audio")
                except sr.RequestError as e:
                    print(f"⚠️ India: Service error - {e}")
                except Exception as e:
                    print(f"⚠️ India: {type(e).__name__}")
                
                # Try US
                try:
                    text = self.rec.recognize_google(audio, language='en-US')
                    if text and len(text.strip()) > 0:
                        results.append(text.strip())
                        print(f"✅ US: '{text.strip()}'")
                except sr.UnknownValueError:
                    print("⚠️ US: Could not understand audio")
                except sr.RequestError as e:
                    print(f"⚠️ US: Service error - {e}")
                except Exception as e:
                    print(f"⚠️ US: {type(e).__name__}")
                
                # Try UK
                try:
                    text = self.rec.recognize_google(audio, language='en-GB')
                    if text and len(text.strip()) > 0:
                        results.append(text.strip())
                        print(f"✅ UK: '{text.strip()}'")
                except sr.UnknownValueError:
                    print("⚠️ UK: Could not understand audio")
                except sr.RequestError as e:
                    print(f"⚠️ UK: Service error - {e}")
                except Exception as e:
                    print(f"⚠️ UK: {type(e).__name__}")
                
                if not results:
                    print("❌ No speech detected by any engine")
                    return ""
                
                # Voting logic
                if len(results) == 1:
                    final = results[0]
                elif FUZZY:
                    groups = []
                    for r in results:
                        found = False
                        for g in groups:
                            if fuzz.ratio(r.lower(), g['text'].lower()) >= 85:
                                g['count'] += 1
                                if len(r) > len(g['text']):
                                    g['text'] = r
                                found = True
                                break
                        if not found:
                            groups.append({'text': r, 'count': 1})
                    
                    best = max(groups, key=lambda x: x['count'])
                    final = best['text']
                    print(f"🎯 Final: '{final}' (votes: {best['count']}/{len(results)})")
                else:
                    final = results[0]
                
                print(f"✅ RECOGNIZED: '{final}'\n")
                return final
        
        except sr.WaitTimeoutError:
            print("⏱️ Timeout - no speech detected\n")
            return ""
        except Exception as e:
            print(f"❌ Error: {type(e).__name__} - {e}\n")
            return ""

mic = EnhancedVoiceRecognition()

# ==================== BACKGROUND VOICE LISTENER - NON-BLOCKING ====================
def background_voice_listener():
    """Lightweight background listener that doesn't interfere"""
    print("🎙️ Background voice monitoring started")
    
    listener_rec = sr.Recognizer()
    listener_rec.energy_threshold = 400
    listener_rec.pause_threshold = 0.8
    
    with sr.Microphone() as source:
        listener_rec.adjust_for_ambient_noise(source, duration=1.0)
        print("✅ Background listener calibrated\n")
        
        while active:
            try:
                # Skip if already registering
                if registration_in_progress:
                    time.sleep(0.5)
                    continue
                
                # Quick listen
                audio = listener_rec.listen(source, timeout=2, phrase_time_limit=6)
                
                # Quick recognition
                try:
                    text = listener_rec.recognize_google(audio, language='en-IN').lower()
                    
                    if any(phrase in text for phrase in [
                        "register this person",
                        "register person",
                        "start registration"
                    ]):
                        print(f"🎤 Background detected: '{text}'")
                        speak("Registration command detected.", block=False)
                        command_queue.put(("register", text))
                except:
                    pass
            
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                if active:
                    time.sleep(0.3)

# ==================== SCENARIO ANNOUNCEMENT ====================
def announce_counts(known_count, unknown_count):
    """Announce scenario with beep confirmation"""
    global last_scenario_time, registration_in_progress
    
    if registration_in_progress:
        return
    
    now = time.time()
    if now - last_scenario_time < SCENARIO_ANNOUNCE_INTERVAL:
        return
    
    last_scenario_time = now
    total = known_count + unknown_count
    
    if total == 0:
        speak("No persons detected.", block=False)
        return
    
    # Grammar handling
    if unknown_count > 0 and known_count == 0:
        if unknown_count == 1:
            speak("There is one unknown person in front of you.", block=True)
        else:
            speak(f"There are {unknown_count} unknown persons in front of you.", block=True)
        
        speak("Say register this person if you want to register them.", block=True)
        time.sleep(0.5)
        speak("Give your response after the beep.", block=True)
        beep()
        time.sleep(0.5)
        
        # Listen for response
        try:
            temp_rec = sr.Recognizer()
            with sr.Microphone() as source:
                temp_rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = temp_rec.listen(source, timeout=6, phrase_time_limit=6)
                
                try:
                    text = temp_rec.recognize_google(audio, language='en-IN').lower()
                    print(f"📥 User response: '{text}'")
                    
                    if "register" in text:
                        speak("Starting registration.", block=False)
                        command_queue.put(("register", text))
                    else:
                        speak("Continuing monitoring.", block=False)
                except:
                    speak("No clear response. Continuing monitoring.", block=False)
        except:
            speak("No response heard.", block=False)
    
    elif known_count > 0 and unknown_count == 0:
        if known_count == 1:
            speak("There is one known person in front of you.", block=False)
        else:
            speak(f"There are {known_count} known persons in front of you.", block=False)
    
    else:
        speak(f"There are {total} persons: {known_count} known and {unknown_count} unknown.", block=True)
        time.sleep(0.3)
        speak("Say register this person if you want to register unknown persons.", block=True)
        time.sleep(0.5)
        speak("Give your response after the beep.", block=True)
        beep()
        time.sleep(0.5)
        
        try:
            temp_rec = sr.Recognizer()
            with sr.Microphone() as source:
                temp_rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = temp_rec.listen(source, timeout=6, phrase_time_limit=6)
                
                try:
                    text = temp_rec.recognize_google(audio, language='en-IN').lower()
                    print(f"📥 User response: '{text}'")
                    
                    if "register" in text:
                        speak("Starting registration.", block=False)
                        command_queue.put(("register", text))
                    else:
                        speak("Continuing monitoring.", block=False)
                except:
                    speak("No clear response. Continuing monitoring.", block=False)
        except:
            speak("No response heard.", block=False)

# ==================== REGISTRATION ====================
def do_registration():
    """Registration process"""
    global is_registering, registration_in_progress
    
    if is_registering or registration_in_progress:
        return
    
    is_registering = True
    registration_in_progress = True
    
    try:
        speak("Starting registration process.", block=True)
        time.sleep(0.5)
        
        # Get name
        speak("Please say your full name after the beep.", block=True)
        beep()
        time.sleep(0.5)
        
        name_heard = mic.listen_once_ultra_accurate(timeout=20)
        
        if not name_heard:
            speak("I did not hear your name. Registration cancelled.", block=True)
            return
        
        name_heard = ' '.join(word.capitalize() for word in name_heard.split())
        
        speak(f"I heard your name as {name_heard}.", block=True)
        time.sleep(0.5)
        speak("Please confirm. Say yes or repeat your name.", block=True)
        beep()
        time.sleep(0.5)
        
        confirmation = mic.listen_once_ultra_accurate(timeout=20)
        
        if not confirmation:
            speak("No confirmation. Registration cancelled.", block=True)
            return
        
        conf_lower = confirmation.lower()
        if any(word in conf_lower for word in ['yes', 'yeah', 'correct', 'right', 'okay']):
            speak(f"Name confirmed as {name_heard}.", block=True)
        else:
            name_heard = ' '.join(word.capitalize() for word in confirmation.split())
            speak(f"Using {name_heard} as your name.", block=True)
        
        time.sleep(0.5)
        
        # Capture face
        speak("Now capturing your face.", block=True)
        time.sleep(0.3)
        speak("Please look at the camera and stay still.", block=True)
        time.sleep(1)
        speak(f"Capturing {MIN_FRAMES_CAPTURE} frames.", block=True)
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
            speak(f"Only captured {len(frames_data)} frames. Please try again.", block=True)
            return
        
        speak(f"Captured {len(frames_data)} frames successfully.", block=True)
        time.sleep(0.3)
        speak("Selecting best quality image.", block=True)
        
        best_frame_data = max(frames_data, key=lambda x: x['sharpness'])
        best_frame = best_frame_data['frame']
        
        # Save photo
        photos_dir = "registered_photos"
        os.makedirs(photos_dir, exist_ok=True)
        photo_filename = f"{name_heard.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        photo_path = os.path.join(photos_dir, photo_filename)
        cv2.imwrite(photo_path, best_frame)
        
        speak("Best frame saved.", block=True)
        
        # Average encoding
        all_encodings = [fd['encoding'] for fd in frames_data]
        avg_encoding = np.mean(all_encodings, axis=0)
        
        time.sleep(0.5)
        
        # ID Card
        speak("Now please show your ID card.", block=True)
        time.sleep(0.3)
        speak("Accepted cards: Aadhaar or PAN only.", block=True)
        time.sleep(0.5)
        speak("Hold the card clearly with good lighting.", block=True)
        time.sleep(1)
        
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
                                speak("Card is blurry. Hold steady.", block=False)
                                last_feedback = time.time()
                        elif type_extracted and not name_extracted:
                            if (time.time() - last_feedback) > 3:
                                speak(f"{type_extracted} detected but name unclear.", block=False)
                                last_feedback = time.time()
                        elif name_extracted and type_extracted:
                            card_name = name_extracted
                            card_type = type_extracted
                            speak(f"{card_type} card detected successfully.", block=True)
                            break
                        elif "not accepted" in message.lower():
                            if (time.time() - last_feedback) > 3:
                                speak("Card type not accepted. Show Aadhaar or PAN only.", block=False)
                                last_feedback = time.time()
            
            attempts += 1
            time.sleep(0.2)
        
        if not card_name or not card_type:
            speak("Could not read card. Please try again.", block=True)
            return
        
        speak(f"Card read. Type: {card_type}.", block=True)
        time.sleep(0.5)
        speak(f"Name on card: {card_name}.", block=True)
        time.sleep(0.5)
        
        # Verify match
        if not names_match(name_heard, card_name):
            speak(f"Name mismatch. You said {name_heard} but card shows {card_name}.", block=True)
            speak("Registration cancelled.", block=True)
            return
        
        speak("Name verified. Names match.", block=True)
        beep()
        time.sleep(0.5)
        
        # Save
        speak("Saving to database.", block=True)
        
        if save_face(name_heard, avg_encoding, card_type, card_name, photo_path):
            load_db()
            speak(f"Registration completed for {name_heard}.", block=True)
            beep()
            time.sleep(0.5)
            speak(f"{card_type} card verified and stored.", block=True)
            time.sleep(0.5)
            speak("Welcome to the system.", block=True)
            beep()
        else:
            speak("Failed to save. Please try again.", block=True)
    
    except Exception as e:
        print(f"❌ Registration error: {e}")
        speak("An error occurred during registration.", block=True)
    
    finally:
        is_registering = False
        registration_in_progress = False

# ==================== COMMAND PROCESSOR ====================
def process_commands():
    """Process commands from queue"""
    while active:
        try:
            if not command_queue.empty():
                cmd, text = command_queue.get()
                if cmd == "register":
                    do_registration()
            time.sleep(0.1)
        except:
            pass

# ==================== CONTINUOUS DISPLAY ====================
def continuous_display():
    """Continuously display camera feed"""
    global current_frame, current_locs, current_names
    
    while active:
        try:
            with frame_lock:
                if current_frame is not None:
                    disp = current_frame.copy()
                    
                    for (t, r, b, l), name in zip(current_locs, current_names):
                        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                        cv2.rectangle(disp, (l, t), (r, b), color, 3)
                        
                        lbl_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                        cv2.rectangle(disp, (l, b), (l + lbl_size[0] + 20, b + lbl_size[1] + 15), color, -1)
                        cv2.putText(disp, name, (l + 10, b + lbl_size[1] + 8), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.imshow('Drishya AI', disp)
            
            cv2.waitKey(1)
            time.sleep(0.03)  # ~30 FPS
        except:
            time.sleep(0.1)

# ==================== MAIN PROCESSING LOOP ====================
def main_loop():
    """Main processing loop - continuously updates frame"""
    global current_frame, current_locs, current_names
    
    last_known = 0
    last_unknown = 0
    
    while active:
        try:
            with cam_lock:
                if not cap or not cap.isOpened():
                    time.sleep(0.1)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
            
            # Detect faces
            locs, encs = detect_faces(frame)
            
            now = time.time()
            names = []
            known_count = 0
            unknown_count = 0
            
            for (t, r, b, l), enc in zip(locs, encs):
                idx = match_face(enc)
                
                if idx is not None:
                    name = known_names[idx]
                    known_count += 1
                    
                    # Greet known persons
                    if name not in last_greet or (now - last_greet[name]) > COOLDOWN:
                        last_greet[name] = now
                        speak(f"Hello {name}. Welcome back.", block=False)
                        beep()
                        
                        if name in known_cards:
                            info = known_cards[name]
                            time.sleep(0.3)
                            speak(f"Verified using {info['card_type']} card.", block=False)
                            beep()
                else:
                    name = "Unknown"
                    unknown_count += 1
                
                names.append(name)
            
            # Update display frame
            with frame_lock:
                current_frame = frame
                current_locs = locs
                current_names = names
            
            # Announce if counts changed
            if (known_count != last_known or unknown_count != last_unknown):
                last_known = known_count
                last_unknown = unknown_count
                announce_counts(known_count, unknown_count)
            
            time.sleep(0.1)  # Process at ~10 Hz
        
        except Exception as e:
            print(f"⚠️ Loop error: {e}")
            time.sleep(0.5)

# ==================== SHUTDOWN ====================
def shutdown():
    """Shutdown system"""
    global active, cap
    active = False
    speak("System shutting down. Goodbye.", block=True)
    time.sleep(1)
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# ==================== MAIN ====================
def main():
    """Main entry point - COMPLETE VERSION"""
    global cap, current_frame
    
    # Load database first
    print("📂 Loading face database...")
    load_db()
    
    # Open camera
    print("\n🎥 Opening camera...")
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap and cap.isOpened():
            print("✅ Camera opened successfully")
            break
    
    if not cap or not cap.isOpened():
        print("❌ Camera error - Cannot open camera")
        speak("Cannot open camera. Please check connection.", block=True)
        return
    
    # Configure camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create window and show first frame immediately
    cv2.namedWindow('Drishya AI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Drishya AI', 1024, 768)
    cv2.moveWindow('Drishya AI', 50, 50)
    
    # Capture and display first frame
    ret, first_frame = cap.read()
    if ret:
        current_frame = first_frame
        cv2.imshow('Drishya AI', first_frame)
        cv2.waitKey(1)
        print("✅ Camera feed visible on screen")
    
    # System information
    print("\n" + "="*70)
    print("🚀 DRISHYA AI - SYSTEM ACTIVE")
    print("="*70)
    print("\n✅ KEY FEATURES:")
    print("   • Camera continuously visible from start")
    print("   • Real-time person detection and counting")
    print("   • Dynamic updates (no freezing)")
    print("   • Background voice monitoring (non-intrusive)")
    print("   • Ultra-accurate name recognition")
    print("   • Grammar-aware announcements")
    print("   • Beep-based confirmation system")
    print("   • 120-frame capture with best quality")
    print("   • ID card verification (Aadhaar/PAN)")
    print("\n⌨️  Press Ctrl+C to exit")
    print("="*70 + "\n")
    
    # Start TTS announcements
    speak("Drishya AI system activated.", block=True)
    time.sleep(0.3)
    speak("Camera feed is now active.", block=True)
    
    # Start all system threads
    print("🔧 Starting system threads...")
    
    # Thread 1: Display thread (highest priority - continuous video)
    display_thread = threading.Thread(target=continuous_display, daemon=True, name="DisplayThread")
    display_thread.start()
    print("✅ Display thread started (30 FPS)")
    
    # Thread 2: Command processor
    cmd_thread = threading.Thread(target=process_commands, daemon=True, name="CommandProcessor")
    cmd_thread.start()
    print("✅ Command processor started")
    
    # Thread 3: Background voice listener
    voice_thread = threading.Thread(target=background_voice_listener, daemon=True, name="VoiceListener")
    voice_thread.start()
    print("✅ Background voice listener started")
    
    # Wait a moment for threads to initialize
    time.sleep(0.5)
    
    # Announce system ready
    speak("All systems operational.", block=True)
    time.sleep(0.3)
    speak("I am now monitoring continuously.", block=True)
    time.sleep(0.3)
    speak("Say register this person when you want to register someone.", block=True)
    
    print("\n" + "="*70)
    print("✅ ALL SYSTEMS READY - MONITORING ACTIVE")
    print("="*70)
    print("\n📊 System Status:")
    print(f"   • Display Thread: Running")
    print(f"   • Command Processor: Running")
    print(f"   • Voice Listener: Running")
    print(f"   • Face Database: {len(known_names)} person(s) loaded")
    print(f"   • Camera Resolution: 640x480")
    print(f"   • Face Detection: Active")
    print("\n🎤 Listening for commands...")
    print("="*70 + "\n")
    
    # Start main processing loop
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n🛑 Keyboard interrupt detected")
        shutdown()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        shutdown()

if __name__ == "__main__":
    main()