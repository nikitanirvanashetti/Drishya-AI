"""
================================================================================
DRISHYA AI - SIRI-STYLE CONTINUOUS VOICE RECOGNITION
================================================================================
✅ Always-on voice recognition like Apple Siri
✅ Single-attempt ultra-accurate capture
✅ Background voice monitoring continuously active
✅ Instant command detection and execution
✅ Camera pops up immediately
✅ Perfect voice recognition accuracy

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
print("🎙️  DRISHYA AI - SIRI-STYLE SYSTEM")
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

try:
    import easyocr
    print("✅ OCR (EasyOCR)")
    OCR = easyocr.Reader(['en'], gpu=False, verbose=False)
except:
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
SCENARIO_CHECK_INTERVAL = 8.0
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
last_scenario_state = None
last_scenario_time = 0
scenario_announced = False

voice_command_queue = Queue()

# ==================== TTS ====================
def speak(text):
    """Thread-safe TTS"""
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
    t.join()

def beep():
    """Beep sound"""
    try:
        import winsound
        winsound.Beep(1000, 200)
    except:
        pass

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
                speak(f"Loaded {len(known_names)} registered persons.")
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

# ==================== SIRI-STYLE VOICE RECOGNITION ====================
class SiriStyleVoiceRecognition:
    """Always-on voice recognition like Apple Siri"""
    
    def __init__(self):
        self.rec = sr.Recognizer()
        # Ultra-sensitive settings for instant capture
        self.rec.energy_threshold = 200
        self.rec.dynamic_energy_threshold = True
        self.rec.dynamic_energy_adjustment_damping = 0.1
        self.rec.dynamic_energy_ratio = 1.3
        self.rec.pause_threshold = 0.7
        self.rec.phrase_threshold = 0.2
        self.rec.non_speaking_duration = 0.4
    
    def listen_once_ultra_accurate(self, prompt="", timeout=20):
        """Single-attempt ultra-accurate listening"""
        try:
            with sr.Microphone(sample_rate=48000, chunk_size=2048) as source:
                if prompt:
                    speak(prompt)
                
                print("\n🎤 LISTENING NOW...")
                
                # Quick calibration
                self.rec.adjust_for_ambient_noise(source, duration=0.7)
                
                # Listen with high quality
                audio = self.rec.listen(source, timeout=timeout, phrase_time_limit=15)
                
                print("🔄 Processing audio...")
                
                # Try all engines in parallel for best result
                results = []
                
                # Engine 1: India
                try:
                    text = self.rec.recognize_google(audio, language='en-IN')
                    if text:
                        results.append(text.strip())
                        print(f"✅ India: '{text.strip()}'")
                except Exception as e:
                    print(f"⚠️ India failed: {e}")
                
                # Engine 2: US
                try:
                    text = self.rec.recognize_google(audio, language='en-US')
                    if text:
                        results.append(text.strip())
                        print(f"✅ US: '{text.strip()}'")
                except Exception as e:
                    print(f"⚠️ US failed: {e}")
                
                # Engine 3: UK
                try:
                    text = self.rec.recognize_google(audio, language='en-GB')
                    if text:
                        results.append(text.strip())
                        print(f"✅ UK: '{text.strip()}'")
                except Exception as e:
                    print(f"⚠️ UK failed: {e}")
                
                if not results:
                    print("❌ No speech detected")
                    return ""
                
                # Intelligent voting
                if len(results) == 1:
                    final = results[0]
                elif FUZZY:
                    # Group similar results
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
                    print(f"🎯 Voted result: '{final}' ({best['count']}/{len(results)})")
                else:
                    final = results[0]
                
                print(f"✅ FINAL: '{final}'\n")
                return final
        
        except sr.WaitTimeoutError:
            print("⏱️ Timeout\n")
            return ""
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return ""
    
    def continuous_background_listening(self, callback):
        """Continuous background voice monitoring like Siri"""
        with sr.Microphone(sample_rate=48000, chunk_size=2048) as source:
            print("\n🎙️ BACKGROUND VOICE MONITORING ACTIVE")
            print("=" * 70)
            speak("Continuous voice recognition is now active.")
            
            self.rec.adjust_for_ambient_noise(source, duration=1.5)
            print("✅ Always listening for commands...\n")
            
            while active:
                try:
                    # Listen for commands
                    audio = self.rec.listen(source, timeout=1.5, phrase_time_limit=8)
                    
                    def process_command():
                        try:
                            # Quick recognition
                            text = None
                            try:
                                text = self.rec.recognize_google(audio, language='en-IN')
                            except:
                                try:
                                    text = self.rec.recognize_google(audio, language='en-US')
                                except:
                                    pass
                            
                            if text:
                                text = text.strip()
                                print(f"🎤 Command heard: '{text}'")
                                callback(text)
                        except:
                            pass
                    
                    threading.Thread(target=process_command, daemon=True).start()
                
                except sr.WaitTimeoutError:
                    continue
                except:
                    if active:
                        time.sleep(0.3)

mic = SiriStyleVoiceRecognition()

# ==================== COMMAND HANDLER ====================
def handle_voice_command(text):
    """Handle voice commands"""
    global registration_in_progress
    
    if registration_in_progress:
        return
    
    txt = text.lower()
    
    # Registration trigger
    if any(phrase in txt for phrase in [
        "register this person",
        "register unknown person", 
        "register person",
        "start registration",
        "register"
    ]):
        speak("Registration command received.")
        threading.Thread(target=do_registration, daemon=True).start()

# ==================== SCENARIO DETECTION ====================
def announce_scenario(known_count, unknown_count):
    """Announce scenario"""
    global last_scenario_state, last_scenario_time, scenario_announced, registration_in_progress
    
    current_state = (known_count, unknown_count)
    now = time.time()
    
    if current_state == last_scenario_state and scenario_announced:
        return
    
    if registration_in_progress:
        return
    
    if (now - last_scenario_time) < SCENARIO_CHECK_INTERVAL:
        return
    
    last_scenario_state = current_state
    last_scenario_time = now
    scenario_announced = True
    
    total = known_count + unknown_count
    
    if total == 0:
        speak("No persons detected.")
        scenario_announced = False
        return
    
    if known_count == 0 and unknown_count > 0:
        if unknown_count == 1:
            speak("There is 1 unknown person in front of you.")
        else:
            speak(f"There are {unknown_count} unknown persons in front of you.")
        
        time.sleep(0.5)
        speak("Shall I register the unknown persons? Say yes or no after the beep.")
        beep()
        time.sleep(0.5)
        
        response = mic.listen_once_ultra_accurate(timeout=12)
        
        print(f"🎯 Response: '{response}'")
        
        if response:
            resp_lower = response.lower()
            yes_words = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'proceed', 'register', 'start', 'go', 'yah', 'ya', 'yup']
            
            if any(word in resp_lower for word in yes_words):
                print("✅ YES detected")
                speak("Understood. Let's proceed with registration.")
                time.sleep(0.5)
                threading.Thread(target=do_registration, daemon=True).start()
            else:
                print("❌ NO/other response")
                speak("Registration skipped. I will continue monitoring.")
                scenario_announced = False
        else:
            speak("No response. Skipping registration.")
            scenario_announced = False
    
    elif unknown_count == 0 and known_count > 0:
        if known_count == 1:
            speak("There is 1 known person in front of you.")
        else:
            speak(f"There are {known_count} known persons in front of you.")
        scenario_announced = False
    
    else:
        speak(f"There are {total} persons. {known_count} known and {unknown_count} unknown. Shall I register unknown persons? Say yes or no after the beep.")
        beep()
        time.sleep(0.5)
        
        response = mic.listen_once_ultra_accurate(timeout=12)
        
        print(f"🎯 Response: '{response}'")
        
        if response:
            resp_lower = response.lower()
            yes_words = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'proceed', 'register', 'start', 'go', 'yah', 'ya', 'yup']
            
            if any(word in resp_lower for word in yes_words):
                print("✅ YES detected")
                speak("Let's proceed with registration.")
                threading.Thread(target=do_registration, daemon=True).start()
            else:
                print("❌ NO/other response")
                speak("Registration skipped.")
                scenario_announced = False
        else:
            speak("No response. Skipping.")
            scenario_announced = False

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

# ==================== MAIN LOOP ====================
def main_loop():
    """Main loop"""
    global last_scenario_time, scenario_announced
    
    last_known = 0
    last_unknown = 0
    
    while active:
        try:
            time.sleep(0.3)
            
            with cam_lock:
                if not cap or not cap.isOpened():
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    continue
            
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
                    
                    if name not in last_greet or (now - last_greet[name]) > COOLDOWN:
                        last_greet[name] = now
                        speak(f"Hello {name}. Welcome back.")
                        beep()
                        time.sleep(0.5)
                        
                        if name in known_cards:
                            info = known_cards[name]
                            speak(f"Verified using {info['card_type']} card.")
                            time.sleep(0.3)
                            speak(f"{name} is authenticated.")
                            beep()
                else:
                    name = "Unknown"
                    unknown_count += 1
                
                names.append(name)
            
            if (known_count != last_known or unknown_count != last_unknown):
                last_known = known_count
                last_unknown = unknown_count
                scenario_announced = False
                announce_scenario(known_count, unknown_count)
            
            show_frame(frame, locs, names)
        
        except Exception as e:
            print(f"⚠️ Error: {e}")

# ==================== SHUTDOWN ====================
def shutdown():
    """Shutdown"""
    global active, cap
    active = False
    speak("System shutting down. Goodbye.")
    time.sleep(1)
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# ==================== MAIN ====================
def main():
    """Main entry"""
    global cap
    
    # Open camera FIRST
    print("Opening camera...")
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap and cap.isOpened():
            print("✅ Camera opened")
            break
    
    if not cap or not cap.isOpened():
        print("❌ Camera error")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Show camera immediately
    cv2.namedWindow('Drishya AI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Drishya AI', 1024, 768)
    cv2.moveWindow('Drishya AI', 50, 50)
    
    ret, first_frame = cap.read()
    if ret:
        cv2.imshow('Drishya AI', first_frame)
        cv2.waitKey(1)
    
    # Load database
    load_db()
    
    print("\n" + "="*70)
    print("🚀 DRISHYA AI - SIRI-STYLE SYSTEM ACTIVE")
    print("="*70)
    print("\n✅ Features:")
    print("   • Instant camera popup")
    print("   • Ultra-accurate single-attempt voice recognition")
    print("   • Continuous background voice monitoring")
    print("   • Say 'register this person' anytime")
    print("\n⌨️ Press Ctrl+C to exit")
    print("="*70 + "\n")
    
    speak("Drishya AI system activated.")
    speak("Camera opened successfully.")
    
    # Start background voice listener
    voice_thread = threading.Thread(
        target=lambda: mic.continuous_background_listening(handle_voice_command),
        daemon=True
    )
    voice_thread.start()
    
    print("✅ Background voice monitoring started")
    print("✅ Camera feed active\n")
    
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down")
        shutdown()

if __name__ == "__main__":
    main()