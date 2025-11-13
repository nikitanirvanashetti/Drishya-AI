"""
================================================================================
DRISHYA AI - ENHANCED LIVE REGISTRATION SYSTEM (FIXED)
================================================================================
✅ Smooth operation with proper coordination
✅ Accurate single-attempt name recognition
✅ No endless loops - proper state management
✅ Enhanced Google Speech API with multiple attempts
✅ Clean terminal output with proper timing

INSTALLATION:
pip install opencv-python face-recognition pyttsx3 SpeechRecognition pyaudio rapidfuzz easyocr numpy scipy Pillow

RUN:
python drishya_ai_enhanced.py
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
from collections import Counter
from queue import Queue

print("\n" + "="*70)
print("🎙️  DRISHYA AI - ENHANCED SYSTEM")
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
    print("⚠️  Install rapidfuzz for better name matching")

try:
    import easyocr
    print("✅ OCR (EasyOCR)")
    OCR = easyocr.Reader(['en'], gpu=False, verbose=False)
except:
    print("⚠️  OCR unavailable (pip install easyocr)")
    OCR = None

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
    print("✅ Audio Processing (SciPy)")
except:
    SCIPY_AVAILABLE = False
    print("⚠️  Install scipy for better audio quality")

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
SCENARIO_CHECK_INTERVAL = 10.0  # Increased to avoid spam
MIN_FRAMES_CAPTURE = 120
CARD_CAPTURE_DURATION = 10

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

# ==================== ENHANCED TTS ====================
def speak(text, force=True):
    """Thread-safe TTS with console output"""
    print(f"🔊 {text}")
    
    if not force:
        return
    
    def _speak():
        try:
            with tts_lock:
                engine = pyttsx3.init()
                engine.setProperty('rate', 155)
                engine.setProperty('volume', 1.0)
                
                voices = engine.getProperty('voices')
                if voices:
                    for v in voices:
                        if 'david' in v.name.lower() or 'male' in v.name.lower():
                            engine.setProperty('voice', v.id)
                            break
                    else:
                        engine.setProperty('voice', voices[0].id)
                
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
        except Exception as e:
            print(f"⚠️  TTS error: {e}")
    
    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    t.join()  # Wait for speech to complete

def beep():
    """Cross-platform beep"""
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
        speak("Face database initialized.")
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
                speak(f"Loaded {len(known_names)} registered persons from database.")
    except Exception as e:
        print(f"⚠️  Load error: {e}")
        speak("Error loading database.")

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
    except Exception as e:
        print(f"Error saving: {e}")
        return False

# ==================== FACE DETECTION ====================
def detect_faces(frame):
    """Detect faces and compute encodings"""
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
    """Match face encoding to known faces"""
    if not known_encodings:
        return None
    dists = face_recognition.face_distance(known_encodings, enc)
    idx = np.argmin(dists)
    return idx if dists[idx] < FACE_TOL else None

def calculate_sharpness(image):
    """Calculate image sharpness using Laplacian variance"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    except:
        return 0

# ==================== NAME MATCHING ====================
def normalize(name):
    """Normalize name for comparison"""
    if not name:
        return ""
    return ' '.join(re.sub(r'[^a-z\s]', '', name.lower()).split())

def names_match(n1, n2):
    """Check if two names match"""
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

# ==================== OCR WITH VALIDATION ====================
def is_image_blurry(frame, threshold=100):
    """Check if image is blurry"""
    sharpness = calculate_sharpness(frame)
    return sharpness < threshold, sharpness

def extract_card_info(frame):
    """Extract name and card type from ID card with validation"""
    if not OCR:
        return None, None, False, "OCR not available"
    
    try:
        # Check for blur
        is_blurry, sharpness = is_image_blurry(frame, threshold=100)
        if is_blurry:
            return None, None, True, f"Image is blurry"
        
        results = OCR.readtext(frame, paragraph=False)
        if not results:
            return None, None, False, "No text detected"
        
        text = " ".join([r[1] for r in results]).upper()
        
        # Detect card type
        card_type = None
        if any(k in text for k in ["AADHAAR", "AADHAR", "UID", "UNIQUE IDENTIFICATION"]):
            card_type = "AADHAAR"
        elif any(k in text for k in ["INCOME TAX", "PAN", "PERMANENT ACCOUNT"]):
            card_type = "PAN"
        
        if not card_type:
            if any(k in text for k in ["VOTER", "ELECTION", "DRIVING", "LICENSE", "LICENCE"]):
                return None, None, False, "Card type not accepted"
            return None, None, False, "Card not recognized"
        
        # Extract name
        lines = text.split('\n')
        potential_names = []
        
        for line in lines:
            words = line.strip().split()
            if 2 <= len(words) <= 4:
                if not any(c.isdigit() for c in line):
                    clean_line = re.sub(r'[^A-Z\s]', '', line.strip())
                    if len(clean_line) >= 5 and len(clean_line) <= 50:
                        potential_names.append(clean_line.title())
        
        if potential_names:
            return potential_names[0], card_type, False, "Success"
        
        return None, card_type, False, f"{card_type} detected, name unclear"
    
    except Exception as e:
        return None, None, False, f"Error: {str(e)}"

# ==================== ENHANCED VOICE RECOGNITION ====================
class EnhancedVoiceRecognition:
    """Enhanced voice recognition with multiple attempts for accuracy"""
    
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = 300
        self.rec.dynamic_energy_threshold = True
        self.rec.dynamic_energy_adjustment_damping = 0.15
        self.rec.dynamic_energy_ratio = 1.5
        self.rec.pause_threshold = 0.8
        self.rec.phrase_threshold = 0.3
        self.rec.non_speaking_duration = 0.5
    
    def listen_with_multiple_attempts(self, timeout=20, max_attempts=3):
        """
        Listen with multiple recognition attempts for highest accuracy
        Returns the most confident result
        """
        try:
            with sr.Microphone(sample_rate=48000) as source:
                print("\n🎤 Listening for your response...")
                
                # Adjust for ambient noise
                self.rec.adjust_for_ambient_noise(source, duration=1.0)
                
                try:
                    # Record audio
                    audio = self.rec.listen(source, timeout=timeout, phrase_time_limit=20)
                    
                    # Try multiple recognition engines for best accuracy
                    results = []
                    
                    # Attempt 1: Google India (Best for Indian accents)
                    try:
                        text = self.rec.recognize_google(audio, language='en-IN')
                        if text:
                            results.append(text.strip())
                            print(f"✅ Attempt 1 (India): {text.strip()}")
                    except:
                        pass
                    
                    # Attempt 2: Google US
                    try:
                        text = self.rec.recognize_google(audio, language='en-US')
                        if text:
                            results.append(text.strip())
                            print(f"✅ Attempt 2 (US): {text.strip()}")
                    except:
                        pass
                    
                    # Attempt 3: Google UK
                    try:
                        text = self.rec.recognize_google(audio, language='en-GB')
                        if text:
                            results.append(text.strip())
                            print(f"✅ Attempt 3 (UK): {text.strip()}")
                    except:
                        pass
                    
                    if not results:
                        print("❌ No speech detected")
                        return ""
                    
                    # If all results are similar, return the first one
                    if len(results) == 1:
                        return results[0]
                    
                    # Find most common result (voting)
                    if FUZZY:
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
                        
                        # Return most common
                        if groups:
                            best = max(groups, key=lambda x: x['count'])
                            print(f"🎯 Final result: {best['text']} (confidence: {best['count']}/{len(results)})")
                            return best['text']
                    
                    # Fallback: return first result
                    return results[0]
                
                except sr.WaitTimeoutError:
                    print("⏱️  Timeout - no speech detected")
                    return ""
        
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return ""

mic = EnhancedVoiceRecognition()

# ==================== SCENARIO DETECTION ====================
def announce_scenario(known_count, unknown_count):
    """Announce current scenario to blind user (only once per state change)"""
    global last_scenario_state, last_scenario_time, scenario_announced, registration_in_progress
    
    current_state = (known_count, unknown_count)
    now = time.time()
    
    # Skip if same scenario was just announced
    if current_state == last_scenario_state and scenario_announced:
        return
    
    # Skip if in registration
    if registration_in_progress:
        return
    
    # Skip if not enough time passed
    if (now - last_scenario_time) < SCENARIO_CHECK_INTERVAL:
        return
    
    last_scenario_state = current_state
    last_scenario_time = now
    scenario_announced = True
    
    total = known_count + unknown_count
    
    if total == 0:
        speak("No persons detected in front of you.")
        scenario_announced = False
        return
    
    # Build announcement
    if known_count == 0 and unknown_count > 0:
        if unknown_count == 1:
            speak("There is 1 unknown person in front of you.")
        else:
            speak(f"There are {unknown_count} unknown persons in front of you.")
        
        time.sleep(1)  # Wait 1 second before asking
        speak("Shall I register the unknown persons with their names?")
        time.sleep(0.5)
        speak("Please say yes to proceed or no to skip.")
        time.sleep(0.3)
        speak("Please respond after the beep.")
        beep()
        time.sleep(0.5)  # Give user time to prepare after beep
        response = mic.listen_with_multiple_attempts(timeout=15)
        
        if response and any(word in response.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'proceed', 'register', 'start']):
            speak("Understood. Let's proceed with the registration of unknown persons.")
            time.sleep(0.5)
            threading.Thread(target=do_registration, daemon=True).start()
        else:
            speak("Registration skipped. I will continue monitoring.")
            scenario_announced = False  # Allow re-announcement later
    
    elif unknown_count == 0 and known_count > 0:
        if known_count == 1:
            speak("There is 1 known person in front of you.")
        else:
            speak(f"There are {known_count} known persons in front of you.")
        scenario_announced = False
    
    else:
        speak(f"There are {total} persons in front of you.")
        time.sleep(0.3)
        speak(f"{known_count} known and {unknown_count} unknown.")
        time.sleep(1)
        speak("Shall I register the unknown persons?")
        time.sleep(0.5)
        speak("Please say yes or no.")
        time.sleep(0.3)
        speak("Please respond after the beep.")
        beep()
        time.sleep(0.5)
        response = mic.listen_with_multiple_attempts(timeout=15)
        
        if response and any(word in response.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'proceed']):
            speak("Let's proceed with registration.")
            threading.Thread(target=do_registration, daemon=True).start()
        else:
            speak("Registration skipped.")
            scenario_announced = False

# ==================== REGISTRATION PROCESS ====================
def do_registration():
    """Complete registration process with 120-frame capture"""
    global is_registering, registration_in_progress, scenario_announced
    
    if is_registering or registration_in_progress:
        return
    
    is_registering = True
    registration_in_progress = True
    scenario_announced = True  # Prevent re-announcement during registration
    
    try:
        speak("Starting registration process.")
        time.sleep(0.5)
        
        # Step 1: Get name
        speak("Please introduce yourself and tell me your full name clearly.")
        time.sleep(0.5)
        speak("Please tell me your name after the beep.")
        beep()
        time.sleep(0.5)  # Give user time to prepare after beep
        
        name_heard = mic.listen_with_multiple_attempts(timeout=25, max_attempts=3)
        
        if not name_heard:
            speak("I did not hear your name. Registration cancelled.")
            return
        
        # Clean and format name
        name_heard = ' '.join(word.capitalize() for word in name_heard.split())
        
        speak(f"I heard your name as {name_heard}.")
        time.sleep(0.5)
        speak("Please confirm. Is this your correct name?")
        time.sleep(0.3)
        speak("Say yes to confirm or repeat your name.")
        time.sleep(0.3)
        speak("Please respond after the beep.")
        beep()
        time.sleep(0.5)
        
        confirmation = mic.listen_with_multiple_attempts(timeout=25)
        
        if not confirmation:
            speak("No confirmation received. Registration cancelled.")
            return
        
        # Check confirmation
        conf_lower = confirmation.lower()
        if "yes" in conf_lower or "correct" in conf_lower or "right" in conf_lower or "yeah" in conf_lower:
            speak(f"Name confirmed as {name_heard}.")
        else:
            # User repeated the name
            confirmation_clean = ' '.join(word.capitalize() for word in confirmation.split())
            speak(f"I heard {confirmation_clean}. Using this as your name.")
            name_heard = confirmation_clean
        
        time.sleep(0.5)
        
        # Step 2: Capture face with 120 frames
        speak("Now I will capture your face.")
        time.sleep(0.3)
        speak("Please look directly at the camera and stay still.")
        time.sleep(0.5)
        speak(f"I will capture a minimum of {MIN_FRAMES_CAPTURE} frames to get the best quality image.")
        time.sleep(1)
        speak("Starting face capture now. Please remain still.")
        time.sleep(0.5)
        
        frames_data = []
        frame_count = 0
        start_time = time.time()
        last_update = 0
        
        while frame_count < MIN_FRAMES_CAPTURE and (time.time() - start_time) < 30:
            with cam_lock:
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        locs, encs = detect_faces(frame)
                        if locs and encs:
                            # Calculate sharpness
                            sharpness = calculate_sharpness(frame)
                            frames_data.append({
                                'frame': frame.copy(),
                                'encoding': encs[0],
                                'location': locs[0],
                                'sharpness': sharpness
                            })
                            frame_count += 1
                            
                            # Speak progress every 30 frames
                            if frame_count % 30 == 0 and (time.time() - last_update) > 2:
                                print(f"📸 Captured {frame_count} frames...")
                                last_update = time.time()
            
            time.sleep(0.05)
        
        if len(frames_data) < MIN_FRAMES_CAPTURE:
            speak(f"Could not capture enough frames. Only captured {len(frames_data)} frames.")
            speak("Please ensure your face is clearly visible and try again.")
            return
        
        speak(f"Successfully captured {len(frames_data)} frames.")
        time.sleep(0.5)
        speak("Now selecting the best quality image.")
        
        # Select best frame
        best_frame_data = max(frames_data, key=lambda x: x['sharpness'])
        best_frame = best_frame_data['frame']
        best_sharpness = best_frame_data['sharpness']
        
        print(f"✅ Best frame sharpness: {best_sharpness:.2f}")
        
        # Save photo
        photos_dir = "registered_photos"
        os.makedirs(photos_dir, exist_ok=True)
        
        photo_filename = f"{name_heard.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        photo_path = os.path.join(photos_dir, photo_filename)
        
        cv2.imwrite(photo_path, best_frame)
        speak("Best quality frame selected and saved.")
        
        # Calculate average encoding
        all_encodings = [fd['encoding'] for fd in frames_data]
        avg_encoding = np.mean(all_encodings, axis=0)
        
        time.sleep(0.5)
        
        # Step 3: ID Card verification
        speak("Now I need to verify your identity with your ID card.")
        time.sleep(0.3)
        speak("Please show your Aadhaar card or PAN card to the camera.")
        time.sleep(0.3)
        speak("Note: I only accept Aadhaar or PAN cards.")
        time.sleep(0.5)
        speak("Please hold the card clearly in front of the camera with good lighting.")
        time.sleep(1)
        speak("Starting card capture. Please hold your card steady.")
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
                                speak("Card image is blurry or shaking. Please hold it steady.")
                                last_feedback = time.time()
                        elif type_extracted and not name_extracted:
                            if (time.time() - last_feedback) > 3:
                                speak(f"{type_extracted} card detected but name is not clear.")
                                last_feedback = time.time()
                        elif name_extracted and type_extracted:
                            card_name = name_extracted
                            card_type = type_extracted
                            speak(f"{card_type} card detected successfully.")
                            break
                        elif "not accepted" in message:
                            if (time.time() - last_feedback) > 3:
                                speak("This card type is not accepted. Please show Aadhaar or PAN card only.")
                                last_feedback = time.time()
            
            attempts += 1
            time.sleep(0.2)
        
        if not card_name or not card_type:
            speak("Could not read your ID card clearly after multiple attempts.")
            speak("Registration cancelled. Please try again with better lighting.")
            return
        
        time.sleep(0.5)
        speak(f"Card read successfully. The card type is {card_type}.")
        time.sleep(0.5)
        speak(f"Name on the card is {card_name}.")
        time.sleep(0.5)
        
        # Verify name match
        if not names_match(name_heard, card_name):
            speak(f"Name mismatch detected.")
            speak(f"You said {name_heard}, but the card shows {card_name}.")
            speak("Registration cancelled.")
            return
        
        speak("Name verified successfully. The name matches perfectly.")
        beep()
        time.sleep(0.5)
        
        # Save to database
        speak("Saving your information to the system database.")
        
        if save_face(name_heard, avg_encoding, card_type, card_name, photo_path):
            load_db()
            speak(f"Registration completed successfully for {name_heard}.")
            beep()
            time.sleep(0.5)
            speak(f"Your {card_type} card has been verified and stored securely.")
            time.sleep(0.5)
            speak("Your face has been registered with the best quality photo.")
            time.sleep(0.5)
            speak(f"{name_heard} is now fully verified and authenticated.")
            beep()
            time.sleep(0.5)
            speak("Registration process complete. Welcome to the system.")
        else:
            speak("Failed to save your information. Please try again later.")
    
    except Exception as e:
        print(f"❌ Registration error: {e}")
        speak("An error occurred during registration. Please try again.")
    
    finally:
        is_registering = False
        registration_in_progress = False
        scenario_announced = False  # Allow new announcements

# ==================== CAMERA DISPLAY ====================
def show_frame(frame, locs, names):
    """Display frame with bounding boxes"""
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
    """Main processing loop with continuous monitoring"""
    global last_scenario_time, scenario_announced
    
    last_known = 0
    last_unknown = 0
    
    while active:
        try:
            time.sleep(0.5)  # Slower polling to reduce CPU
            
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
                    
                    # Greet known persons (with cooldown)
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
            
            # Only announce if counts changed
            if (known_count != last_known or unknown_count != last_unknown):
                last_known = known_count
                last_unknown = unknown_count
                scenario_announced = False  # Reset flag
                announce_scenario(known_count, unknown_count)
            
            show_frame(frame, locs, names)
        
        except Exception as e:
            print(f"⚠️  Loop error: {e}")

# ==================== SHUTDOWN ====================
def shutdown():
    """Shutdown system"""
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
    """Main entry point"""
    global cap
    
    load_db()
    
    print("\n" + "="*70)
    print("🚀 STARTING DRISHYA AI - ENHANCED LIVE SYSTEM")
    print("="*70 + "\n")
    
    speak("Drishya AI system activated.")
    time.sleep(0.5)
    speak("Opening camera for continuous monitoring.")
    
    # Open camera
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap and cap.isOpened():
            print(f"✅ Camera opened")
            break
    
    if not cap or not cap.isOpened():
        speak("Camera error. Cannot open camera. Please check your camera connection.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    speak("Camera opened successfully.")
    
    cv2.namedWindow('Drishya AI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Drishya AI', 1024, 768)
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY - CONTINUOUS MONITORING ACTIVE")
    print("="*70)
    print("\n📋 FEATURES:")
    print("   ✓ Automatic scenario detection (no loops)")
    print("   ✓ Permission-based registration workflow")
    print("   ✓ 120-frame capture with best quality selection")
    print("   ✓ Enhanced voice recognition with multiple attempts")
    print("   ✓ Real-time ID card validation")
    print("   ✓ Complete voice feedback")
    print("   ✓ Smooth operation with proper timing")
    print("\n⌨️  Press Ctrl+C to exit")
    print("="*70 + "\n")
    
    speak("System is now active and monitoring continuously.")
    time.sleep(0.5)
    speak("I will announce when I detect persons in front of you.")
    
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n\n🛑 Keyboard interrupt detected")
        shutdown()

if __name__ == "__main__":
    main()