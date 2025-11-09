# Drishya AI - Enhanced Production System with Smart Verification & GUI Switch
# Run Instructions:
# 1. UNINSTALL opencv-python: pip uninstall opencv-python opencv-contrib-python -y
# 2. INSTALL correct version: pip install opencv-contrib-python
# 3. Install other dependencies: pip install face_recognition dlib pyttsx3 SpeechRecognition pyaudio numpy easyocr pillow pytesseract
# 4. Save unified_name_extractor.py in the same folder
# 5. python newmain.py

import cv2
import face_recognition
import numpy as np
import pyttsx3
import speech_recognition as sr
import json
import os
import time
import threading
from collections import deque, defaultdict
import sys
import logging
import re
from datetime import datetime
import subprocess
import webbrowser

# Import the unified name extractor
from unified_name_extractor import UnifiedNameExtractor
extractor = UnifiedNameExtractor()

# FORCE SET TESSERACT PATH
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(TESSERACT_PATH):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[SUCCESS] ✅ Tesseract {version} configured successfully")
        TESSERACT_AVAILABLE = True
    except:
        print("[WARNING] ⚠️ Tesseract found but not working properly")
        TESSERACT_AVAILABLE = False
else:
    print(f"[ERROR] ❌ Tesseract not found at: {TESSERACT_PATH}")
    print("Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[ERROR] easyocr not installed. Run: pip install easyocr")
    sys.exit(1)

# -------------------- CONFIGURATION --------------------
JSON_FILE = "faces.json"
ID_FOLDER = "verified_ids"
TOLERANCE = 0.35
MIN_MARGIN = 0.12
STABILITY_WINDOW = 20
STABILITY_THRESHOLD = 0.75
MIN_VISIBLE_TIME = 2.5
COOLDOWN_IDENTITY = 10.0
MAX_FACES = 5
IOU_THRESHOLD = 0.45
CAPTURE_FRAMES = 120
SELECT_BEST = 10
CAMERA_INDEX = 0
NO_FACE_TIMEOUT = 30

os.makedirs(ID_FOLDER, exist_ok=True)

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drishya.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------- SPEECH --------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
speech_lock = threading.Lock()

speech_text = ""
speech_display_time = 0

def speak(text, duration=3.0):
    """Thread-safe speech with overlay display"""
    global speech_text, speech_display_time
    logger.info(f"[SPEAK]: {text}")
    speech_text = text
    speech_display_time = time.time() + duration
    
    with speech_lock:
        try:
            engine.say(text)
            engine.runAndWait()
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"[SPEECH ERROR]: {e}")

# -------------------- VOICE RECOGNITION --------------------
def listen_for_response(prompt, timeout=8, check_gui_switch=False):
    """Listen for speech with optional GUI switch detection"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    
    speak(prompt, duration=4.0)
    time.sleep(0.3)
    
    for attempt in range(2):
        try:
            with sr.Microphone() as source:
                if attempt == 0:
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                logger.info("[LISTENING]")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)
                
                try:
                    text = recognizer.recognize_google(audio, language='en-IN')
                    logger.info(f"[GOOGLE]: '{text}'")
                    
                    # Check for GUI switch command
                    if check_gui_switch:
                        text_lower = text.lower()
                        gui_keywords = ['switch to gui', 'open gui', 'show gui', 'gui mode', 
                                       'switch gui', 'open interface', 'web interface']
                        if any(keyword in text_lower for keyword in gui_keywords):
                            return "SWITCH_TO_GUI"
                    
                    return text.strip()
                except:
                    try:
                        text = recognizer.recognize_sphinx(audio)
                        logger.info(f"[SPHINX]: '{text}'")
                        return text.strip()
                    except:
                        pass
                
                if attempt == 0:
                    speak("Could not hear. Try again.", duration=2.0)
                    
        except sr.WaitTimeoutError:
            if attempt == 0:
                speak("No response. Try again.", duration=2.0)
        except Exception as e:
            logger.error(f"[LISTEN ERROR]: {e}")
    
    return ""

def clean_name(text):
    """Clean spoken name"""
    if not text:
        return ""
    
    text = text.strip()
    filler_patterns = ['my name is', 'i am', 'this is', "i'm", 'call me']
    text_lower = text.lower()
    
    for pattern in filler_patterns:
        if text_lower.startswith(pattern):
            text = text[len(pattern):].strip()
            break
    
    filler_words = ['my', 'name', 'is', 'am', 'i', 'the', 'a', 'an']
    words = [w for w in text.split() if w.lower() not in filler_words]
    
    return ' '.join(words).title() if words else text.title()

# -------------------- DATABASE --------------------
def load_faces():
    """Load face database - FIXED VERSION"""
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
            
            # FIXED: Proper verification check
            card_type = entry.get('card_type')
            full_name = entry.get('full_name')
            is_verified = bool(card_type and full_name)
            verification_map[name] = is_verified
        
        logger.info(f"[DATABASE] Loaded {len(names)} faces, {sum(verification_map.values())} verified")
        return encodings, names, verification_map
        
    except Exception as e:
        logger.error(f"[LOAD ERROR]: {e}")
        return [], [], {}

def save_face(name, encoding, card_type, full_name):
    """Save face to database with verification details"""
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
            'verified': True
        })
        
        temp_file = JSON_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        os.replace(temp_file, JSON_FILE)
        logger.info(f"[SAVE SUCCESS] {name} - Verified with {card_type}")
        return True
    except Exception as e:
        logger.error(f"[SAVE ERROR]: {e}")
        return False

def update_verification_status(name, card_type, full_name):
    """Update existing person's verification status"""
    try:
        if not os.path.exists(JSON_FILE) or os.path.getsize(JSON_FILE) == 0:
            return False
        
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        
        updated = False
        for entry in data:
            if entry['name'].lower() == name.lower():
                entry['card_type'] = card_type
                entry['full_name'] = full_name
                entry['verified'] = True
                entry['verification_timestamp'] = datetime.now().isoformat()
                updated = True
                break
        
        if updated:
            temp_file = JSON_FILE + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, JSON_FILE)
            logger.info(f"[UPDATE SUCCESS] {name} verified with {card_type}")
            return True
        
        logger.warning(f"[UPDATE FAILED] Name '{name}' not found in database")
        return False
    except Exception as e:
        logger.error(f"[UPDATE ERROR]: {e}")
        return False

def normalize_name(name):
    """Normalize name for comparison"""
    if not name:
        return ""
    normalized = re.sub(r'[^a-z\s]', '', name.lower().strip())
    normalized = ' '.join(normalized.split())
    return normalized

def names_match(name1, name2, threshold=95):
    """Check if two names match with fuzzy matching"""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    if not norm1 or not norm2:
        return False
    
    if norm1 == norm2:
        return True
    
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if words1 & words2:
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = (intersection / union) * 100
        return similarity >= threshold
    
    return False

def find_matching_verified_name(spoken_name):
    """Find if spoken name matches any verified name in database"""
    try:
        if not os.path.exists(JSON_FILE) or os.path.getsize(JSON_FILE) == 0:
            return None, None
        
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            if not entry.get('verified', False):
                continue
            
            db_spoken_name = entry.get('name', '')
            db_full_name = entry.get('full_name', '')
            
            if names_match(spoken_name, db_spoken_name) or names_match(spoken_name, db_full_name):
                return db_spoken_name, db_full_name
        
        return None, None
    except Exception as e:
        logger.error(f"[FIND VERIFIED NAME ERROR]: {e}")
        return None, None

# -------------------- GUI SERVER CONTROL --------------------
gui_server_process = None

def start_gui_server():
    """Start the GUI server in background"""
    global gui_server_process
    
    try:
        if gui_server_process is not None:
            logger.info("[GUI] Server already running")
            return True
        
        logger.info("[GUI] Starting web server...")
        speak("Starting web interface. Please wait.", duration=3.0)
        
        gui_server_process = subprocess.Popen(
            [sys.executable, 'gui_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
        )
        
        time.sleep(3)
        
        if gui_server_process.poll() is None:
            logger.info("[GUI] Server started successfully")
            speak("Web interface is ready. Opening in browser.", duration=3.0)
            
            time.sleep(1)
            webbrowser.open('http://localhost:5000')
            
            speak("You can now use the web interface on your computer or phone.", duration=4.0)
            return True
        else:
            logger.error("[GUI] Server failed to start")
            speak("Failed to start web interface.", duration=2.0)
            return False
            
    except Exception as e:
        logger.error(f"[GUI START ERROR]: {e}")
        speak("Error starting web interface.", duration=2.0)
        return False

def stop_gui_server():
    """Stop the GUI server"""
    global gui_server_process
    
    if gui_server_process is not None:
        try:
            gui_server_process.terminate()
            gui_server_process.wait(timeout=5)
            gui_server_process = None
            logger.info("[GUI] Server stopped")
        except:
            if gui_server_process is not None:
                gui_server_process.kill()
                gui_server_process = None

# -------------------- ID VERIFICATION CLASS --------------------
class IDVerifier:
    """Enhanced ID verification with improved card detection"""
    
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("[ID VERIFY] EasyOCR initialized")
        except Exception as e:
            logger.error(f"[ID VERIFY INIT ERROR]: {e}")
            raise
        
        self.verified_cards = []
    
    def calculate_sharpness(self, image):
        """Calculate image sharpness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def preprocess_for_mobile_display(self, image):
        """Enhanced preprocessing for mobile screen/document detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(binary)
        contrast = cv2.convertScaleAbs(denoised, alpha=1.8, beta=20)
        return contrast
    
    def detect_screen_glare(self, image):
        """Detect if image has screen glare/reflections"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > 240)
        total_pixels = gray.size
        glare_ratio = bright_pixels / total_pixels
        return glare_ratio > 0.15
    
    def enhance_mobile_id(self, image):
        """Special enhancement for IDs shown on mobile screens"""
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
        """Capture ID card - supports physical cards AND mobile displays"""
        speak(card_prompt, duration=3.0)
        speak("You can show physical card, or display it on your mobile phone screen, or show a printed document.", duration=5.0)
        time.sleep(1)
        
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < 15 and len(frames) < 40:
            ret, frame = cap.read()
            if not ret:
                continue
            
            display_frame = frame.copy()
            cv2.putText(display_frame, "Capturing ID Card...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "Physical/Mobile/Document OK", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frames: {len(frames)}/40", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            try:
                cv2.imshow("Drishya AI - ID Verification", display_frame)
                cv2.waitKey(1)
            except:
                pass
            
            sharpness = self.calculate_sharpness(frame)
            has_glare = self.detect_screen_glare(frame)
            
            if sharpness > 80 or (sharpness > 50 and not has_glare):
                frames.append((frame.copy(), sharpness))
                logger.info(f"[ID CAPTURE] Frame {len(frames)}/40 (sharpness={sharpness:.1f})")
            
            time.sleep(0.25)
        
        if not frames:
            return None
        
        best_frame = max(frames, key=lambda x: x[1])[0]
        logger.info(f"[ID CAPTURE] Selected best frame")
        return best_frame
    
    def extract_text_multiple_methods(self, image):
        """Use multiple OCR methods for robust extraction"""
        all_text = []
        
        try:
            results = self.reader.readtext(image, paragraph=False)
            text1 = " ".join([r[1] for r in results])
            all_text.append(text1)
            logger.info(f"[OCR-Original]: {text1[:150]}")
        except Exception as e:
            logger.error(f"[OCR-Original Error]: {e}")
        
        try:
            enhanced = self.enhance_mobile_id(image)
            results = self.reader.readtext(enhanced, paragraph=False)
            text2 = " ".join([r[1] for r in results])
            all_text.append(text2)
            logger.info(f"[OCR-Enhanced]: {text2[:150]}")
        except Exception as e:
            logger.error(f"[OCR-Enhanced Error]: {e}")
        
        if TESSERACT_AVAILABLE:
            try:
                preprocessed = self.preprocess_for_mobile_display(image)
                text3 = pytesseract.image_to_string(preprocessed, lang='eng')
                all_text.append(text3)
                logger.info(f"[OCR-Tesseract]: {text3[:150]}")
            except Exception as e:
                logger.error(f"[OCR-Tesseract Error]: {e}")
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
            results = self.reader.readtext(contrast, paragraph=False)
            text4 = " ".join([r[1] for r in results])
            all_text.append(text4)
            logger.info(f"[OCR-Contrast]: {text4[:150]}")
        except Exception as e:
            logger.error(f"[OCR-Contrast Error]: {e}")
        
        combined_text = " ".join(all_text).upper()
        return combined_text
    
    def detect_card_type(self, image):
        """ENHANCED: Detect Aadhaar or PAN with MAXIMUM ACCURACY"""
        try:
            text = self.extract_text_multiple_methods(image)
            logger.info(f"[FULL OCR TEXT]: {text[:500]}")
            
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
                    logger.warning(f"[INVALID CARD] Detected: {keyword}")
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
            
            logger.info(f"[FINAL SCORE] Aadhaar={aadhaar_score}, PAN={pan_score}")
            
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
            logger.error(f"[CARD DETECT ERROR]: {e}")
            return "Unknown"
    
    def extract_first_name(self, image):
        """Extract FIRST NAME from ID card"""
        try:
            first_name, full_name, debug = extractor.extract_name_from_image(image)
            
            if first_name and full_name:
                logger.info(f"[EXTRACTED] First: '{first_name}', Full: '{full_name}'")
                speak(f"The name extracted from your ID card is {full_name}.", duration=4.0)
                
                print("\n" + "=" * 60)
                print(f"✅ NAME EXTRACTED SUCCESSFULLY")
                print(f"   First Name: {first_name}")
                print(f"   Full Name:  {full_name}")
                print("=" * 60 + "\n")
                
                return first_name, full_name
            else:
                logger.warning("[NAME EXTRACTION] Failed")
                speak("Could not extract name from the ID card.", duration=3.0)
                return None, None
                
        except Exception as e:
            logger.error(f"[NAME EXTRACT ERROR]: {e}")
            return None, None
    
    def verify_with_spoken_name(self, cap, spoken_name):
        """Verify ID and compare with spoken name"""
        speak(f"Hello {spoken_name}. Let's verify your identity.", duration=3.0)
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            if attempt > 0:
                speak(f"Attempt {attempt + 1}. Please show your ID card again.", duration=3.0)
            
            id_image = self.capture_id_card(cap)
            if id_image is None:
                speak("Could not capture image. Try again.", duration=2.5)
                continue
            
            card_type = self.detect_card_type(id_image)
            
            if card_type == "Invalid":
                speak("Invalid ID. Please show Aadhaar or PAN.", duration=3.0)
                continue
            elif card_type == "Unknown":
                speak("Could not identify card. Show more clearly.", duration=3.0)
                continue
            
            speak(f"{card_type} card detected. Extracting name.", duration=2.5)
            
            first_name, full_name = self.extract_first_name(id_image)
            
            if not first_name:
                speak("Could not extract name. Try again.", duration=3.0)
                continue
            
            if names_match(spoken_name, first_name):
                logger.info(f"[VERIFICATION SUCCESS] '{spoken_name}' matches '{first_name}'")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                id_path = os.path.join(ID_FOLDER, f"{first_name}_{card_type}_{timestamp}.jpg")
                cv2.imwrite(id_path, id_image)
                
                speak(f"Perfect! Your name matches. Verification completed.", duration=4.0)
                return card_type, first_name, full_name
            else:
                logger.warning(f"[VERIFICATION FAILED] '{spoken_name}' != '{first_name}'")
                speak(f"Name mismatch. You said {spoken_name}, but ID shows {first_name}.", duration=5.0)
        
        speak("Verification failed. Name does not match ID card.", duration=3.0)
        return None, None, None

# -------------------- MULTI-FRAME CAPTURE --------------------
def capture_multiple_frames(cap):
    """Capture frames with visual feedback"""
    speak(f"Look at camera. Capturing for 15 seconds.", duration=3.0)
    
    frames_data = []
    start_time = time.time()
    interval = 15.0 / CAPTURE_FRAMES
    next_capture = time.time()
    captured = 0
    
    while captured < CAPTURE_FRAMES and time.time() - start_time < 20:
        ret, frame = cap.read()
        if not ret:
            continue
        
        display_frame = frame.copy()
        progress = int((captured / CAPTURE_FRAMES) * 100)
        cv2.putText(display_frame, f"Capturing: {progress}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Photos: {captured}/{CAPTURE_FRAMES}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        try:
            cv2.imshow("Drishya AI - Face Capture", display_frame)
            cv2.waitKey(1)
        except:
            pass
        
        if time.time() >= next_capture:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            
            if len(locations) > 0:
                frames_data.append((frame.copy(), sharpness))
                captured += 1
            
            next_capture = time.time() + interval
    
    if len(frames_data) < 10:
        speak("Not enough clear photos.", duration=2.5)
        return None
    
    frames_data.sort(key=lambda x: x[1], reverse=True)
    best_frames = [f[0] for f in frames_data[:SELECT_BEST]]
    
    logger.info(f"[CAPTURE] Selected {len(best_frames)} best")
    return best_frames

def compute_robust_encoding(frames):
    """Compute average encoding"""
    encodings = []
    
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        
        if len(locations) > 0:
            face_encodings = face_recognition.face_encodings(rgb, locations)
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])
    
    if not encodings:
        return None
    
    avg_encoding = np.mean(encodings, axis=0)
    logger.info(f"[ENCODING] Averaged {len(encodings)} encodings")
    return avg_encoding

# -------------------- REGISTRATION --------------------
def register_new_person(cap, verifier):
    """Register new person with ID verification"""
    speak("Welcome! Let's register you.", duration=2.5)
    
    spoken_name_raw = listen_for_response("Please tell me your first name.", timeout=8, check_gui_switch=True)
    
    if spoken_name_raw == "SWITCH_TO_GUI":
        return "SWITCH_TO_GUI"
    
    if not spoken_name_raw:
        speak("Could not hear name. Registration cancelled.", duration=3.0)
        return False
    
    spoken_name = clean_name(spoken_name_raw)
    
    if not spoken_name:
        speak("Invalid name. Registration cancelled.", duration=2.5)
        return False
    
    speak(f"I heard {spoken_name}. Is that correct? Say yes or no.", duration=3.5)
    confirmation = listen_for_response("Confirm name:", timeout=6)
    
    if not confirmation or "no" in confirmation.lower():
        speak("Name not confirmed. Registration cancelled.", duration=3.0)
        return False
    
    db_spoken_name, db_full_name = find_matching_verified_name(spoken_name)
    if db_spoken_name:
        logger.warning(f"[REGISTRATION] Name '{spoken_name}' already exists")
        speak(f"{spoken_name} is already registered. Cannot register again.", duration=4.0)
        return False
    
    card_type, extracted_first, extracted_full = verifier.verify_with_spoken_name(cap, spoken_name)
    
    if not card_type:
        speak("ID verification failed. Registration cancelled.", duration=3.0)
        return False
    
    best_frames = capture_multiple_frames(cap)
    if best_frames is None:
        speak("Could not capture face. Registration cancelled.", duration=3.0)
        return False
    
    encoding = compute_robust_encoding(best_frames)
    if encoding is None:
        speak("Could not process face. Registration cancelled.", duration=3.0)
        return False
    
    success = save_face(spoken_name, encoding, card_type, extracted_full)
    
    if success:
        speak(f"Registration completed! Welcome {spoken_name}. Verified with {card_type}.", duration=5.0)
        logger.info(f"[REGISTRATION SUCCESS] {spoken_name}")
        return True
    else:
        speak("Failed to save information. Try again.", duration=3.0)
        return False

# -------------------- MAIN SYSTEM --------------------
def main():
    """Main system with GUI switch capability"""
    logger.info("[SYSTEM START] Drishya AI Enhanced")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"[CAMERA ERROR] Cannot open camera {CAMERA_INDEX}")
        speak("Camera error. System exiting.", duration=2.0)
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        verifier = IDVerifier()
    except Exception as e:
        logger.error(f"[VERIFIER INIT ERROR]: {e}")
        speak("Verification system error. Exiting.", duration=2.0)
        cap.release()
        return
    
    known_encodings, known_names, verification_status = load_faces()
    
    speak("Drishya AI system activated. Scanning for faces.", duration=3.0)
    speak("Say switch to GUI anytime to open web interface.", duration=3.0)
    logger.info("[SYSTEM] Ready")
    
    last_announcement = {}
    no_face_timer = time.time()
    gui_check_timer = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[CAMERA] Frame read failed")
                continue
            
            display_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            # Periodic GUI switch check
            if time.time() - gui_check_timer > 30:
                gui_check_timer = time.time()
                logger.debug("[SYSTEM] Listening for GUI switch command...")
            
            if len(face_locations) > 0:
                no_face_timer = time.time()
                
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    current_time = time.time()
                    
                    if len(known_encodings) == 0:
                        name = "Unknown"
                        color = (0, 0, 255)
                        display_name = "Unknown"
                    else:
                        distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_idx = np.argmin(distances)
                        best_distance = distances[best_match_idx]
                        
                        if best_distance < TOLERANCE:
                            name = known_names[best_match_idx]
                            is_verified = verification_status.get(name, False)
                            
                            if is_verified:
                                color = (0, 255, 0)
                                display_name = f"{name} (Verified)"
                            else:
                                color = (0, 165, 255)
                                display_name = f"{name} (Not Verified)"
                            
                            if name not in last_announcement or (current_time - last_announcement[name]) > COOLDOWN_IDENTITY:
                                last_announcement[name] = current_time
                                
                                if is_verified:
                                    speak(f"Hello {name}. Welcome back.", duration=2.5)
                                else:
                                    speak(f"Hello {name}. You need verification.", duration=3.0)
                                    
                                    card_type, extracted_first, extracted_full = verifier.verify_with_spoken_name(cap, name)
                                    
                                    if card_type and extracted_first:
                                        update_success = update_verification_status(name, card_type, extracted_full)
                                        
                                        if update_success:
                                            speak(f"Verification complete, {name}.", duration=3.0)
                                            known_encodings, known_names, verification_status = load_faces()
                                        else:
                                            speak("Failed to update status.", duration=2.0)
                        else:
                            name = "Unknown"
                            color = (0, 0, 255)
                            display_name = "Unknown"
                            
                            if "Unknown" not in last_announcement or (current_time - last_announcement["Unknown"]) > COOLDOWN_IDENTITY:
                                last_announcement["Unknown"] = current_time
                                speak("Unknown person detected. Register? Say yes or no. Or say switch to GUI.", duration=5.0)
                                
                                response = listen_for_response("Register or switch to GUI?", timeout=8, check_gui_switch=True)
                                
                                if response == "SWITCH_TO_GUI":
                                    logger.info("[GUI] User requested GUI switch")
                                    if start_gui_server():
                                        speak("GUI is running. This window will continue monitoring.", duration=4.0)
                                elif response and "yes" in response.lower():
                                    logger.info("[REGISTRATION] Starting")
                                    
                                    result = register_new_person(cap, verifier)
                                    
                                    if result == "SWITCH_TO_GUI":
                                        if start_gui_server():
                                            speak("GUI is running.", duration=2.0)
                                    elif result:
                                        known_encodings, known_names, verification_status = load_faces()
                                else:
                                    speak("Registration skipped.", duration=2.0)
                    
                    cv2.putText(display_frame, display_name if 'display_name' in locals() else name, 
                               (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            else:
                if time.time() - no_face_timer > NO_FACE_TIMEOUT:
                    logger.warning(f"[TIMEOUT] No face for {NO_FACE_TIMEOUT}s")
                    speak("No person detected. Shutting down.", duration=3.0)
                    break
            
            global speech_text, speech_display_time
            if time.time() < speech_display_time and speech_text:
                cv2.putText(display_frame, f"Speaking: {speech_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(display_frame, f"Registered: {len(known_names)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            verified_count = sum(verification_status.values())
            cv2.putText(display_frame, f"Verified: {verified_count}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            try:
                cv2.imshow("Drishya AI - Enhanced Verification", display_frame)
            except Exception as e:
                logger.error(f"[DISPLAY ERROR]: {e}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                speak("System shutting down.", duration=2.0)
                logger.info("[SYSTEM] Manual shutdown")
                break
            elif key == ord('r'):
                speak("Starting manual registration.", duration=2.0)
                result = register_new_person(cap, verifier)
                if result == "SWITCH_TO_GUI":
                    start_gui_server()
                elif result:
                    known_encodings, known_names, verification_status = load_faces()
            elif key == ord('g'):
                logger.info("[GUI] Manual GUI switch (G key)")
                start_gui_server()
    
    except KeyboardInterrupt:
        logger.info("[SYSTEM] Interrupted by user")
        speak("System interrupted.", duration=2.0)
    
    except Exception as e:
        logger.error(f"[SYSTEM ERROR]: {e}")
        speak("System error occurred.", duration=2.0)
    
    finally:
        stop_gui_server()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("[SYSTEM] Shutdown complete")
        speak("Thank you for using Drishya AI.", duration=1.5)

if __name__ == "__main__":
    main()