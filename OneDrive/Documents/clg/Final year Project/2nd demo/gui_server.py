"""
================================================================================
DRISHYA AI - PREMIUM WEB GUI SERVER
================================================================================

QUICK START:
============
1. Keep this file in the same folder as main.py
2. Install: pip install flask flask-socketio
3. Run: python gui_server.py
4. Laptop: http://localhost:5000
5. Phone: http://YOUR_LAPTOP_IP:5000

The GUI shows:
- Live camera feed with face detection
- Real-time person count (known/unknown)
- List of detected persons with verification status
- Database statistics
- Works on mobile devices
"""

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import face_recognition
import numpy as np
import json
import os
import time
import threading
from datetime import datetime
import socket

# ==================== CONFIGURATION ====================
JSON_FILE = "faces.json"
TOLERANCE = 0.38
CAMERA_INDEX = 0

# ==================== FLASK SETUP ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'drishya-ai-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==================== GLOBAL STATE ====================
camera = None
camera_lock = threading.Lock()
system_state = {
    'status': 'Initializing',
    'faces_detected': 0,
    'known_faces': 0,
    'unknown_faces': 0,
    'current_names': [],
    'database_count': 0,
    'verified_count': 0,
    'face_details': []
}
state_lock = threading.Lock()

known_encodings = []
known_names = []
verification_map = {}

# ==================== DATABASE ====================
def load_faces():
    """Load face database"""
    global known_encodings, known_names, verification_map
    
    if not os.path.exists(JSON_FILE):
        return [], [], {}
    
    try:
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
            verification_map[name] = {
                'verified': bool(card_type),
                'card_type': card_type or 'N/A',
                'card_name': entry.get('card_name', 'N/A')
            }
        
        known_encodings = encodings
        known_names = names
        
        with state_lock:
            system_state['database_count'] = len(names)
            system_state['verified_count'] = sum(1 for v in verification_map.values() if v['verified'])
        
        print(f"[DATABASE] Loaded {len(names)} faces")
        return encodings, names, verification_map
    except Exception as e:
        print(f"[LOAD ERROR]: {e}")
        return [], [], {}

# ==================== FACE RECOGNITION ====================
def recognize_face(encoding, known_encodings, known_names):
    """Recognize face"""
    if len(known_encodings) == 0:
        return "Unknown"
    
    distances = face_recognition.face_distance(known_encodings, encoding)
    
    if len(distances) == 0:
        return "Unknown"
    
    best_idx = np.argmin(distances)
    
    if distances[best_idx] > TOLERANCE:
        return "Unknown"
    
    return known_names[best_idx]

# ==================== FRAME PROCESSING ====================
def process_frame(frame):
    """Process frame and detect faces"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
    locations = face_recognition.face_locations(small, model="hog")
    locations = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locations]
    
    detection_data = {
        'total_faces': len(locations),
        'known_count': 0,
        'unknown_count': 0,
        'names': [],
        'face_details': []
    }
    
    if len(locations) > 0:
        try:
            encodings = face_recognition.face_encodings(rgb, locations)
            
            names = []
            for encoding in encodings:
                name = recognize_face(encoding, known_encodings, known_names)
                names.append(name)
            
            # Count
            for name in names:
                is_verified = verification_map.get(name, {}).get('verified', False)
                
                detection_data['face_details'].append({
                    'name': name,
                    'verified': is_verified,
                    'card_type': verification_map.get(name, {}).get('card_type', 'N/A')
                })
                
                if name != "Unknown":
                    detection_data['known_count'] += 1
                else:
                    detection_data['unknown_count'] += 1
            
            detection_data['names'] = list(set(names))
            
            # Draw
            for (top, right, bottom, left), name in zip(locations, names):
                is_verified = verification_map.get(name, {}).get('verified', False)
                
                if name == "Unknown":
                    color = (0, 0, 255)  # Red
                    label = "Unknown"
                elif is_verified:
                    color = (0, 255, 0)  # Green
                    label = f"{name} ✓"
                else:
                    color = (255, 165, 0)  # Orange
                    label = f"{name} ⚠"
                
                # Box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                
                # Label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (left, bottom), 
                            (left + label_size[0] + 20, bottom + label_size[1] + 15), 
                            color, -1)
                cv2.putText(frame, label, (left + 10, bottom + label_size[1] + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"[PROCESSING ERROR]: {e}")
    
    return frame, detection_data

def generate_frames():
    """Generate video frames"""
    global camera
    
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(CAMERA_INDEX)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
    
    frame_skip = 0
    
    while True:
        with camera_lock:
            if camera is None:
                break
            success, frame = camera.read()
        
        if not success:
            break
        
        frame_skip += 1
        
        # Process every 2nd frame
        if frame_skip % 2 == 0:
            frame, detection_data = process_frame(frame)
            
            # Update state
            with state_lock:
                system_state['faces_detected'] = detection_data['total_faces']
                system_state['known_faces'] = detection_data['known_count']
                system_state['unknown_faces'] = detection_data['unknown_count']
                system_state['current_names'] = detection_data['names']
                system_state['face_details'] = detection_data['face_details']
                
                if detection_data['total_faces'] > 0:
                    system_state['status'] = 'Active - Detecting'
                else:
                    system_state['status'] = 'Active - Monitoring'
            
            # Emit update
            socketio.emit('state_update', system_state)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==================== ROUTES ====================
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get status"""
    with state_lock:
        return jsonify(system_state)

@socketio.on('connect')
def handle_connect():
    """Handle connection"""
    print('[CLIENT CONNECTED]')
    emit('state_update', system_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle disconnection"""
    print('[CLIENT DISCONNECTED]')

@socketio.on('refresh_database')
def handle_refresh():
    """Reload database"""
    load_faces()
    emit('database_refreshed', {'count': len(known_names)})

# ==================== HTML TEMPLATE ====================
def create_html():
    """Create HTML template"""
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Drishya AI</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 10px;
        }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
        .header {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .header h1 {
            color: white;
            font-size: 2.5em;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p { color: rgba(255,255,255,0.9); font-size: 1.1em; }
        
        .main-content {
            display: grid;
            grid-template-columns: 1.8fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 1024px) {
            .main-content { grid-template-columns: 1fr; }
            .header h1 { font-size: 2em; }
        }
        
        .glass-panel {
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s;
        }
        
        .glass-panel:hover { transform: translateY(-5px); }
        
        .panel-title {
            color: white;
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 20px;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            padding-bottom: 75%;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
        }
        
        .video-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .stats-panel { display: flex; flex-direction: column; gap: 20px; }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 15px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .stat-label { color: rgba(255,255,255,0.8); font-weight: 500; }
        .stat-value { color: white; font-size: 1.3em; font-weight: 700; }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-active { background: #4ade80; box-shadow: 0 0 10px #4ade80; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .names-list {
            list-style: none;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .names-list li {
            background: rgba(255,255,255,0.15);
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            border-left: 4px solid transparent;
        }
        
        .names-list li.verified {
            border-left-color: #10b981;
            background: rgba(16,185,129,0.25);
        }
        
        .names-list li.unknown {
            border-left-color: #ef4444;
            background: rgba(239,68,68,0.2);
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75em;
            margin-left: 8px;
        }
        
        .badge-verified { background: rgba(16,185,129,0.3); color: #d1fae5; }
        
        .live-badge {
            display: inline-flex;
            align-items: center;
            background: rgba(239,68,68,0.3);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-left: 10px;
        }
        
        .live-dot {
            width: 8px;
            height: 8px;
            background: #ef4444;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }
        
        .footer {
            text-align: center;
            color: white;
            margin-top: 40px;
            padding: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            margin-top: 15px;
        }
        
        .btn:hover { background: rgba(255,255,255,0.3); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Drishya AI <span class="live-badge"><span class="live-dot"></span>LIVE</span></h1>
            <p>Siri-Style Voice-Controlled Face Recognition System</p>
        </div>
        
        <div class="main-content">
            <div class="glass-panel">
                <h3 class="panel-title">📹 Live Camera Feed</h3>
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" alt="Live Feed">
                </div>
            </div>
            
            <div class="stats-panel">
                <div class="glass-panel">
                    <h3 class="panel-title">⚡ System Status</h3>
                    <div class="stat-row">
                        <span class="stat-label">Status:</span>
                        <span class="stat-value">
                            <span class="status-indicator status-active" id="status-ind"></span>
                            <span id="status-text">Active</span>
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Database:</span>
                        <span class="stat-value" id="db-count">0</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Verified:</span>
                        <span class="stat-value" style="color: #4ade80;" id="verified-count">0</span>
                    </div>
                </div>
                
                <div class="glass-panel">
                    <h3 class="panel-title">👥 Detection</h3>
                    <div class="stat-row">
                        <span class="stat-label">Total:</span>
                        <span class="stat-value" id="total-faces">0</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Known:</span>
                        <span class="stat-value" style="color: #4ade80;" id="known-faces">0</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Unknown:</span>
                        <span class="stat-value" style="color: #ef4444;" id="unknown-faces">0</span>
                    </div>
                </div>
                
                <div class="glass-panel">
                    <h3 class="panel-title">🔎 Detected Persons</h3>
                    <ul class="names-list" id="names-list">
                        <li style="text-align: center; opacity: 0.6;">No faces detected</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>🚀 Drishya AI</strong> - Voice-Controlled Recognition</p>
            <p>💬 Use voice commands in main.py for registration</p>
            <button class="btn" onclick="refreshDB()">🔄 Refresh Database</button>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('connect', () => console.log('Connected'));
        
        socket.on('state_update', (data) => {
            document.getElementById('status-text').textContent = data.status;
            document.getElementById('db-count').textContent = data.database_count;
            document.getElementById('verified-count').textContent = data.verified_count || 0;
            document.getElementById('total-faces').textContent = data.faces_detected;
            document.getElementById('known-faces').textContent = data.known_faces;
            document.getElementById('unknown-faces').textContent = data.unknown_faces;
            
            const list = document.getElementById('names-list');
            if (data.face_details && data.face_details.length > 0) {
                list.innerHTML = data.face_details.map(face => {
                    let cls = 'unknown';
                    let badge = '';
                    
                    if (face.name !== 'Unknown' && face.verified) {
                        cls = 'verified';
                        badge = '<span class="badge badge-verified">✓ Verified</span>';
                    }
                    
                    return `<li class="${cls}">${face.name}${badge}</li>`;
                }).join('');
            } else {
                list.innerHTML = '<li style="text-align: center; opacity: 0.6;">No faces detected</li>';
            }
        });
        
        function refreshDB() {
            socket.emit('refresh_database');
        }
    </script>
</body>
</html>'''
    
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("[INFO] ✅ Created templates/index.html")

# ==================== UTILITY ====================
def get_local_ip():
    """Get local IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# ==================== MAIN ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔍 DRISHYA AI - WEB GUI SERVER")
    print("="*70 + "\n")
    
    create_html()
    load_faces()
    
    print(f"[INFO] 📊 Loaded {len(known_names)} faces")
    print(f"[INFO] ✅ {system_state['verified_count']} verified\n")
    
    local_ip = get_local_ip()
    
    print("="*70)
    print("🚀 SERVER STARTED")
    print("="*70)
    print(f"\n💻 This computer: http://localhost:5000")
    print(f"📱 Mobile (same WiFi): http://{local_ip}:5000\n")
    print("💡 Keep this window open while using GUI")
    print("⌨️  Press Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    with state_lock:
        system_state['status'] = 'Active - Monitoring'
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped")
    finally:
        if camera:
            camera.release()
        print("[INFO] Goodbye! 👋")