import atexit
import base64
import os
import signal
import threading
import time
from functools import wraps

import cv2
from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
from ultralytics import YOLO

# ── Gemini API setup ──────────────────────────────────────────────────────────
try:
    from google import genai as google_genai
    import PIL.Image
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBRwc8i8-RfkJvz0Lw_8IfE7sHN2VGKPDY")
    if GEMINI_API_KEY:
        gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("[Gemini] ENABLED ✓ — using google-genai SDK")
    else:
        GEMINI_AVAILABLE = False
        print("[Gemini] No API key found.")
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Gemini] google-genai not installed. Run: pip install google-genai pillow")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "mysmarteyes-secret-2026"
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 30

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
print("[Model] Loading YOLOv8s...")
model = YOLO("yolov8s.pt")
print("[Model] Ready.")

# ── Global camera state ───────────────────────────────────────────────────────
camera           = None
output_frame     = None
lock             = threading.Lock()
ip_webcam_url    = ""
camera_connected = False
streaming_active = False

# ── In-memory user store ──────────────────────────────────────────────────────
USERS = {
    "demo": {"password": "demo123", "name": "Demo User", "lang": "en"},
}

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
DANGER_OBJECTS = {"car", "truck", "bus", "motorcycle", "bicycle", "dog", "knife", "scissors", "fire"}

TRANSLATIONS = {
    "en": {"ahead": "ahead of you", "left": "to your left", "right": "to your right",
           "v_close": "very close", "nearby": "nearby", "moderate": "at a moderate distance",
           "far": "far away", "warning": "Warning!", "conf": "confident",
           "clear": "The path ahead appears clear. No objects detected.", "a": "A"},
    "hi": {"ahead": "आपके सामने", "left": "आपके बाईं ओर", "right": "आपके दाईं ओर",
           "v_close": "बहुत पास", "nearby": "पास में", "moderate": "मध्यम दूरी पर",
           "far": "दूर", "warning": "चेतावनी!", "conf": "विश्वास",
           "clear": "रास्ता साफ दिखता है। कोई वस्तु नहीं मिली।", "a": "एक"},
    "kn": {"ahead": "ನಿಮ್ಮ ಮುಂದೆ", "left": "ನಿಮ್ಮ ಎಡಕ್ಕೆ", "right": "ನಿಮ್ಮ ಬಲಕ್ಕೆ",
           "v_close": "ತುಂಬಾ ಹತ್ತಿರ", "nearby": "ಹತ್ತಿರದಲ್ಲಿ", "moderate": "ಮಧ್ಯಮ ದೂರದಲ್ಲಿ",
           "far": "ದೂರದಲ್ಲಿ", "warning": "ಎಚ್ಚರಿಕೆ!", "conf": "ವಿಶ್ವಾಸ",
           "clear": "ದಾರಿ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುತ್ತದೆ. ಯಾವುದೇ ವಸ್ತು ಕಂಡುಬಂದಿಲ್ಲ.", "a": "ಒಂದು"},
}

LANG_NAMES_FULL = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
GEMINI_LANG_PROMPTS = {"en": "English", "hi": "Hindi (हिंदी)", "kn": "Kannada (ಕನ್ನಡ)"}
GEMINI_GREETINGS = {
    "en": "Hello {name}! I am your SmartEyes AI assistant. What would you like to do today? Say detect, navigate, location, or change language.",
    "hi": "नमस्ते {name}! मैं आपका SmartEyes AI सहायक हूँ। आज आप क्या करना चाहते हैं?",
    "kn": "ನಮಸ್ಕಾರ {name}! ನಾನು ನಿಮ್ಮ SmartEyes AI ಸಹಾಯಕ. ಇಂದು ನೀವು ಏನು ಮಾಡಲು ಬಯಸುತ್ತೀರಿ?",
}

# ── Auth decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            if request.is_json or request.path.startswith("/api"):
                return jsonify({"status": "error", "message": "Not authenticated"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

# ── Camera cleanup ────────────────────────────────────────────────────────────
def _release_camera():
    global camera, streaming_active
    streaming_active = False
    if camera is not None:
        camera.release()
        print("[Camera] Released on shutdown.")

atexit.register(_release_camera)
signal.signal(signal.SIGINT,  lambda s, f: (_release_camera(), exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (_release_camera(), exit(0)))


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def gemini_describe_scene(frame_bgr, lang="en", detected_objects=None):
    if not GEMINI_AVAILABLE:
        return None
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(frame_rgb)
        lang_name = GEMINI_LANG_PROMPTS.get(lang, "English")
        yolo_context = ""
        if detected_objects:
            items = [f"{o['name']} ({o['confidence']}%)" for o in detected_objects]
            yolo_context = f"YOLO pre-detected: {', '.join(items)}. "
        prompt = (
            f"You are an AI assistant for a visually impaired person. "
            f"Analyze this camera image and describe what you see in {lang_name}. "
            f"{yolo_context}"
            f"Rules: 1) Respond in {lang_name} ONLY. 2) Mention object positions (left/right/ahead). "
            f"3) Warn about dangers FIRST. 4) Max 2-3 sentences. 5) Sound natural. "
            f"Do NOT mention confidence percentages."
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, pil_image]
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Vision error: {e}")
        return None


def gemini_translate(text, lang):
    if not GEMINI_AVAILABLE or lang == "en":
        return text
    try:
        lang_name = GEMINI_LANG_PROMPTS.get(lang, "English")
        prompt = f"Translate to {lang_name}. Return ONLY the translation:\n\n{text}"
        response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Translation error: {e}")
        return text


# ══════════════════════════════════════════════════════════════════════════════
# YOLO DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def describe_detections_yolo(results, frame_width, frame_height, lang="en"):
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    if not results or len(results[0].boxes) == 0:
        return t["clear"], [], False

    descriptions = []
    danger_alerts = []
    object_counts = {}
    has_danger = False
    objects_list = []

    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        cls_name = model.names[int(box.cls)]
        xywh = box.xywh[0]
        x_center = xywh[0].item() / frame_width
        box_area = xywh[2].item() * xywh[3].item()
        area_ratio = box_area / (frame_width * frame_height)

        pos_key = "left" if x_center < 0.33 else ("right" if x_center > 0.66 else "ahead")
        dist_key = "v_close" if area_ratio > 0.15 else ("nearby" if area_ratio > 0.05 else ("moderate" if area_ratio > 0.02 else "far"))
        conf_pct = int(confidence * 100)

        objects_list.append({"name": cls_name, "confidence": conf_pct, "position": pos_key, "distance": dist_key})

        key = f"{cls_name}_{pos_key}"
        if key not in object_counts:
            object_counts[key] = {"name": cls_name, "position": t[pos_key], "distance": t[dist_key], "conf": conf_pct, "count": 0}
        object_counts[key]["count"] += 1

        if cls_name in DANGER_OBJECTS:
            has_danger = True
            danger_alerts.append(f"{t['warning']} {t['a']} {cls_name} {t[pos_key]}, {t[dist_key]}. {conf_pct}% {t['conf']}.")

    if not object_counts:
        return t["clear"], [], False

    for info in object_counts.values():
        c = info["count"]
        desc = (f"{t['a']} {info['name']} {info['position']}, {info['distance']}. {info['conf']}% {t['conf']}."
                if c == 1 else
                f"{c} {info['name']}s {info['position']}, {info['distance']}. {info['conf']}% {t['conf']}.")
        descriptions.append(desc)

    result_text = (" ".join(danger_alerts) + " " + " ".join(descriptions) if danger_alerts else " ".join(descriptions))
    return result_text, objects_list, has_danger


def run_full_detection(frame, lang="en"):
    fh, fw = frame.shape[:2]
    results = model(frame, verbose=False)
    _, objects_list, has_danger = describe_detections_yolo(results, fw, fh, lang="en")
    annotated = results[0].plot()

    if GEMINI_AVAILABLE:
        gemini_text = gemini_describe_scene(frame, lang=lang, detected_objects=objects_list)
        result_text = gemini_text if gemini_text else describe_detections_yolo(results, fw, fh, lang=lang)[0]
    else:
        result_text = describe_detections_yolo(results, fw, fh, lang=lang)[0]

    return result_text, objects_list, has_danger, annotated


# ══════════════════════════════════════════════════════════════════════════════
# FRAME GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_frames():
    global output_frame, camera, streaming_active
    while streaming_active:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue
        with lock:
            output_frame = frame.copy()
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def _handle_command(text, lang):
    action = "none"
    response_text = ""

    def tr(msg):
        return gemini_translate(msg, lang) if GEMINI_AVAILABLE and lang != "en" else msg

    if any(kw in text for kw in ["detect", "what do you see", "what is", "look", "around", "see", "ahead",
                                   "आसपास", "क्या है", "देखो", "ಮುಂದೆ", "ನೋಡು", "ಸುತ್ತಲೂ", "ಏನಿದೆ"]):
        action = "detect"
        with lock:
            current_frame = output_frame.copy() if output_frame is not None else None
        if current_frame is None:
            response_text = tr("No camera frame available. Please connect your camera first.")
        else:
            response_text, _, _, _ = run_full_detection(current_frame, lang=lang)

    elif any(kw in text for kw in ["take me to", "navigate to", "go to", "directions to",
                                    "मुझे", "ले जाओ", "ನನ್ನನ್ನು", "ಕರೆದೊಯ್ಯಿ", "ಹೋಗು"]):
        action = "navigate"
        dest = text
        for kw in ["take me to", "navigate to", "go to", "directions to"]:
            if kw in dest:
                dest = dest.split(kw)[-1].strip()
                break
        response_text = tr(f"Opening navigation to {dest}.")

    elif any(kw in text for kw in ["where am i", "my location", "location", "where",
                                    "मेरी लोकेशन", "मैं कहाँ", "ನನ್ನ ಸ್ಥಳ", "ಎಲ್ಲಿದ್ದೇನೆ"]):
        action = "location"
        response_text = tr("Getting your current location.")

    elif any(kw in text for kw in ["stop", "pause", "रुको", "ನಿಲ್ಲಿಸು", "ಸ್ಟಾಪ್"]):
        action = "stop"
        response_text = tr("Detection paused.")

    elif any(kw in text for kw in ["resume", "continue", "start", "play", "जारी", "ಮುಂದುವರಿ"]):
        action = "resume"
        response_text = tr("Detection resumed.")

    elif any(kw in text for kw in ["hindi", "switch to hindi", "हिंदी"]):
        action = "set_lang"
        session["lang"] = "hi"
        lang = "hi"
        response_text = "भाषा हिंदी में बदल दी गई।"

    elif any(kw in text for kw in ["kannada", "switch to kannada", "ಕನ್ನಡ", "ಕನ್ನಡದಲ್ಲಿ"]):
        action = "set_lang"
        session["lang"] = "kn"
        lang = "kn"
        response_text = "ಭಾಷೆಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ."

    elif any(kw in text for kw in ["english", "switch to english", "अंग्रेजी"]):
        action = "set_lang"
        session["lang"] = "en"
        lang = "en"
        response_text = "Language switched to English."

    elif "status" in text:
        action = "status"
        cam_status = "connected" if camera_connected else "not connected"
        response_text = tr(f"Camera is {cam_status}. Language is {LANG_NAMES_FULL.get(lang, lang)}.")

    elif any(kw in text for kw in ["help", "commands", "मदद", "ಸಹಾಯ"]):
        action = "help"
        response_text = tr("Say detect to scan, navigate to go somewhere, where am I for location, stop or resume, switch to Hindi or Kannada to change language.")

    else:
        if GEMINI_AVAILABLE:
            try:
                lang_name = GEMINI_LANG_PROMPTS.get(lang, "English")
                prompt = (f"A visually impaired user said: '{text}'. "
                          f"This is a voice command for an AI vision app. "
                          f"Reply helpfully in {lang_name} in 1-2 sentences.")
                resp = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                response_text = resp.text.strip()
            except Exception:
                response_text = f"I heard: {text}. Say help for available commands."
        else:
            response_text = f"I heard: {text}. Say help for available commands."

    return action, response_text, lang


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        data     = request.get_json() or {}
        action   = data.get("action", "login")
        username = data.get("username", "").strip().lower()
        password = data.get("password", "").strip()
        name     = data.get("name", username).strip()
        lang     = data.get("lang", "en")

        if action == "signup":
            if username in USERS:
                return jsonify({"status": "error", "message": "Username already taken."})
            if len(password) < 6:
                return jsonify({"status": "error", "message": "Password must be 6+ characters."})
            USERS[username] = {"password": password, "name": name, "lang": lang}
            session.permanent = True
            session.update({"username": username, "name": name, "lang": lang})
            return jsonify({"status": "success", "redirect": "/app"})

        elif action == "guest":
            session.permanent = True
            session.update({"username": "guest", "name": "Guest", "lang": "en"})
            return jsonify({"status": "success", "redirect": "/app"})

        else:
            user = USERS.get(username)
            if not user or user["password"] != password:
                return jsonify({"status": "error", "message": "Invalid username or password."})
            session.permanent = True
            session.update({"username": username, "name": user["name"], "lang": user.get("lang", "en")})
            return jsonify({"status": "success", "redirect": "/app"})

    return render_template("login.html")

@app.route("/app")
@login_required
def app_page():
    return render_template("dashboard.html",
        user_name=session.get("name", "User"),
        user_lang=session.get("lang", "en"),
        gemini_enabled=GEMINI_AVAILABLE)

@app.route("/navigate")
@login_required
def navigate_page():
    return render_template("navigation.html",
        user_name=session.get("name", "User"),
        user_lang=session.get("lang", "en"))

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success", "redirect": "/"})


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/greeting")
@login_required
def greeting():
    lang = session.get("lang", "en")
    name = session.get("name", "User")
    text = GEMINI_GREETINGS.get(lang, GEMINI_GREETINGS["en"]).format(name=name)
    return jsonify({"status": "success", "greeting": text, "lang": lang, "name": name, "gemini": GEMINI_AVAILABLE})

@app.route("/set_camera", methods=["POST"])
@login_required
def set_camera():
    global camera, ip_webcam_url, camera_connected, streaming_active
    data = request.get_json() or {}
    url  = data.get("url", "0").strip()
    if not url:
        return jsonify({"status": "error", "message": "No camera URL provided."})
    streaming_active = False
    time.sleep(0.15)
    if camera is not None:
        camera.release()
    ip_webcam_url = url
    camera = cv2.VideoCapture(int(url) if url.isdigit() else url)
    if camera.isOpened():
        camera_connected = True
        streaming_active = True
        return jsonify({"status": "success", "message": "Camera connected."})
    camera_connected = False
    return jsonify({"status": "error", "message": "Could not connect to camera."})

@app.route("/stop_stream", methods=["POST"])
@login_required
def stop_stream():
    global streaming_active, camera, camera_connected
    streaming_active = False
    camera_connected = False
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "success"})

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detect_object", methods=["POST"])
@login_required
def detect_object():
    lang = session.get("lang", "en")
    with lock:
        current_frame = output_frame.copy() if output_frame is not None else None
    if current_frame is None:
        return jsonify({"status": "error", "message": "No frame available. Connect camera first."})
    result_text, objects_list, has_danger, annotated = run_full_detection(current_frame, lang=lang)
    _, buffer = cv2.imencode(".jpg", annotated)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"status": "success", "result": result_text, "objects": objects_list,
                    "has_danger": has_danger, "image": img_b64, "lang": lang, "gemini_used": GEMINI_AVAILABLE})

@app.route("/status")
@login_required
def status():
    return jsonify({"camera_connected": camera_connected, "streaming": streaming_active,
                    "has_frame": output_frame is not None, "camera_url": ip_webcam_url,
                    "user": session.get("name"), "lang": session.get("lang", "en"), "gemini": GEMINI_AVAILABLE})

@app.route("/api/set_lang", methods=["POST"])
@login_required
def set_lang():
    data = request.get_json() or {}
    lang = data.get("lang", "en")
    if lang not in ("en", "hi", "kn"):
        return jsonify({"status": "error", "message": "Use: en, hi, kn"})
    session["lang"] = lang
    username = session.get("username")
    if username and username in USERS:
        USERS[username]["lang"] = lang
    return jsonify({"status": "success", "lang": lang})

@app.route("/voice_text", methods=["POST"])
@login_required
def voice_text():
    lang = session.get("lang", "en")
    data = request.get_json() or {}
    text = data.get("text", "").lower().strip()
    if not text:
        return jsonify({"status": "error", "message": "No text received."})
    action, response_text, lang = _handle_command(text, lang)
    return jsonify({"status": "success", "text": text, "response": response_text,
                    "action": action, "lang": session.get("lang", "en")})


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MySmartEyes — AI Vision Assistant")
    print(f"  Gemini: {'ENABLED' if GEMINI_AVAILABLE else 'DISABLED'}")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)
