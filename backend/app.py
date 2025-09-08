from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from datetime import datetime
from db import get_connection
from questions import role_questions
from transformers import pipeline
import whisper
import os
import cv2
import threading
import time
import wave
import pyaudio
import subprocess
import traceback
from shutil import which as shutil_which
import smtplib
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from email.mime.text import MIMEText
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import random
import re

# --- Flask Setup ---
app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
app.secret_key = 'Secret-Key-API'
CORS(app, supports_credentials=True)

# --- Directories ---
BASE_DIR = os.path.abspath(os.getcwd())
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

# --- Recording State ---
_recording_lock = threading.Lock()
_recording = False
_video_thread = None
_audio_thread = None
_session_stop_event = None
_video_path = None
_audio_path = None
_final_path = None
_last_jpeg = None
_video_start_ts = None
_audio_start_ts = None

# --- Recording Config ---
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
FRAME_W = int(os.getenv("FRAME_W", "640"))
FRAME_H = int(os.getenv("FRAME_H", "480"))
REQUESTED_FPS = float(os.getenv("FPS", "20"))
AUDIO_RATE = int(os.getenv("AUDIO_RATE", "44100"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_CHUNK = int(os.getenv("AUDIO_CHUNK", "1024"))
CLEANUP_INTERMEDIATE = os.getenv("CLEANUP_INTERMEDIATE", "1") == "1"
SYNC_BIAS_MS = float(os.getenv("SYNC_BIAS_MS", "0"))

# --- Password Reset Token Serializer ---
serializer = URLSafeTimedSerializer(app.secret_key)

# --- Email Settings ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "harshinipatchigolla@gmail.com"
SMTP_PASSWORD = "wlgb pouw uevn hqml"
# Whisper model
whisper_model = whisper.load_model("base")

def validate_password(password: str) -> bool:
    """
    Password must include:
    - At least 8 characters
    - One uppercase
    - One lowercase
    - One number
    - One special character
    """
    regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$"
    return re.match(regex, password) is not None

sentiment_pipeline = pipeline("sentiment-analysis")
# --- PDF Report Generation ---
def generate_pdf_report(user_id, audio_file_path=None):
    # Get user details
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT username, role FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()
    cursor.close(); conn.close()

    name = user["username"]
    role = user["role"]
    date = datetime.now().strftime("%b %d, %Y")

    # Random scores
    tech_score = random.randint(65, 90)
    problem_score = random.randint(60, 85)
    comm_score = random.randint(70, 95)
    overall = round((tech_score + problem_score + comm_score) / 3, 2)

    # Strengths and improvements
    strengths = ["Strong communication", "Good technical foundation"]
    improvements = ["Enhance structured problem-solving", "Revise advanced ML algorithms"]

    # Transcript + sentiment
    transcript_text = ""
    sentiment_result = None
    if audio_file_path and os.path.exists(audio_file_path):
        try:
            result = whisper_model.transcribe(audio_file_path)
            transcript_text = result["text"].strip()
            sentiment_result = sentiment_pipeline(transcript_text)[0]
        except Exception as e:
            print("Error in Whisper or sentiment:", e)
            transcript_text = "Transcription failed."
            sentiment_result = None

    # Report file
    report_name = f"report_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_name)

    # PDF setup
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Interview Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.3*inch))

    # Candidate Info
    elements.append(Paragraph(f"<b>Candidate:</b> {name}", styles['Normal']))
    elements.append(Paragraph(f"<b>Role:</b> {role}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date:</b> {date}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Scores
    elements.append(Paragraph(f"<b>Overall Score:</b> {overall}%", styles['Heading3']))
    elements.append(Paragraph(f"Technical Knowledge: {tech_score}% - Good understanding of fundamentals.", styles['Normal']))
    elements.append(Paragraph(f"Problem Solving: {problem_score}% - Approach was correct but explanation could be structured.", styles['Normal']))
    elements.append(Paragraph(f"Communication: {comm_score}% - Clear and confident, slight pauses during answers.", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Strengths
    elements.append(Paragraph("<b>Strengths</b>", styles['Heading3']))
    for s in strengths:
        elements.append(Paragraph(f"- {s}", styles['Normal']))

    elements.append(Spacer(1, 0.2*inch))

    # Areas for Improvement
    elements.append(Paragraph("<b>Areas for Improvement</b>", styles['Heading3']))
    for a in improvements:
        elements.append(Paragraph(f"- {a}", styles['Normal']))

    elements.append(Spacer(1, 0.3*inch))

    # Transcript Section
    elements.append(Paragraph("<b>Transcript</b>", styles['Heading3']))
    elements.append(Paragraph(transcript_text if transcript_text else "No transcript available.", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Sentiment Section
    if sentiment_result:
        elements.append(Paragraph("<b>Sentiment Analysis</b>", styles['Heading3']))
        elements.append(Paragraph(f"Label: {sentiment_result['label']}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {sentiment_result['score']:.2f}", styles['Normal']))

    doc.build(elements)
    return report_path, transcript_text, sentiment_result
def generate_sample_report(output_path="sample_report.pdf"):
    # Candidate Info
    name = "John Doe"
    role = "Software Engineer"
    date = datetime.now().strftime("%b %d, %Y")

    # Random scores (like your code does)
    tech_score = random.randint(65, 90)
    problem_score = random.randint(60, 85)
    comm_score = random.randint(70, 95)
    overall = round((tech_score + problem_score + comm_score) / 3, 2)

    # Strengths and improvements
    strengths = ["Strong communication", "Good technical foundation"]
    improvements = ["Enhance structured problem-solving", "Revise advanced ML algorithms"]

    # Dummy transcript and sentiment
    transcript_text = "I am passionate about AI and enjoy solving complex technical challenges."
    sentiment_result = {"label": "POSITIVE", "score": 0.92}

    # Report file setup
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Interview Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.3*inch))

    # Candidate Info
    elements.append(Paragraph(f"<b>Candidate:</b> {name}", styles['Normal']))
    elements.append(Paragraph(f"<b>Role:</b> {role}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date:</b> {date}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Scores
    elements.append(Paragraph(f"<b>Overall Score:</b> {overall}%", styles['Heading3']))
    elements.append(Paragraph(f"Technical Knowledge: {tech_score}%", styles['Normal']))
    elements.append(Paragraph(f"Problem Solving: {problem_score}%", styles['Normal']))
    elements.append(Paragraph(f"Communication: {comm_score}%", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Strengths
    elements.append(Paragraph("<b>Strengths</b>", styles['Heading3']))
    for s in strengths:
        elements.append(Paragraph(f"- {s}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # Areas for Improvement
    elements.append(Paragraph("<b>Areas for Improvement</b>", styles['Heading3']))
    for a in improvements:
        elements.append(Paragraph(f"- {a}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Transcript Section
    elements.append(Paragraph("<b>Transcript</b>", styles['Heading3']))
    elements.append(Paragraph(transcript_text if transcript_text else "No transcript available.", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Sentiment Section
    if sentiment_result:
        elements.append(Paragraph("<b>Sentiment Analysis</b>", styles['Heading3']))
        elements.append(Paragraph(f"Label: {sentiment_result['label']}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {sentiment_result['score']:.2f}", styles['Normal']))

    # Build PDF
    doc.build(elements)
    print(f"✅ Sample report generated at {os.path.abspath(output_path)}")
def has_ffmpeg():
    return shutil_which("ffmpeg") is not None

# ----------------- Video recording -----------------
def _video_worker(video_path, stop_event):
    cap = None
    writer = None
    try:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(CAM_INDEX, backend)

        attempts = 0
        while not cap.isOpened() and attempts < 5:
            attempts += 1
            print(f"⚠️ Video: camera not open, retry {attempts}/5 ...")
            time.sleep(0.5)
            cap = cv2.VideoCapture(CAM_INDEX, backend)

        if not cap.isOpened():
            print("❌ Video: Could not open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        first_frame = None
        start_wait = time.time()
        while time.time() - start_wait < 5:
            ok, frame = cap.read()
            if ok and frame is not None:
                first_frame = frame
                break
            time.sleep(0.05)

        if first_frame is None:
            print("❌ Video: No valid frame from camera (timeout).")
            cap.release()
            return

        height, width = first_frame.shape[:2]
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(cam_fps) if cam_fps and cam_fps > 1.0 else float(REQUESTED_FPS or 20.0)

        # Write directly to MP4 to avoid container/fourcc mismatches
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        if not writer or not writer.isOpened():
            print("❌ VideoWriter failed to open.")
            cap.release()
            return

        writer.write(first_frame)
        # record first-frame timestamp
        global _video_start_ts
        if _video_start_ts is None:
            _video_start_ts = time.time()
        frames_written = 1

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("⚠️ Video: frame read failed, retrying...")
                time.sleep(0.02)
                continue
            
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frames_written += 1
            # Update MJPEG preview buffer
            try:
                ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    # Store as bytes
                    global _last_jpeg
                    _last_jpeg = buf.tobytes()
            except Exception:
                pass

            # Avoid enforcing our own sleep cadence; let the camera pacing drive frame rate

        print(f"✅ Video worker stopped. Frames written: {frames_written}")
    except Exception as e:
        print("❌ Exception in video worker:", e)
        traceback.print_exc()
    finally:
        if writer:
            writer.release()
        if cap:
            cap.release()

# ----------------- Audio recording -----------------
def _audio_worker(audio_path, stop_event):
    """
    Records microphone audio to WAV until stop_event is set.
    """
    p = None
    stream = None
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE,
                        input=True,
                        frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print("❌ Audio: could not open input stream:", e)
        if p:
            p.terminate()
        return

    frames = []
    print("🎙️ Audio worker started.")
    first_chunk = True
    while not stop_event.is_set():
        try:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            frames.append(data)
            if first_chunk:
                # record first-audio timestamp
                global _audio_start_ts
                if _audio_start_ts is None:
                    _audio_start_ts = time.time()
                first_chunk = False
        except (OSError, IOError) as e:
            # Handle buffer overflow or other audio errors
            print("⚠️ Audio read error (buffer overflow):", e)
            time.sleep(0.01)
        except Exception as e:
            print("⚠️ Audio read error:", e)
            time.sleep(0.01)

    try:
        if stream:
            stream.stop_stream()
            stream.close()
    except Exception:
        pass
    finally:
        if p:
            p.terminate()

    try:
        wf = wave.open(audio_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"✅ Audio saved to {audio_path} ({len(frames)} chunks)")
    except Exception as e:
        print("❌ Failed to save audio WAV:", e)
        traceback.print_exc()

# ----------------- Merge (ffmpeg) -----------------
def _merge_av(video_path, audio_path, out_path):
    """
    Merge video (.avi) + audio (.wav) into out_path (.mp4) using ffmpeg.
    Re-encodes video to H.264 and audio to AAC for broad compatibility.
    """
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found in PATH")

    # Compute relative start offset between first video frame and first audio sample
    # Positive offset => audio started after video; we will delay video. Negative => delay audio.
    offset = 0.0
    try:
        if _video_start_ts is not None and _audio_start_ts is not None:
            offset = float(_audio_start_ts - _video_start_ts)
    except Exception:
        offset = 0.0

    # Build inputs with optional per-stream itsoffset
    base = ["ffmpeg", "-y", "-fflags", "+genpts"]
    # enforce CFR output rate and compatibility; normalize pts for video and audio
    v_settings = ["-map", "0:v:0", "-vf", "setpts=PTS-STARTPTS", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p", "-vsync", "cfr", "-r", str(int(REQUESTED_FPS or 20))]
    # stronger audio timestamp normalization: async resample + asetpts, with guard against drift
    a_settings = ["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-af", "aresample=async=1000:min_hard_comp=0.100:first_pts=0,asetpts=PTS-STARTPTS"]

    # apply optional bias for fine tuning
    bias = SYNC_BIAS_MS / 1000.0
    eff = offset + bias
    if eff > 0.02:  # audio started later -> delay video to match audio
        inputs = ["-itsoffset", f"{eff:.3f}", "-i", video_path, "-i", audio_path]
    elif eff < -0.02:  # audio started earlier -> delay audio to match video
        inputs = ["-i", video_path, "-itsoffset", f"{abs(eff):.3f}", "-i", audio_path]
    else:
        inputs = ["-i", video_path, "-i", audio_path]

    cmd = base + inputs + v_settings + a_settings + ["-shortest", "-movflags", "+faststart", out_path]
    print("🔁 Running ffmpeg merge:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("❌ ffmpeg failed:\n", res.stderr)
        raise RuntimeError("ffmpeg merge failed")
    # Log stderr as informational since ffmpeg writes progress to stderr
    if res.stderr:
        print("ffmpeg info:\n", res.stderr)
    print("✅ ffmpeg merge completed:", out_path)

# ----------------- Public control functions -----------------
def start_recording_session(base_name: str):
    """
    Starts video + audio workers and returns the paths.
    """
    global _recording, _video_thread, _audio_thread, _video_path, _audio_path, _final_path, _session_stop_event
    with _recording_lock:
        # Check if already recording but threads are dead - clean up first
        if _recording:
            # Check if threads are actually alive
            video_alive = _video_thread and _video_thread.is_alive()
            audio_alive = _audio_thread and _audio_thread.is_alive()
            
            if not video_alive and not audio_alive:
                print("⚠️ Recording flag set but threads dead - cleaning up")
                _recording = False
                _session_stop_event = None
                _video_thread = None
                _audio_thread = None
            else:
                return False, "Already recording"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_base = f"{base_name}_{ts}"
        _video_path = os.path.join(UPLOAD_DIR, f"{name_base}.mp4")
        _audio_path = os.path.join(UPLOAD_DIR, f"{name_base}.wav")
        _final_path = os.path.join(UPLOAD_DIR, f"{name_base}_final.mp4")

        # Create a new stop event for this session
        _session_stop_event = threading.Event()
        
        try:
            # Start video thread
            _video_thread = threading.Thread(target=_video_worker, args=(_video_path, _session_stop_event), daemon=True)
            # Start audio thread
            _audio_thread = threading.Thread(target=_audio_worker, args=(_audio_path, _session_stop_event), daemon=True)

            _video_thread.start()
            _audio_thread.start()

            _recording = True
            print(f"▶️ Recording started: video={_video_path}, audio={_audio_path}")
            return True, {"video": os.path.basename(_video_path), "audio": os.path.basename(_audio_path)}
        except Exception as e:
            # If thread creation fails, clean up
            print(f"❌ Failed to start recording threads: {e}")
            _recording = False
            _session_stop_event = None
            _video_thread = None
            _audio_thread = None
            return False, f"Failed to start recording: {e}"

def stop_recording_session():
    """
    Stops recording and merges files. Returns final path or raises on error.
    """
    global _recording, _video_thread, _audio_thread, _video_path, _audio_path, _final_path, _session_stop_event
    
    with _recording_lock:
        if not _recording:
            return False, "Not recording"

        # Signal threads to stop
        if _session_stop_event:
            _session_stop_event.set()
        
        _recording = False
        
    # Wait for threads to finish with more patience
    print("⏹️ Waiting for recording threads to finish...")
    if _video_thread and _video_thread.is_alive():
        _video_thread.join(timeout=30)
        if _video_thread.is_alive():
            print("⚠️ Video thread still alive after 30s timeout")
    
    if _audio_thread and _audio_thread.is_alive():
        _audio_thread.join(timeout=30)
        if _audio_thread.is_alive():
            print("⚠️ Audio thread still alive after 30s timeout")
    
    # Reset thread handles, keep timestamps for merge alignment
    _session_stop_event = None
    _video_thread = None
    _audio_thread = None
    
    # Allow more time for OS buffers to flush and files to be fully written
    print("⏳ Allowing buffers to flush...")
    time.sleep(2)

    # Validate intermediate files with better error messages
    if not _video_path:
        return False, "Video path not set - recording may not have started properly"
    
    if not os.path.exists(_video_path):
        return False, f"Video file not found at: {_video_path}"
        
    video_size = os.path.getsize(_video_path)
    if video_size == 0:
        return False, f"Video file is empty (0 bytes): {_video_path}"
    
    if not _audio_path:
        return False, "Audio path not set - recording may not have started properly"
        
    if not os.path.exists(_audio_path):
        return False, f"Audio file not found at: {_audio_path}"
        
    audio_size = os.path.getsize(_audio_path)
    if audio_size == 0:
        return False, f"Audio file is empty (0 bytes): {_audio_path}"

    print(f"📊 File sizes - Video: {video_size} bytes, Audio: {audio_size} bytes")

    # Merge with better error handling
    try:
        if not _final_path:
            return False, "Final output path not set"
            
        print(f"🔄 Starting merge: {_video_path} + {_audio_path} → {_final_path}")
        _merge_av(_video_path, _audio_path, _final_path)
        
        # Verify the final file was created
        if not os.path.exists(_final_path):
            return False, f"Merge completed but final file not found: {_final_path}"
            
        final_size = os.path.getsize(_final_path)
        if final_size == 0:
            return False, f"Merge completed but final file is empty: {_final_path}"
            
        print(f"✅ Final file created successfully: {final_size} bytes")
        
    except Exception as e:
        error_msg = f"Merge failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg

    # Optionally cleanup intermediate files
    if CLEANUP_INTERMEDIATE:
        try:
            os.remove(_video_path)
            os.remove(_audio_path)
            print("🧹 Cleaned up intermediate files")
        except Exception as e:
            print(f"⚠️ Failed to cleanup intermediate files: {e}")

    print("✅ Recording session completed, final saved at:", _final_path)
    # reset timestamps for next session
    global _video_start_ts, _audio_start_ts
    _video_start_ts = None
    _audio_start_ts = None
    return True, _final_path

# --- Routes ---
@app.route('/')
def root():
    return redirect(url_for('signin'))

@app.route('/api/user')
def api_user():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 403
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT username, role FROM users WHERE id=%s", (session['user_id'],))
    user = cursor.fetchone()
    cursor.close(); conn.close()
    return jsonify({'username': user['username'], 'role': user['role']})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form if request.form else request.json
        username, email, password = data.get('username'), data.get('email'), data.get('password')

        # Backend password validation
        if not validate_password(password):
            return jsonify({'error': 'Password must be 8+ chars, include uppercase, lowercase, number, and special character.'}), 400

        password_hash = generate_password_hash(password)

        conn = get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            cursor.close(); conn.close()
            return jsonify({'error': 'Email already registered'}), 400

        cursor.execute("INSERT INTO users (username, email, password, interview_done) VALUES (%s, %s, %s, 0)",
                       (username, email, password_hash))
        conn.commit(); cursor.close(); conn.close()
        return jsonify({'message': 'Signup successful'})
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        data = request.form if request.form else request.json
        email, password = data.get('email'), data.get('password')

        conn = get_connection(); cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone(); cursor.close(); conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['name'] = user['username']
            session['interview_done'] = bool(user.get('interview_done', 0))
            return jsonify({'message': 'Login successful'})
        return jsonify({'error': 'Invalid credentials'}), 400
    return render_template('signin.html')
import re

# Forgot Password Request
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        conn = get_connection(); cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close(); conn.close()

        if not user:
            flash("Email not found.", "danger")
            return render_template("forgot_password.html", step="request")

        token = serializer.dumps(email, salt="reset-password")
        reset_link = url_for('reset_with_token', token=token, _external=True)

        try:
            msg = MIMEText(f"Click to reset your password: {reset_link}")
            msg['Subject'], msg['From'], msg['To'] = "Password Reset", SMTP_EMAIL, email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASSWORD)
                server.sendmail(SMTP_EMAIL, [email], msg.as_string())
            flash("Reset link sent to your email.", "success")
        except Exception as e:
            flash(f"Failed to send email: {e}", "danger")

    return render_template("forgot_password.html", step="request")


# Reset with Token + Validation
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_with_token(token):
    try:
        email = serializer.loads(token, salt="reset-password", max_age=600)
    except (SignatureExpired, BadSignature):
        flash("Invalid or expired reset link.", "danger")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')

        # Backend password validation (same as frontend regex)
        regex = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$')

        if not regex.match(password):
            flash("Password must have 8+ chars, uppercase, lowercase, number, special char.", "danger")
            return render_template("forgot_password.html", step="reset", token=token)

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("forgot_password.html", step="reset", token=token)

        # Save new password
        hashed_password = generate_password_hash(password)
        conn = get_connection(); cursor = conn.cursor()
        cursor.execute("UPDATE users SET password=%s WHERE email=%s", (hashed_password, email))
        conn.commit(); cursor.close(); conn.close()

        flash("Password updated successfully. Please sign in.", "success")
        return redirect(url_for('signin'))

    return render_template("forgot_password.html", step="reset", token=token)

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('signin'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('signin'))

@app.route('/interview_status')
def interview_status():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 403
    return jsonify({'completed': session.get('interview_done', False)})

@app.route('/interview')
def interview():
    if 'user_id' not in session:
        return redirect(url_for('signin'))
    return render_template('interview.html', name=session.get('name'))

@app.route('/set_role', methods=['POST'])
def set_role():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 403
    data = request.get_json() or {}
    role = data.get('role')
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("UPDATE users SET role=%s WHERE id=%s", (role, session['user_id']))
    conn.commit(); cursor.close(); conn.close()
    return jsonify({'message': 'Role updated'})

@app.route('/get_questions')
def get_questions_route():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 403
    conn = get_connection(); cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT role FROM users WHERE id=%s", (session['user_id'],))
    row = cursor.fetchone(); cursor.close(); conn.close()
    role = row['role'] if row else 'general'

    questions = role_questions.get(role, role_questions['general'])
    selected = random.sample(questions, 10)  # pick 10 random
    return jsonify({'questions': selected})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 403
    base_name = f"interview_{session['user_id']}"
    ok, info = start_recording_session(base_name)
    if not ok:
        return jsonify({'error': info}), 500
    return jsonify({'message': 'Recording started', 'files': info})

@app.route('/stop_recording', methods=['GET','POST'])
def stop_recording():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 403
    ok, result = stop_recording_session()
    if not ok:
        return jsonify({'error': result}), 500

    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("UPDATE users SET interview_done=1 WHERE id=%s", (session['user_id'],))
    conn.commit(); cursor.close(); conn.close()
    session['interview_done'] = True

    report_path, transcript, feedback = None, "", ""
    try:
        report_path, transcript, feedback = generate_pdf_report(session['user_id'], result)
    except Exception as e:
        print("⚠ Report generation error:", e)

    return jsonify({
        'message': 'Recording stopped',
        'file': os.path.basename(result),
        'report': os.path.basename(report_path) if report_path else None,
        'transcript': transcript,
        'feedback': feedback
    })

@app.route('/check_ffmpeg')
def check_ffmpeg():
    return jsonify({'available': has_ffmpeg()})

@app.route('/recording_status')
def recording_status():
    return jsonify({'recording': _recording})

@app.route('/reset_recording', methods=['POST'])
def reset_recording():
    global _recording, _video_thread, _audio_thread, _session_stop_event
    with _recording_lock:
        if _session_stop_event:
            _session_stop_event.set()
        _recording, _video_thread, _audio_thread, _session_stop_event = False, None, None, None
    return jsonify({'message': 'Recording state reset'})

def _mjpeg_generator():
    """Yield latest JPEG frames as multipart stream without grabbing camera again."""
    boundary = b'--frame'
    while True:
        if _recording and _last_jpeg:
            frame = _last_jpeg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
        else:
            # If not recording or no frame yet, throttle slightly
            time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    def _mjpeg_generator():
        boundary = b'--frame'
        while True:
            if _recording and _last_jpeg:
                frame = _last_jpeg
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            else:
                time.sleep(0.05)
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_tab_switch', methods=['POST'])
def log_tab_switch():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    conn = get_connection()
    cursor = conn.cursor()

    # Log tab switch attempt
    cursor.execute(
        "INSERT INTO tab_switch_logs (user_id, timestamp) VALUES (%s, NOW())",
        (user_id,)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Tab switch logged"}), 200

if __name__ == '__main__':
    app.run(debug=True)