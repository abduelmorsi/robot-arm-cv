# Robot Arm Control Platform

Multi-modal control system for a robotic arm combining:
- EMG (muscle) signal acquisition via BITalino
- Machine learning gesture inference (RandomForest model + scaler)
- Speech / STT (speech-to-text) command input
- Direct / manual hand control (for calibration & fallback)
 - Computer vision hand pose control (MediaPipe + OpenCV) streaming finger states
- Microcontroller firmware (PlatformIO / `main.cpp`)

> This README consolidates the control methods and provides setup + troubleshooting guidance.

---
## Repository Layout
```
platformio.ini          # Firmware build config (PlatformIO)
src/
  main.cpp              # Firmware entry (MCU side)
  emg_control.py        # EMG-based gesture classification & command streaming
  emg_plot.py           # Live EMG oscilloscope (debug / tuning)
  hand_control.py       # Computer-vision hand pose → finger angle mapping (MediaPipe)
  stt_control.py        # Speech-to-text control script
  final_rf_model.pkl    # Trained RandomForest model for gesture recognition
  final_scaler.pkl      # Feature scaler (must match model training)
```

---
## Control Modalities
### 1. EMG Control (`emg_control.py`)
Captures raw EMG from BITalino, filters + extracts time & frequency features, scales them with `final_scaler.pkl`, classifies gestures via `final_rf_model.pkl`, then sends mapped servo angles to the arm over serial.

Detailed workflow:
1. Acquire raw samples at `sampling_rate` from BITalino (channel list `channels`).
2. Buffer samples during a user-triggered recording window (press Enter to start/stop).
3. Slice buffered signal into overlapping windows (`window_size=1000`, 50% overlap).
4. Apply notch (50 Hz) + FIR bandpass (20–400 Hz) filtering.
5. Extract time-domain + frequency-domain features (`extract_time_features`, `extract_freq_features`).
6. Scale features (same order as training) via preloaded scaler.
7. Predict gesture + probability with RandomForest; accumulate per-window predictions.
8. Majority vote → final gesture; verify confidence threshold (default >0.7) + gesture change.
9. Map gesture → 5 servo angles (`gesture_to_angles`) and transmit CSV string (e.g., `0,180,0,0,0\n`).

Electrode placement tips:
- Single differential EMG: place one electrode over muscle belly, another a few cm along fiber direction, reference (ground) on bony prominence.
- Clean skin with alcohol, allow to dry.
- Secure cables to reduce motion artifacts.

Windows pairing (if using COM port):
1. Pair BITalino in Bluetooth Settings.
2. Open Device Manager → Ports → note "Standard Serial over Bluetooth" COMx.
3. Set `mac_address` to that COM port string OR use the actual MAC on non-Windows platforms.

Adjusting filters:
- Change `notch_freq` to 60.0 Hz if in a 60 Hz mains region.
- Tighten `low_cutoff` / `high_cutoff` depending on muscle group frequency content.

Adding new gestures:
1. Collect labeled recordings (multiple trials) per gesture.
2. Extract feature vectors in same order.
3. Retrain RandomForest; save new `final_rf_model.pkl` + `final_scaler.pkl` together.
4. Update `gesture_to_angles` mapping.

Performance tuning:
- Lower `buffer_size` for more responsive windows (ensure still enough samples for stable features).
- Add temporal smoothing (e.g., majority of last N final decisions) to reduce flicker.
- Log feature vectors to CSV for offline analysis (optional addition inside loop).

### 2. Live EMG Plot (`emg_plot.py`)
Rolling real‑time plot to validate signal quality, electrode placement, and channel noise before running classification.

Key parameters inside the script:
- `sampling_rate` (default 1000 Hz)
- `channels = [0]` for A0 (adjust if using more channels)
- `plot_window` controls visible sample count
- Uses BITalino either via MAC address (Linux/macOS) or COM port (Windows). Example: `BITalino("COM4")`.

### 3. Speech / STT Control (`stt_control.py`)
Listens for spoken commands ("open hand", "close hand", etc.) using OpenAI Whisper API then maps recognized text to servo angles.

Environment & credentials:
1. Create a `.env` file in project root:
  ```
  OPENAI_API_KEY=sk-...your-key...
  ```
2. The script loads it via `python-dotenv` (`load_dotenv()`). Ensure dependency installed or remove if not needed.

Runtime flow:
1. Optional microphone selection (lists all input devices via `sounddevice`).
2. Record fixed window (`DURATION` seconds, default 5) after pressing Enter.
3. Write temporary WAV → send to Whisper (`model="whisper-1"`).
4. Normalize command string, attempt fuzzy match against known gestures.
5. Transmit servo angles over serial.

Extending commands:
- Edit `command_to_angles` function: add new `elif any(word in command for word in ["...", "..."]): return [angles]` lines.
- Keep output length = 5 values (one per finger/servo) unless firmware changed.

Latency considerations:
- Whisper API call adds network latency; for offline use consider local Whisper or Vosk.
- Reduce `DURATION` (e.g., 2s) for faster turnaround; ensure speech clarity.

Error handling:
- Low audio level warning (<0.01 peak) suggests gain or mic placement issues.
- Empty transcript triggers retry.

### 4. Computer Vision Hand Control (`hand_control.py`)
Real-time webcam-based finger state detection using **MediaPipe Hands** + **OpenCV**. Extracts landmark positions, infers each finger OPEN / CLOSED (simple tip-vs-base vertical comparison) and maps to coarse servo angles (0° open, 180° closed) which are serialized as a CSV string to the microcontroller (e.g., `0,180,0,180,0\n`).

Current logic:
1. Capture frame → flip (mirrored view)
2. Run MediaPipe Hands (single hand)
3. For each finger (thumb/index/middle/ring/pinky) compare y of tip vs base landmark
4. Encode state (OPEN→0, CLOSED→180)
5. Send over serial at 9600 baud

Potential improvements:
- Use joint angle calculation (vector dot product) for smoother granularity
- Add temporal smoothing / debounce to reduce jitter
- Dynamic calibration routine to personalize thresholds
- Map continuous flexion to proportional servo angles instead of binary 0/180

Detailed usage:
1. Ensure webcam accessible (index 0) or change `cv2.VideoCapture(0)`.
2. Launch script: `python src\hand_control.py`.
3. Observe overlay: landmarks + per-finger OPEN/CLOSED annotation.
4. Servo command string printed only implicitly on Arduino side; add local print if debugging.

Tuning parameters (inside script):
- `min_detection_confidence` / `min_tracking_confidence`: raise (0.7) for stability, may drop FPS.
- Add a frame resize (e.g., `cv2.resize`) before processing for speed.

Improving finger logic:
- Replace simple y-tip > y-base rule with angle between (MCP,PIP,DIP) vectors to compute flexion percent.
- Normalize flexion 0–1 → map to servo degrees (0–180) for proportional control.
- Introduce exponential moving average across last N frames to smooth transitions.

### 5. Firmware (`main.cpp`)
Runs on the microcontroller (configured via `platformio.ini`). Receives commands (e.g., over serial) and actuates servos / motors accordingly.

Serial protocol (current expectation):
- CSV of five integers (0–180) followed by `\n`. Example: `0,180,0,180,0`.
- Order: [thumb, index, middle, ring, pinky] (confirm with your `main.cpp`).
- Baud: 9600 (hand/speech/emg scripts) OR adjust consistently across all sources.

Firmware adjustment checklist:
1. Match baud rate with Python scripts.
2. Parse incoming line (e.g., using `Serial.readStringUntil('\n')`).
3. Split by comma, `atoi` each angle.
4. Clamp to servo limits; optionally add rate limiting for smoother motion.
5. Implement failsafe: if no packet in X ms → move to safe pose / stop.

Testing firmware in isolation:
```
python - <<EOF
import serial, time
ser = serial.Serial('COM6',9600,timeout=1)
for i in range(3):
  ser.write(b'0,0,0,0,0\n'); time.sleep(0.5)
  ser.write(b'180,180,180,180,180\n'); time.sleep(0.5)
ser.close()
EOF
```
Confirm expected motion before integrating ML / STT / CV streams.

---
## Prerequisites
### Hardware
- BITalino (e.g., BITalino (r)evolution Core / Plugged)
- Robotic arm hardware + servo driver / motor drivers
- Stable Bluetooth adapter (for BITalino) OR USB cable if using a BT-serial bridge
- Quality EMG electrodes & alcohol wipes

### Software
- Python 3.11+ (confirm with `python --version`)
- PlatformIO (VS Code extension w/ GUI) or PlatformIO Core CLI for firmware
- Windows: Ensure the BITalino is paired (Bluetooth Settings) and note the COM port from Device Manager (Ports section: "Standard Serial over Bluetooth link (COMx)").
- Webcam drivers (for CV hand control)

---
## Python Environment Setup
Create and activate a virtual environment (recommended):
```
python -m venv .venv
.venv\Scripts\activate
```
Install core dependencies (adjust if already present):
```
pip install numpy matplotlib scikit-learn pybluez bitalino pyserial
# Computer vision hand control
pip install opencv-python mediapipe
# For speech (adjust if you choose a specific backend):
pip install SpeechRecognition pyaudio # or alternative microphone backend
```
If `pyaudio` fails on Windows, install a precompiled wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/ or consider `pip install sounddevice` and adapt the script.

Place `final_rf_model.pkl` and `final_scaler.pkl` in `src/` (already present). These must correspond (same training session) or prediction quality will degrade.

Feature order integrity:
- The scaler & model expect features in exactly the order produced in code (`extract_time_features` dict insertion order + subsequent `extract_freq_features`).
- If you modify feature sets, retrain both model & scaler together; do not reorder without retraining.

Retraining outline:
1. Collect raw EMG trials per gesture (consistent duration & conditions).
2. Apply identical preprocessing (filters, windowing).
3. Extract features; build dataframe (rows=windows, columns=features, label=gesture).
4. Split train/validation; fit scaler on train only; transform both.
5. Train RandomForest (tune n_estimators, max_depth via cross-validation).
6. Persist `joblib.dump(model, 'final_rf_model.pkl')`; `joblib.dump(scaler, 'final_scaler.pkl')`.
7. Copy files into `src/`.

---
## Running the Tools
### 1. Test BITalino Connection
```
python src\emg_plot.py
```
If successful you will see a live updating plot. Close with Ctrl+C.

### 2. Run EMG Classification
```
python src\emg_control.py
```
Check console logs for recognized gestures and transmitted commands.

### 3. Run Speech Control
```
python src\stt_control.py
```
Speak clearly after any activation phrase (if configured). Ensure ambient noise is minimized.

### 4. Manual Control
```
python src\hand_control.py
```
Runs CV-based finger detection; sends servo angles continuously while landmarks are detected.

### 5. Build & Upload Firmware
#### Option A: PlatformIO GUI (VS Code)
1. Install the PlatformIO VS Code extension (if not already)
2. Clone this repository:
  ```
  git clone <repo-url>
  cd robot-arm-en
  ```
3. Open the project folder in VS Code (File → Open Folder…)
4. Wait for PlatformIO to index and install toolchains (first run may take a few minutes)
5. Select the correct environment (left sidebar → PlatformIO → PROJECT TASKS → env name)
6. Click the checkmark (Build) to compile
7. Connect the board via USB; select the correct serial port if prompted
8. Click the right arrow (Upload) to flash firmware
9. Open Serial Monitor (plug icon) → set baud (e.g., 115200) to observe logs

#### Option B: CLI
```
pio run
pio run -t upload
pio device monitor --baud 115200
```

---
## Configuration Tips
| Area | Setting | Where |
|------|---------|-------|
| EMG sample rate | `sampling_rate` | `emg_plot.py` / `emg_control.py` |
| BITalino channel list | `channels` | same scripts |
| Gesture → Servo map | mapping dict | inside `emg_control.py` (edit) |
| Speech keywords | keyword list / grammar | `stt_control.py` |
| Serial baud rate | value | `main.cpp` + Python scripts |
| CV finger logic | `get_finger_angles` | `hand_control.py` |

---
## Troubleshooting
### BITalino: "communication port does not exist or port in use"
- Close other apps (OpenSignals, previous Python runs) holding the port.
- Verify correct COM port in Device Manager.
- Power cycle BITalino + toggle Bluetooth off/on.
- If MAC address connection fails on Windows, switch to COM port usage.

### Python Error: `SystemError: argument 2 (impossible<bad format char>)`
- Usually indicates wrong address format / stack mismatch. Use COM port instead of MAC on Windows.
- Ensure `pybluez` and `bitalino` packages are up to date.

### Empty / Noisy EMG Signal
- Check electrode placement: muscle belly + reference (ground) over bony area.
- Reduce cable motion; keep good skin contact (low impedance).
- Confirm selected channel matches the wired BITalino input.

### Model Produces Wrong Gesture
- Confirm scaler + model pair (no mixing revisions).
- Inspect feature extraction path: consistent sample window size & order.
- Try capturing a new calibration set and retraining if physiology changed.

### Speech Recognition Poor
- Lower background noise.
- Consider using offline models (e.g., Vosk) for stability:
  - `pip install vosk`
  - Adjust `stt_control.py` to use Vosk if Internet-free operation needed.

---
## Extending
- Add additional EMG channels: expand `channels`, adapt feature extraction to multi-channel vectors.
- Introduce smoothing / majority voting for gesture classification.
- Integrate a higher-level command scheduler (queue + safety watchdog).
- Add web dashboard for real-time telemetry via WebSocket.

---
## Safety Notes
- Always test new motion profiles with reduced speed / torque.
- Provide an immediate emergency stop (hardware button or serial command) when experimenting with new control algorithms.
- Log all incoming high-level commands for audit.

---
## Versioning & Reproducibility
Include in future commits:
- `requirements.txt` for Python dependencies
- Model training notebook / script with version hashes
- CHANGELOG for control protocol revisions

---
## Quick Checklist Before a Demo
- [ ] BITalino paired & COM port confirmed
- [ ] EMG signal quality verified (`emg_plot.py`)
- [ ] Model + scaler files present & matching
- [ ] Firmware uploaded & serial responsive
- [ ] Speech mic test performed
 - [ ] Webcam recognized + hand landmarks detected
- [ ] Emergency stop accessible

---
## License
See `LICENSE` file.

---
## Acknowledgments
- BITalino biosignal toolkit
- Open-source Python ecosystem (NumPy, scikit-learn, matplotlib)

---
Feel free to open issues or extend this document as the project evolves.
