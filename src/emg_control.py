import numpy as np
from bitalino import BITalino
from scipy.signal import iirnotch, filtfilt, firwin, welch
from scipy.stats import skew, kurtosis
import joblib
import time
from collections import Counter
import msvcrt
from serial.tools import list_ports
from serial import Serial

# === Load model and scaler ===
model = joblib.load("final_rf_model.pkl")
scaler = joblib.load("final_scaler.pkl")

# === BITalino settings ===
mac_address = "20:18:06:13:03:33"
sampling_rate = 1000
channels = [0]
buffer_size = 100
signal_buffer = []

# === Filter setup ===
notch_freq = 50.0
q = 30.0
lowcut = 20.0
highcut = 400.0
order = 50

nyq = sampling_rate / 2
b_notch, a_notch = iirnotch(notch_freq, q, sampling_rate)
b_band = firwin(order + 1, [lowcut / nyq, highcut / nyq], pass_zero=False)

# === Arduino connection ===
def find_port():
    ports = list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description:
            return port.device
    return None

try:
    arduino_port = find_port()
    if arduino_port is None:
        print("Arduino not found. Available ports:")
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        arduino_port = 'COM6'  # default port
    arduino = Serial(arduino_port, 9600, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
    time.sleep(2)
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit(1)

# === Gesture to servo angles mapping ===
def gesture_to_angles(gesture):
    """Map detected gesture to servo angles for robot arm control"""
    gesture_map = {
        "rest": [0, 0, 0, 0, 0],           # All fingers open
        "fist": [180, 180, 180, 180, 180], # All fingers closed
        "open": [0, 0, 0, 0, 0],           # All fingers open
        "pinch": [180, 180, 0, 0, 0],      # Thumb and index closed, others open
        "point": [180, 0, 180, 180, 180],  # Only index finger open
        "thumbs_up": [0, 180, 180, 180, 180], # Only thumb open
        "peace": [180, 0, 0, 180, 180],    # Index and middle open
        "ok": [0, 0, 180, 180, 180],       # Thumb, index, middle in OK position
    }
    return gesture_map.get(gesture.lower(), [0, 0, 0, 0, 0])

def send_to_arduino(angles):
    """Send servo angles to Arduino"""
    try:
        command = ','.join(map(str, angles)) + '\n'
        arduino.write(command.encode())
        arduino.flush()
        print(f"📤 Sent to Arduino: {angles}")
    except Exception as e:
        print(f"❌ Error sending to Arduino: {e}")

# === Signal processing functions ===
def apply_filters(sig):
    sig = filtfilt(b_notch, a_notch, sig)
    sig = filtfilt(b_band, 1.0, sig)
    return sig

def extract_time_features(signal):
    peak = np.max(np.abs(signal))
    mean = np.mean(signal)
    
    rms = np.sqrt(np.mean(signal**2))
    std = np.std(signal)
    abs_mean = np.mean(np.abs(signal))
    root_mean_sqrt = np.mean(np.sqrt(np.abs(signal)))
    return {
        "mean": mean,
        "rms": rms,
        "std": std,
        "skew": skew(signal),
        "kurtosis": kurtosis(signal),
        "peak": peak,
        "impulse_factor": peak / abs_mean if abs_mean else 0,
        "crest_factor": peak / rms if rms else 0,
        "clearance_factor": peak / (root_mean_sqrt ** 2) if root_mean_sqrt else 0,
        "shape_factor": rms / abs_mean if abs_mean else 0,
        "snr": 10 * np.log10(np.mean(signal**2) / np.var(signal)) if np.var(signal) else 0
    }

def extract_freq_features(signal, fs):
    f, Pxx = welch(signal, fs=fs, nperseg=len(signal))
    Pxx_norm = Pxx / np.sum(Pxx)
    mean_freq = np.sum(f * Pxx_norm)
    median_freq = f[np.where(np.cumsum(Pxx_norm) >= 0.5)[0][0]]
    peak_freq = f[np.argmax(Pxx)]
    peak_amplitude = np.max(Pxx)
    band_power = np.sum(Pxx)
    occupied_bw = f[np.where(np.cumsum(Pxx_norm) >= 0.95)[0][0]]
    power_bw = np.sqrt(np.sum(Pxx * f**2) / np.sum(Pxx))
    return {
        "mean_freq": mean_freq,
        "median_freq": median_freq,
        "peak_freq": peak_freq,
        "peak_amplitude": peak_amplitude,
        "band_power": band_power,
        "occupied_bw": occupied_bw,
        "power_bw": power_bw
    }

# === Start BITalino ===
device = BITalino(mac_address)
device.battery(20)
device.start(sampling_rate, channels)
print("✅ BITalino connected.")
print("🤖 Robot arm control ready.")

try:
    trial = 1
    last_gesture = None
    
    print("\n🎯 EMG-Controlled Robot Arm")
    print("   - Perform gestures to control the robot arm")
    print("   - Press [Enter] to start recording a gesture")
    print("   - Press [Enter] again to stop recording and execute")
    print("   - Press [Ctrl+C] to exit\n")
    
    while True:
        input(f"➡️ Trial {trial}: Press [Enter] to START gesture recording...")
        print("🎬 Recording... Perform your gesture now.")
        signal_buffer.clear()

        # Record until user presses Enter again
        print("   (Recording... Press [Enter] again to STOP and execute gesture.)")
        start_time = time.time()
        while True:
            if msvcrt.kbhit():
                if msvcrt.getch() == b'\r':  # Enter key
                    break
            data = device.read(buffer_size)
            emg = data[:, 5]
            signal_buffer.extend(emg.tolist())

        print("🛑 Recording stopped.")

        if len(signal_buffer) < 200:
            print("⚠️ Not enough data. Try again.")
            continue

        # === Predict using multiple windows from the recorded signal ===
        all_preds = []
        all_confs = []

        # Process signal in overlapping windows
        for i in range(0, len(signal_buffer) - 200 + 1, 100):  # 50% overlap
            window = np.array(signal_buffer[i:i+200]).astype(float)
            window -= np.mean(window)
            filtered = apply_filters(window)

            feats = extract_time_features(filtered)
            feats.update(extract_freq_features(filtered, fs=sampling_rate))
            vec = np.array([feats[k] for k in feats])
            scaled = scaler.transform([vec])
            pred = model.predict(scaled)[0]
            conf = np.max(model.predict_proba(scaled))

            if conf > 0.6:  # Only consider high-confidence predictions
                all_preds.append(pred)
                all_confs.append(conf)

        if all_preds:
            # Get majority vote
            vote = Counter(all_preds).most_common(1)[0][0]
            avg_conf = np.mean(all_confs)
            
            print(f"🧠 Detected gesture: {vote} (confidence: {avg_conf:.2f})")
            
            # Only execute if confidence is high enough and gesture changed
            if avg_conf > 0.7 and vote != last_gesture:
                angles = gesture_to_angles(vote)
                send_to_arduino(angles)
                last_gesture = vote
                print(f"🤖 Robot arm executing: {vote}")
            elif vote == last_gesture:
                print(f"🔄 Same gesture as before, no action taken")
            else:
                print(f"⚠️ Low confidence, no action taken")
        else:
            print("❌ No confident predictions, no action taken")

        trial += 1

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    device.stop()
    device.close()
    if 'arduino' in locals():
        arduino.close()
    print("🔌 Disconnected from BITalino and Arduino.")
