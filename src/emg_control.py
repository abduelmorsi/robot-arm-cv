import numpy as np
import pandas as pd
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
sampling_rate = 1000  # Hz
channels = [0]  # Using channel 0 for EMG data
buffer_size = 200  # Number of samples to read at once
signal_buffer = []  # Buffer to store the signal data

# === Filter setup ===
notch_freq = 50.0  # Notch filter frequency to remove powerline noise
quality_factor = 30.0  # Quality factor for notch filter
low_cutoff = 20.0  # Low cutoff frequency for bandpass filter
high_cutoff = 400.0  # High cutoff frequency for bandpass filter
filter_order = 50  # Filter order (ensure signal is long enough)

nyq = sampling_rate / 2  # Nyquist frequency
low = low_cutoff / nyq
high = high_cutoff / nyq

# Filter design
b_notch, a_notch = iirnotch(notch_freq, quality_factor, sampling_rate)
b_bandpass = firwin(filter_order + 1, [low, high], pass_zero=False)

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

# === Feature Extraction Functions ===
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
        "impulse_factor": peak / abs_mean if abs_mean != 0 else 0,
        "crest_factor": peak / rms if rms != 0 else 0,
        "clearance_factor": peak / (root_mean_sqrt ** 2) if root_mean_sqrt != 0 else 0,
        "shape_factor": rms / abs_mean if abs_mean != 0 else 0,
        "snr": 10 * np.log10(np.mean(signal**2) / np.var(signal)) if np.var(signal) != 0 else 0
    }

def extract_freq_features(signal, fs):
    f, Pxx = welch(signal, fs=fs)
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
        while True:
            if msvcrt.kbhit():  # Wait for user to press Enter
                if msvcrt.getch() == b'\r':  # Enter key
                    break
            data = device.read(buffer_size)  # Read EMG data from BITalino
            emg = data[:, 5]  # Extract EMG signal from the appropriate channel (index 5)
            signal_buffer.extend(emg.tolist())  # Add data to the buffer

        print("🛑 Recording stopped.")

        if len(signal_buffer) < 1000:  # Check if enough data has been recorded
            print("⚠️ Not enough data. Try again.")
            continue

        # === Real-time Filtering and Prediction ===
        all_preds = []
        all_confs = []

        # Process the signal in 1000-sample windows with 50% overlap
        window_size = 1000  # Each window is 1 second long (1000 samples at 1000 Hz)
        overlap = 500  # 50% overlap, sliding by 500 samples each time

        for i in range(0, len(signal_buffer) - window_size + 1, overlap):
            window = np.array(signal_buffer[i:i + window_size]).astype(float)
            window -= np.mean(window)  # Remove DC offset

            # Apply the notch filter
            filtered = filtfilt(b_notch, a_notch, window)

            # Apply the bandpass filter
            filtered = filtfilt(b_bandpass, 1.0, filtered)  # Apply bandpass filter

            # Extract features from the filtered window
            feats = extract_time_features(filtered)
            feats.update(extract_freq_features(filtered, fs=sampling_rate))  # Add frequency features
            vec = np.array([feats[k] for k in feats])
            
            # Normalize the feature vector using the pre-trained scaler
            scaled = scaler.transform([vec])

            # Make a prediction using the trained model
            pred = model.predict(scaled)[0]
            conf = np.max(model.predict_proba(scaled))  # Get prediction confidence

            # Print prediction results
            print(f"🧠 Predicted: {pred} | Confidence: {conf:.2f} | RMS: {feats['rms']:.2f} | Peak: {feats['peak']:.1f} | MFreq: {feats['mean_freq']:.1f} Hz")
            all_preds.append(pred)
            all_confs.append(conf)

        if all_preds:
            # Majority voting for the final gesture prediction
            vote = Counter(all_preds).most_common(1)[0][0]
            avg_conf = np.mean(all_confs)
            
            print(f"✅ Final decision for Trial {trial}: Gesture = {vote} (majority vote)")
            print(f"🧠 Average confidence: {avg_conf:.2f}")
            
            # Only execute if confidence is high enough and gesture changed
            if avg_conf > 0.7 and vote != last_gesture:
                angles = gesture_to_angles(vote)
                send_to_arduino(angles)
                last_gesture = vote
                print(f"🤖 Robot arm executing: {vote}")
            elif vote == last_gesture:
                print(f"🔄 Same gesture as before, no action taken")
            else:
                print(f"⚠️ Low confidence ({avg_conf:.2f}), no action taken")
        else:
            print("❌ No predictions made")

        trial += 1

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    device.stop()
    device.close()
    if 'arduino' in locals():
        arduino.close()
    print("🔌 Disconnected from BITalino and Arduino.")
