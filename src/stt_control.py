from openai import OpenAI


import sounddevice as sd
import numpy as np
import queue
import time
import os
from dotenv import load_dotenv
from serial.tools import list_ports
from serial import Serial

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY in your .env file")
client = OpenAI(api_key=OPENAI_API_KEY)
# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds per command

audio_queue = queue.Queue()

# Microphone selection
def list_microphones():
    """List all available microphones"""
    devices = sd.query_devices()
    input_devices = []
    
    print("\n🎤 Available microphones:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            print(f"  {len(input_devices)-1}: {device['name']} (ID: {i})")
    
    return input_devices

def select_microphone():
    """Let user select a microphone"""
    input_devices = list_microphones()
    
    if not input_devices:
        print("❌ No input devices found!")
        return None
    
    try:
        choice = input(f"\nSelect microphone (0-{len(input_devices)-1}) or press Enter for default: ").strip()
        
        if choice == "":
            print("Using default microphone")
            return None
        
        choice = int(choice)
        if 0 <= choice < len(input_devices):
            device_id = input_devices[choice][0]
            device_name = input_devices[choice][1]['name']
            print(f"✅ Selected: {device_name}")
            return device_id
        else:
            print("❌ Invalid choice, using default")
            return None
            
    except ValueError:
        print("❌ Invalid input, using default")
        return None

# Find Arduino port

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

# Record audio

def record_audio(duration=DURATION, device_id=None):
    print(f"Recording for {duration} seconds... Speak clearly!")
    print("3... 2... 1... GO!")
    time.sleep(1)  # Give user time to prepare
    
    recording = sd.rec(
        int(duration * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        dtype='float32',
        device=device_id
    )
    sd.wait()
    
    # Check audio level
    audio_level = np.max(np.abs(recording))
    print(f"Audio level: {audio_level:.4f}")
    
    if audio_level < 0.01:
        print("⚠️ Warning: Audio level very low. Speak louder or check microphone.")
    
    # Convert to int16 for compatibility
    recording_int16 = (recording * 32767).astype(np.int16)
    return recording_int16.flatten()

# Send audio to Whisper API

def transcribe_audio(audio_data):
    import tempfile
    import wave
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        with open(tmp.name, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
    return transcript.text

# Map text command to servo angles

def command_to_angles(command):
    # Simple mapping, expand as needed
    command = command.lower().strip()
    
    print(f"Processing command: '{command}'")
    
    # More flexible matching
    if any(word in command for word in ["open", "open hand", "spread", "aç"]):
        return [0, 0, 0, 0, 0]
    elif any(word in command for word in ["close", "fist", "close hand", "clench"]):
        return [180, 180, 180, 180, 180]
    elif any(word in command for word in ["thumb up", "thumbs up", "like"]):
        return [0, 180, 180, 180, 180]
    elif any(word in command for word in ["peace", "victory", "two"]):
        return [180, 0, 0, 180, 180]
    elif any(word in command for word in ["point", "pointing", "index"]):
        return [180, 0, 180, 180, 180]
    elif any(word in command for word in ["ok", "okay", "circle"]):
        return [0, 0, 180, 180, 180]
    # Add more commands as needed
    else:
        print(f"Unknown command: '{command}'")
        print("Available commands: open hand, close hand, thumb up, peace, point, ok")
        return None

# Main loop

def main():
    print("🎤 Voice-Controlled Robot Arm")
    
    # Select microphone
    selected_mic = select_microphone()
    
    print("\nAvailable commands:")
    print("  - 'open hand' or 'open'")
    print("  - 'close hand' or 'fist'")
    print("  - 'thumb up' or 'like'")
    print("  - 'peace' or 'victory'")
    print("  - 'point' or 'pointing'")
    print("  - 'ok' or 'okay'")
    print("\nPress Ctrl+C to exit\n")
    
    while True:
        try:
            input("Press Enter to start recording...")
            audio_data = record_audio(device_id=selected_mic)
            
            print("🔄 Transcribing...")
            text = transcribe_audio(audio_data)
            print(f"📝 Recognized: '{text}'")
            
            if text.strip() == "" or text.strip() in [".", ",", "!"]:
                print("❌ No speech detected. Try speaking louder or closer to microphone.")
                continue
                
            angles = command_to_angles(text)
            if angles:
                command = ','.join(map(str, angles)) + '\n'
                arduino.write(command.encode())
                arduino.flush()
                print(f"🤖 Sent to robot: {angles}")
                print("✅ Command executed!\n")
            else:
                print("❌ No valid command recognized.\n")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Continuing...\n")

if __name__ == "__main__":
    main()
