import openai
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
openai.api_key = OPENAI_API_KEY

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds per command

audio_queue = queue.Queue()

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

def record_audio(duration=DURATION):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return recording.flatten()

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
        audio_file = open(tmp.name, 'rb')
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        audio_file.close()
    return transcript['text']

# Map text command to servo angles

def command_to_angles(command):
    # Simple mapping, expand as needed
    command = command.lower()
    if "open hand" in command:
        return [0, 0, 0, 0, 0]
    elif "close hand" in command:
        return [180, 180, 180, 180, 180]
    elif "thumb up" in command:
        return [0, 180, 180, 180, 180]
    elif "peace" in command:
        return [180, 0, 0, 180, 180]
    # Add more commands as needed
    else:
        print(f"Unknown command: {command}")
        return None

# Main loop

def main():
    print("Say a command (e.g., 'open hand', 'close hand', 'thumb up', 'peace')...")
    while True:
        audio_data = record_audio()
        try:
            text = transcribe_audio(audio_data)
            print(f"Recognized: {text}")
            angles = command_to_angles(text)
            if angles:
                command = ','.join(map(str, angles)) + '\n'
                arduino.write(command.encode())
                arduino.flush()
                print(f"Sent angles: {angles}")
            else:
                print("No valid command recognized.")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    main()
