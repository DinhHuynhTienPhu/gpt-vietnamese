import pyaudio
import wave
import torch
from transformers import pipeline
import pyttsx3
import numpy as np
from g4f.client import Client

# Load PhoWhisper model (use a model of your choice)
model_name = "vinai/PhoWhisper-small"  # Change to other models if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
transcriber = pipeline("automatic-speech-recognition", model=model_name, device=device)
tts_engine = pyttsx3.init()

# Initialize the client
client = Client()

# Function for speech to text
def listen():
    """Capture audio from the microphone and transcribe it to text."""
    # Open microphone
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()

    print("Listening...")

    # Record audio
    frames = []
    silence_threshold = 50  # Adjust this threshold based on your testing
    silent_chunks = 0
    max_silent_chunks = 12  # Number of silent chunks before stopping
    chunk_size = 2048  # Size of each audio chunk
    total_chunks = 0

    while True:
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)

        # Convert byte data to numpy array to analyze audio
        audio_data = np.frombuffer(data, dtype=np.int16)

        volume = np.sqrt(np.mean(audio_data**2))

        # Print out volume and silence information
        
        print(f"Volume: {volume:.5f}, Silence Threshold: {silence_threshold}, Silent Chunks: {silent_chunks}")
        total_chunks += 1
        # Check if volume is below the silence threshold
        if volume > silence_threshold:
            silent_chunks += 1
            # Stop recording if we've been silent for long enough
            if silent_chunks >= max_silent_chunks or total_chunks >= 100:
                break
        else:
            silent_chunks = 0  # Reset if we hear something

    print("Finished listening.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    mic.terminate()

    # Save the audio to a temporary WAV file for processing
    audio_file = "temp.wav"
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Use PhoWhisper to transcribe the audio
    output = transcriber(audio_file)['text']
    return output

# Function for text to speech
def speak(text):
    """Convert text to speech."""
    # Set voice property to Vietnamese
    voices = tts_engine.getProperty('voices')

    # Find the Vietnamese voice if available
    for voice in voices:
        if "vietnam" in voice.languages or "vi" in voice.id:
            tts_engine.setProperty('voice', voice.id)
            break

    tts_engine.say(text)
    tts_engine.runAndWait()

# Main chatbot loop
if __name__ == "__main__":
    while True:
        # Listen for user input
        user_input = listen()

        # Check if the user wants to exit
        if user_input is None:
            continue
        if user_input.lower() in ["thoát", "kết thúc", "dừng lại"]:
            print("Kết thúc trò chuyện...")
            break

        print("Bạn: " + user_input)
        print("Đang chờ phản hồi...")

        # Create a chat completion request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}],
        )

        # Get the model's response
        gpt_response = response.choices[0].message.content
        print("GPT: " + gpt_response)

        # Speak the response
        speak(gpt_response)
