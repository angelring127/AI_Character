import pyaudio
import wave
import subprocess

p = pyaudio.PyAudio()

devices = p.get_device_count()
# Iterate through all devices
for i in range(devices):
    # Get the device info
    device_info = p.get_device_info_by_index(i)
    # Check if this device is a microphone (an input device)
    if device_info.get('maxInputChannels') > 0:
        print(f"입력: {device_info.get('name')} , Device Index: {device_info.get('index')}")
    else:
        print(f"출력: {device_info.get('name')} , Device Index: {device_info.get('index')}")


def play_wav_file(filename):
    # Open the file
    # Get the number of audio I/O devices
    p = pyaudio.PyAudio()

    command = ["ffmpeg", "-y", "-i", filename, "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le", "output_wav.wav"]
    subprocess.run(command, check=True)
    wav_file = wave.open("output_wav.wav", 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(wav_file.getsampwidth()),
                    channels=wav_file.getnchannels(),
                    rate=wav_file.getframerate(),
                    output_device_index=5,
                    output=True)
    streamTwo = p.open(format=p.get_format_from_width(wav_file.getsampwidth()),
                    channels=wav_file.getnchannels(),
                    rate=wav_file.getframerate(),
                    output=True)

    # Read data from the file
    data = wav_file.readframes(1024)

    # Play the file by streaming the data
    while data:
        stream.write(data)
        streamTwo.write(data)
        data = wav_file.readframes(1024)

    # Close the stream and terminate the PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Use the function
play_wav_file('./server/xtts.wav')