import audio_metadata
import wave
wav = 'S25.wav'
metadata = audio_metadata.load(wav)
print(metadata)

wav = wave.open('S25.wav')
print(wav.getnframes)