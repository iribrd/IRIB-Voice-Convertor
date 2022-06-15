import audio_metadata
wav = 's15xs15.wav'
metadata = audio_metadata.load(wav)
print(metadata)