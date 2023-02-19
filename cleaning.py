import os
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
import sox

# Set the paths to the input and output directories
input_dir = 'path/to/input/directory'
output_dir = 'path/to/output/directory'

# Set the sample rate and duration of the audio segments
sample_rate = 16000
segment_duration = 5 * 1000

# Load the metadata file containing the audio filenames and transcriptions
metadata_file = os.path.join(input_dir, 'metadata.csv')
metadata = pd.read_csv(metadata_file, header=None, names=['filename', 'transcription'])

# Process each audio file in the input directory
for filename in metadata['filename']:
    # Load the audio file
    audio_file = os.path.join(input_dir, filename)
    audio = AudioSegment.from_file(audio_file)

    # Downsample the audio to the appropriate sample rate
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)

    # Split the audio into smaller chunks based on silence
    chunks = split_on_silence(audio, min_silence_len=100, silence_thresh=-16)

    # Save each chunk to a separate WAV file
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f'{filename}_{i}.wav')
        chunk.export(chunk_file, format='wav')

# Set the paths to the input and output directories
input_dir = 'path/to/preprocessed/audio/files'
output_file = 'path/to/csv/file.csv'

# Load the list of audio files
audio_files = sorted(os.listdir(input_dir))

# Generate the CSV file
data = {'wav_filename': [], 'transcription': []}

def get_transcription(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    return '_'.join(basename.split('_')[1:])

for audio_file in audio_files:
    transcription = get_transcription(audio_file)
    data['wav_filename'].append(os.path.join(input_dir, audio_file))
    data['transcription'].append(transcription)

df = pd.DataFrame.from_dict(data)
df.to_csv(output_file, index=False)