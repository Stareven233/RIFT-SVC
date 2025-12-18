import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from functools import partial
import soundfile as sf
import resampy
import numpy as np
import pyloudnorm as pyln
from tqdm import tqdm

ROOT_DIR = Path(__file__, '../..').resolve()


def process_audio_file(file_path: Path, target_sample_rate, target_loudness, headroom=0.99, target_dir=None):
  '''
    Resamples and normalizes the loudness of an audio file.

    Parameters:
    - file_path (Path): Path to the audio file.
    - target_sample_rate (int): Desired sample rate in Hz.
    - target_loudness (float): Desired loudness in LUFS.
    - headroom (float): Maximum absolute amplitude after scaling (default: 0.99).
    - target_dir (Path): RIFT-SVC data dir, containing speaker subdirs.

    Returns:
    - str: Success message or error details.
    '''
  try:
    # Read the audio file
    data, samplerate = sf.read(file_path)

    # If audio is stereo, convert to mono by averaging the channels
    if len(data.shape) > 1 and data.shape[1] > 1:
      data = np.mean(data, axis=1)

    # Resample the audio to the target sample rate if necessary
    if samplerate != target_sample_rate:
      data = resampy.resample(data, samplerate, target_sample_rate)
      samplerate = target_sample_rate

    # Initialize the meter for loudness normalization (ITU-R BS.1770)
    meter = pyln.Meter(samplerate)  # Create BS.1770 meter

    # Measure the loudness of the audio
    loudness = meter.integrated_loudness(data)

    # Normalize the loudness to the target level
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, target_loudness)

    # Handle clipping by scaling audio if necessary
    max_amp = np.max(np.abs(loudness_normalized_audio))
    clipped = False
    scaling_factor = 1.0

    if max_amp > 1.0:
      scaling_factor = headroom / max_amp
      loudness_normalized_audio = loudness_normalized_audio * scaling_factor
      clipped = True

    # Write the resampled and loudness-normalized audio back to the same path
    if target_dir is not None:
      file_path = target_dir / file_path.name
    sf.write(file_path, loudness_normalized_audio, samplerate)

    if clipped:
        return (
          f'Processed (Clipped Handled): {file_path} | '
          f'Original LUFS: {loudness:.2f} | Normalized to: {target_loudness} LUFS | '
          f'Scaling Factor Applied: {scaling_factor:.4f}'
        )
    else:
        return (
          f'Processed: {file_path} | '
          f'Original LUFS: {loudness:.2f} | Normalized to: {target_loudness} LUFS'
        )

  except Exception as e:
    return f'Error processing {file_path}: {e}'


def find_sf_audio_paths(source_path: Path):
  '''
    Gathers all soundfile-supported audio files from the source directory.

    Parameters:
    - source_root (Path): Root directory of the source.

    Returns:
    - list of Path: List of audio file paths.
    '''
  audio_files = [f for f in source_path.iterdir() if f.suffix[1:].upper() in sf.available_formats()]
  return audio_files


def resample_normalize_audios(source_root, target_root=None, target_sample_rate=44100, target_loudness=-18.0, headroom=0.99, max_workers=None):
  '''
    Resamples and normalizes loudness for all soundfile-supported audio files in the source directory.

    Parameters:
    - source_root (Path): Path to the root of the source directory (one of the speaker subdir, containing audios).
    - target_root (Path): Path to the root of the target directory.
    - target_sample_rate (int): Desired sample rate in Hz (e.g., 22050, 44100).
    - target_loudness (float): Desired loudness in LUFS (e.g., -18.0 for pure vocal singing).
    - headroom (float): Maximum absolute amplitude after scaling to prevent clipping (default: 0.99).
    - max_workers (int): Maximum number of worker processes. Defaults to number of CPUs.
    '''
  # Gather all relevant audio files
  audio_files = find_sf_audio_paths(source_root)
  total_files = len(audio_files)

  if total_files == 0:
    print('No audio files found to process.')
    return

  print(f'Found {total_files} audio files to resample and normalize.')

  # Define a partial function with fixed arguments
  target_dir: Path = target_root / source_root.stem
  target_dir.mkdir(exist_ok=False)
  worker = partial(
    process_audio_file,
    target_sample_rate=target_sample_rate,
    target_loudness=target_loudness,
    headroom=headroom,
    target_dir=target_dir,
  )

  # Use ProcessPoolExecutor for CPU-bound tasks
  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_file = {executor.submit(worker, file_path): file_path for file_path in audio_files}

    # Use tqdm for progress bar
    for future in tqdm(as_completed(future_to_file), total=total_files, desc='Processing'):
      result = future.result()
      # Optionally, you can log the result or print it
      # print(f'\n{result}')  # Uncomment to see detailed logs

  print(f'Resampling and normalization in \'{source_root}\' complete.')


def parse_arguments():
  '''
    Parses command-line arguments.

    Returns:
    - argparse.Namespace: Parsed arguments.
    '''
  parser = argparse.ArgumentParser(description='Resample and normalize audio files in a directory.')

  parser.add_argument('--src', type=Path, required=True, help='Path to the organized speaker directory containing audio files. Or the directory containing all spreaker subdirs')
  parser.add_argument('--dest', type=Path, required=False, default=(ROOT_DIR / 'data'), help='Path to the target directory containing speaker subdirs. (audio files stored in subdirs)')
  parser.add_argument('--speakers', type=str, nargs='+', required=False, default=None, help='Speakers to be trained, default all subdir in --source_dir')
  parser.add_argument('--target_sample_rate', type=int, default=44100, help='Desired sample rate in Hz (default: 44100).')
  parser.add_argument('--target_loudness', type=float, default=-18.0, help='Desired loudness in LUFS (default: -18.0).')
  parser.add_argument('--headroom', type=float, default=0.99, help='Maximum absolute amplitude after scaling to prevent clipping (default: 0.99).')
  parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker processes (default: number of CPUs).')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_arguments()

  src: Path = args.src
  assert src.is_dir(), f'{src=} must be directory'
  speaker_dirs = None
  if args.speakers is not None:
    # Build the source audio path for the specified training speakers.
    speaker_dirs = tuple(src / name for name in args.speakers)
  else:
    # Treat all subdirectories in `src` as individual speaker data folders.
    speaker_dirs = tuple(d for d in src.iterdir() if d.is_dir())
  if len(speaker_dirs) == 0:
    # It is assumed that `src` contains all speaker audio for training, and `src.stem` is the speaker's name.
    speaker_dirs = (src,)

  # Resample and normalize the audio files based on provided arguments
  for s in speaker_dirs:
    resample_normalize_audios(
      source_root=s,
      target_root=args.dest,
      target_sample_rate=args.target_sample_rate,
      target_loudness=args.target_loudness,
      headroom=args.headroom,
      max_workers=args.max_workers
    )
