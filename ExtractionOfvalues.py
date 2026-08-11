import json
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# CONFIG - adjust these paths if your folder names are different
# ---------------------------------------------------------------------------

# Folder where you extracted the Kaggle zip. Adjust if yours is named differently.
WLASL_ROOT = "wlasl_data"

# The master metadata file listing every word and its video examples.
JSON_PATH = os.path.join(WLASL_ROOT, "WLASL_v0.3.json")

# The list of video IDs known to be broken/unavailable - we skip these.
MISSING_PATH = os.path.join(WLASL_ROOT, "missing.txt")

# The folder containing the actual .mp4 files, named like "00335.mp4".
# ADJUST THIS if your videos are in a differently-named subfolder.
VIDEOS_DIR = os.path.join(WLASL_ROOT, "videos")

# Where we'll save our extracted landmark .npy files.
OUTPUT_DIR = "MP_Data"

# The 5 words we're building our prototype dataset from.
# Note: this dataset uses "thank you" (two words), not "thanks".
TARGET_WORDS = ["hello", "thank you", "please", "yes", "no"]

# Every video gets standardized to this many frames.
SEQUENCE_LENGTH = 30

# ---------------------------------------------------------------------------
# SETUP - load metadata
# ---------------------------------------------------------------------------

with open(JSON_PATH, "r") as f:
    wlasl_data = json.load(f)

# missing.txt is a plain list of video IDs, one per line - load into a set
# for fast lookup ("is this ID missing?" checks).
with open(MISSING_PATH, "r") as f:
    missing_ids = set(line.strip() for line in f if line.strip())

# Set up MediaPipe's HandLandmarker (the new "Tasks API", replacing the old
# mp.solutions.hands interface which was removed in MediaPipe 1.0.0).
# This requires a model file, hand_landmarker.task, sitting in this same
# folder - see the setup instructions for where to download it from.
MODEL_PATH = "hand_landmarker.task"

# ---------------------------------------------------------------------------
# CORE FUNCTION - process a single video into a fixed-length landmark array
# ---------------------------------------------------------------------------

def extract_landmarks_from_video(video_path):
    """
    Reads every frame of a video, runs hand-tracking on each frame, and
    returns a list of landmark arrays (one per frame that had a hand
    detected). Frames with no hand detected are recorded as all-zeros,
    so timing/length is preserved even if tracking briefly fails.
    """
    # Create a brand new tracker for THIS video only. The Tasks API tracks
    # timestamps internally and requires them to always increase - reusing
    # one tracker across multiple videos would mean the 2nd video's
    # timestamps (starting back at 0) go "backwards" relative to the 1st
    # video's, which raises an error. A fresh tracker per video sidesteps
    # this entirely, since each one starts its own clock at 0.
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(video_path)

    # The new API needs a timestamp (in milliseconds) for each frame, so it
    # can track motion between frames correctly. We calculate how many
    # milliseconds pass per frame based on this video's frame rate.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30  # fallback, in case a video's metadata doesn't report fps
    ms_per_frame = 1000 / fps

    frame_landmarks = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the frame in MediaPipe's own Image type, which the new API
        # requires instead of a plain NumPy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_index * ms_per_frame)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            # Take the first (and, since num_hands=1, only) detected hand.
            hand = result.hand_landmarks[0]
            # Flatten the 21 landmarks' (x, y, z) into one list of 63 numbers.
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            frame_landmarks.append(coords)
        else:
            # No hand detected this frame - fill with zeros so every frame
            # still contributes a row of the same shape.
            frame_landmarks.append([0.0] * 63)

        frame_index += 1

    cap.release()
    landmarker.close()  # frees up the resources this tracker was using
    return frame_landmarks


def standardize_length(frames, target_length=SEQUENCE_LENGTH):
    """
    Takes a list of frames (each a list of 63 numbers) and returns exactly
    target_length frames, either by evenly sampling down (if too long) or
    by repeating frames to pad up (if too short).
    """
    frames = np.array(frames)
    current_length = len(frames)

    if current_length == 0:
        # Video had zero readable frames - return an all-zero sequence
        # rather than crashing, so one bad video doesn't stop the whole run.
        return np.zeros((target_length, 63))

    # np.linspace picks evenly spaced indices across the video, e.g. if a
    # video has 90 frames and we want 30, it picks every 3rd frame.
    # If the video is SHORTER than 30 frames, this naturally repeats some
    # indices instead, padding it back up to 30.
    indices = np.linspace(0, current_length - 1, target_length).astype(int)
    return frames[indices]


# ---------------------------------------------------------------------------
# MAIN LOOP - go word by word, video by video
# ---------------------------------------------------------------------------

for word in TARGET_WORDS:
    # Find this word's entry in the WLASL metadata (case-insensitive match).
    entry = next(
        (item for item in wlasl_data if item["gloss"].lower() == word.lower()),
        None
    )

    if entry is None:
        print(f"Warning: '{word}' not found in WLASL_v0_3.json - skipping.")
        continue

    # Create an output folder for this word, e.g. MP_Data/hello/
    word_output_dir = os.path.join(OUTPUT_DIR, word.replace(" ", "_"))
    os.makedirs(word_output_dir, exist_ok=True)

    saved_count = 0

    for instance in entry["instances"]:
        video_id = instance["video_id"]

        # Skip videos we already know are broken/missing.
        if video_id in missing_ids:
            continue

        video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")

        # Skip if the file genuinely isn't on disk (dataset may not include
        # every single instance listed in the JSON).
        if not os.path.exists(video_path):
            continue

        frames = extract_landmarks_from_video(video_path)
        sequence = standardize_length(frames)

        # Save as MP_Data/hello/0.npy, MP_Data/hello/1.npy, etc.
        save_path = os.path.join(word_output_dir, f"{saved_count}.npy")
        np.save(save_path, sequence)
        saved_count += 1

    print(f"'{word}': saved {saved_count} examples to {word_output_dir}")

print("Done extracting landmarks for all target words.")