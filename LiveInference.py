import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_PATH = "asl_model.h5"
HAND_MODEL_PATH = "hand_landmarker.task"
SEQUENCE_LENGTH = 30

# IMPORTANT: this must be in the exact same order used during training.
# In Colab, this came from: sorted(os.listdir(DATA_PATH))
ACTIONS = ["hello", "no", "please", "thank_you", "yes"]

# Only show a prediction on screen if the model is at least this confident,
# to avoid the label flickering wildly between guesses on uncertain frames.
CONFIDENCE_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

print("Loading model...")
model = load_model(MODEL_PATH)

# Set up MediaPipe's HandLandmarker in LIVE_STREAM-friendly usage. We use
# VIDEO mode (like Phase 2) since we're feeding it a continuous, in-order
# stream of frames with increasing timestamps - the same idea as a video,
# just arriving live instead of all at once.
base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

# Our rolling buffer of the last 30 frames' landmarks.
# deque with maxlen=30 automatically drops the oldest frame once full,
# every time we add a new one - exactly the "sliding window" behavior we want.
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

frame_index = 0
current_prediction = ""
current_confidence = 0.0

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # We don't have real video FPS metadata here (it's a live stream), so we
    # just use a steadily increasing counter as our "timestamp."
    timestamp_ms = frame_index * 33  # roughly matches ~30fps spacing
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    frame_index += 1

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw the 21 landmark points manually (the old convenient
        # draw_landmarks() helper doesn't exist in the new Tasks API).
        h, w, _ = frame.shape
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Build this frame's 63-number landmark array, same format as Phase 2.
        coords = []
        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])
        sequence_buffer.append(coords)
    else:
        # No hand visible - still push a zero-frame, so the buffer keeps
        # moving forward in time consistently (matches how we handled
        # missing detections during data collection in Phase 2).
        sequence_buffer.append([0.0] * 63)

    # Only predict once we have a full 30-frame window, AND only every 5th
    # frame - running the model on every single frame overloads the CPU for
    # no real benefit, since predictions don't need to update 30x/second to
    # look smooth on screen.
    if len(sequence_buffer) == SEQUENCE_LENGTH and frame_index % 5 == 0:
        input_data = np.expand_dims(np.array(sequence_buffer), axis=0).astype(np.float32)

        # Calling the model directly (rather than .predict()) skips a lot of
        # overhead meant for large-batch use cases, which matters here since
        # we're making many small, frequent single-example predictions.
        prediction = model(input_data, training=False).numpy()[0]

        best_index = np.argmax(prediction)
        best_confidence = prediction[best_index]

        if best_confidence >= CONFIDENCE_THRESHOLD:
            current_prediction = ACTIONS[best_index]
            current_confidence = best_confidence
        else:
            current_prediction = "..."
            current_confidence = best_confidence

    # Display the current prediction and confidence on screen.
    display_text = f"{current_prediction} ({current_confidence:.0%})"
    cv2.putText(
        frame, display_text, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
    )

    cv2.imshow("ASL Translator - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()