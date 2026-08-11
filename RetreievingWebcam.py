import cv2

# Open the default webcam.
# The "0" refers to the first camera device connected to your system.
# If you have multiple cameras (e.g. a built-in laptop cam + a USB webcam),
# you may need to try 1, 2, etc. instead.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

# This is the main loop. It runs once per frame, continuously,
# until you press 'q'. Every video app you'll ever build follows this pattern:
# read a frame -> process it -> display it -> repeat.
while True:
    # cap.read() grabs one frame from the camera.
    # ret is True/False (did it succeed?), frame is the actual image data
    # (a NumPy array of pixel values).
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Flip the frame horizontally so it acts like a mirror.
    # Without this, moving your hand right makes it appear to move left
    # on screen, which feels unnatural.
    frame = cv2.flip(frame, 1)

    # Display the frame in a window titled "ASL Translator - Webcam Feed".
    cv2.imshow("ASL Translator - Webcam Feed", frame)

    # Wait 1 millisecond for a key press. If the key pressed is 'q', break
    # out of the loop and stop the program.
    # The "& 0xFF" is a standard bitmask to make this work reliably across OSes.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Always release the camera and close windows when you're done.
# Skipping this can leave your webcam "locked" by Python, so later programs
# fail to access it until you restart.
cap.release()
cv2.destroyAllWindows()

