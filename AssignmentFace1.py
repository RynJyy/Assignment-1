import threading
import cv2
from deepface import DeepFace

# Initialize Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS-compatible backend
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load Reference Image
reference_img = cv2.imread("/Users/ryanjay/PycharmProjects/pythonProject/.venv/Face/Photo on 2-25-25 at 3.11 PM.jpg")
if reference_img is None:
    print("Error: Reference image not found!")
    exit()

def check_face(frame):
    global face_match
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.verify(frame_rgb, reference_img.copy())['verified']
        with lock:
            face_match = result
    except Exception as e:
        print(f"Error in face verification: {e}")
        with lock:
            face_match = False

def start_face_verification(frame):
    thread = threading.Thread(target=check_face, args=(frame.copy(),))
    thread.daemon = True
    thread.start()

# Global Variables
face_match = False
lock = threading.Lock()

counter = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to grab frame")
        continue
    if counter % 38 == 8:  # Run face verification asynchronously every 38 frames
        start_face_verification(frame)
    counter += 1
    with lock:
        text = "MATCH! (press space to leave)" if face_match else "NO MATCH! (press space to leave)"
        color = (0, 255, 0) if face_match else (0, 0, 255)
    cv2.putText(frame, text, (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
    cv2.imshow("video", frame)
    key = cv2.waitKey(10)
    if key == ord(" "):
        break

cap.release()
cv2.destroyAllWindows()
