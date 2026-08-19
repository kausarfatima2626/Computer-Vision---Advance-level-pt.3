import cv2
import scipy.spatial.distance as dist

def calculate_ear(eye_points):
    # Compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])

    # Compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye_points[0], eye_points[3])

    # Eye Aspect Ratio (EAR) formula
    ear = (A + B) / (2.0 * C)
    return ear

def process_driver_frame(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Dummy eye landmark points for demonstration (x, y coordinates)
    # In full deployment, these points are extracted via MediaPipe / dlib
    left_eye = [(30, 40), (32, 45), (35, 45), (38, 40), (35, 35), (32, 35)]
    
    ear = calculate_ear(left_eye)
    EAR_THRESHOLD = 0.25

    print(f"Calculated Eye Aspect Ratio (EAR): {ear:.2f}")

    if ear < EAR_THRESHOLD:
        status = "ALERT: Drowsiness Detected!"
        color = (0, 0, 255)  # Red
    else:
        status = "Status: Driver Awake"
        color = (0, 255, 0)  # Green

    cv2.putText(image, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite("drowsiness_output.jpg", image)
    print(f"Result: {status}")

if __name__ == "__main__":
    process_driver_frame("sample.jpg")
