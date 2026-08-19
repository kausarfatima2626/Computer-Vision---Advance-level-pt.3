import cv2
import numpy as np

def extract_license_plate(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be loaded.")
        return

    # Grayscale & Bilateral Filter for edge preservation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge Detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours and filter for rectangular plate shape
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:  # License plates are 4-sided quadrilaterals
            location = approx
            break

    if location is not None:
        # Draw contour over detected plate
        cv2.drawContours(image, [location], -1, (0, 255, 0), 3)
        print("License Plate Region Successfully Isolated!")
    else:
        print("License Plate Region Not Found.")

    cv2.imwrite("alpr_output.jpg", image)
    print("Output saved as 'alpr_output.jpg'")

if __name__ == "__main__":
    extract_license_plate("sample.jpg")
