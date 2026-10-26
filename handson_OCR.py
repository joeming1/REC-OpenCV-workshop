"""Live camera OCR demo with adjustable HSV masking and threshold trackbars.

Install the Python packages with ``pip install opencv-python pytesseract`` and
install the Tesseract OCR application separately (Windows: UB Mannheim builds
are commonly used). Run with ``python handson_OCR.py``. Press Q or Esc to quit.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

if pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
CAMERA_INDEX = 0
OCR_INTERVAL_SECONDS = 1.0


def nothing(_value):
    """OpenCV trackbar callback placeholder."""


def get_controls():
    h_min = cv2.getTrackbarPos("H min", "Mask controls")
    h_max = cv2.getTrackbarPos("H max", "Mask controls")
    s_min = cv2.getTrackbarPos("S min", "Mask controls")
    s_max = cv2.getTrackbarPos("S max", "Mask controls")
    v_min = cv2.getTrackbarPos("V min", "Mask controls")
    v_max = cv2.getTrackbarPos("V max", "Mask controls")
    threshold = cv2.getTrackbarPos("Threshold", "Mask controls")
    return (
        np.array([min(h_min, h_max), min(s_min, s_max), min(v_min, v_max)]),
        np.array([max(h_min, h_max), max(s_min, s_max), max(v_min, v_max)]),
        threshold,
    )


def make_masked_ocr_image(frame, lower, upper, threshold):
    """Apply HSV color masking, then prepare a clean black-on-white OCR image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower, upper)
    selected = cv2.bitwise_and(frame, frame, mask=color_mask)

    # Make everything outside the selected color white so it won't be read as ink.
    selected[color_mask == 0] = (255, 255, 255)
    gray = cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return color_mask, binary


def recognize_letters(binary_image):
    """Find and recognize the largest single letter-shaped ink region."""
    ink = 255 - binary_image
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if width >= 4 and height >= 8 and area >= 20:
            candidates.append((area, x, y, width, height))
    if not candidates:
        return []

    _, x, y, width, height = max(candidates, key=lambda item: item[0])
    padding = max(10, height // 3)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(binary_image.shape[1], x + width + padding)
    y1 = min(binary_image.shape[0], y + height + padding)
    character = binary_image[y0:y1, x0:x1]
    character = cv2.copyMakeBorder(character, 24, 24, 24, 24,
                                   cv2.BORDER_CONSTANT, value=255)
    character = cv2.resize(character, None, fx=3, fy=3,
                           interpolation=cv2.INTER_CUBIC)
    config = (
        "--oem 3 --psm 10 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )
    prediction = pytesseract.image_to_string(character, config=config).strip()
    prediction = "".join(char for char in prediction if char.isalpha())
    if not prediction:
        return []
    # Deliberately return just one character and one box.
    return [(x, y, width, height, prediction[0])]


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}. Check that it is connected and available.")
        return 1
    # A smaller capture size reduces per-frame processing and display latency.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Mask controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask controls", 440, 320)
    cv2.createTrackbar("H min", "Mask controls", 0, 179, nothing)
    cv2.createTrackbar("H max", "Mask controls", 179, 179, nothing)
    cv2.createTrackbar("S min", "Mask controls", 0, 255, nothing)
    cv2.createTrackbar("S max", "Mask controls", 255, 255, nothing)
    cv2.createTrackbar("V min", "Mask controls", 0, 255, nothing)
    cv2.createTrackbar("V max", "Mask controls", 255, 255, nothing)
    cv2.createTrackbar("Threshold", "Mask controls", 160, 255, nothing)

    if pytesseract is None:
        print("pytesseract is missing. Install it with: pip install pytesseract")
        print("The camera and masking windows will still work; OCR is disabled.")

    last_ocr_time = 0.0
    last_text = ""
    last_boxes = []
    warned_tesseract = False
    ocr_executor = ThreadPoolExecutor(max_workers=1) if pytesseract is not None else None
    ocr_future = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            lower, upper, threshold = get_controls()
            mask, ocr_image = make_masked_ocr_image(frame, lower, upper, threshold)

            now = time.monotonic()
            # Poll completed OCR in the camera thread, but never wait for it.
            if ocr_future is not None and ocr_future.done():
                try:
                    last_boxes = ocr_future.result()
                    last_text = last_boxes[0][4] if last_boxes else ""
                except Exception as error:
                    last_text = "Tesseract unavailable"
                    last_boxes = []
                    if not warned_tesseract:
                        print(f"OCR could not start: {error}")
                        warned_tesseract = True
                ocr_future = None

            if (ocr_executor is not None and ocr_future is None
                    and now - last_ocr_time >= OCR_INTERVAL_SECONDS):
                last_ocr_time = now
                # Work on a copy so the worker never sees a reused frame buffer.
                ocr_future = ocr_executor.submit(recognize_letters, ocr_image.copy())

            display = frame.copy()
            for x, y, w, h, letter in last_boxes:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 2)
                cv2.putText(display, letter, (x, max(y - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 0), 2, cv2.LINE_AA)

            cv2.putText(display, "Letter: " + (last_text or "(none detected)"),
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow("Camera OCR", display)
            cv2.imshow("Color mask", mask)
            cv2.imshow("OCR input", ocr_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if ocr_executor is not None:
            ocr_executor.shutdown(wait=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
