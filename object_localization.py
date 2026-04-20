import cv2

# Load image
img = cv2.imread("image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold to get binary image
_, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours (objects)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If any object is detected
if contours:
    # Take the largest contour (main object)
    c = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(c)

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Show result
cv2.imshow("Object Localization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
