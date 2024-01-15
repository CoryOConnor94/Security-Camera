import cv2
import numpy as np
import imutils

def image_masker(img):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for i in range(0,4):
        bbox = cv2.selectROI(img, False)
        print(bbox)

    pts = np.array([], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked

# Load your image (replace 'your_image.jpg' with the actual image file)
test1 = cv2.imread('your_image.jpg')
gray1 = image_masker("test0.lpg")

# Check if the image is successfully loaded
if test1 is not None:
    # Display the image in a window named "Original"
    cv2.imshow("Original", test1)
    cv2.imshow("Masked image", gray1)

    # Wait for a key event and close the window when any key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Error: Unable to load the image.")