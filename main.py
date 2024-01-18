import cv2
import numpy as np

def create_mask(image):
    # ... (unchanged)

def apply_mask(image, mask):
    # ... (unchanged)

def preprocess_image(image):
    # Resize the image to a width of 200
    resized_image = imutils.resize(image, width=200)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    return blurred_image

def main():
    # Load your image (replace 'your_image.jpg' with the actual image file path)
    image_path = 'your_image.jpg'
    original_image = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if original_image is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(original_image)

        # Create a mask based on user-drawn polygons
        mask = create_mask(preprocessed_image)

        # Apply the mask to the preprocessed image
        masked_image = apply_mask(preprocessed_image, mask)

        # Display the original, preprocessed, and masked images
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Preprocessed Image", preprocessed_image)
        cv2.imshow("Masked Image", masked_image)

        # Wait for a key event and close the windows when any key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Error: Unable to load the image.")

if __name__ == "__main__":
    main()