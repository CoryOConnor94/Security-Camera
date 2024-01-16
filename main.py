import cv2
import numpy as np


def create_mask(image):
    """
    Create a binary mask based on user-selected regions.
    Args:
    - image: Input image to create the mask.
    Returns:
    - mask: Binary mask with selected regions.
    """
    mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")

    # Allow user to select four regions
    for _ in range(4):
        bbox = cv2.selectROI(image, False)
        # You can do something with the bounding box if needed
        print("Selected ROI:", bbox)

    # You need to define points (pts) before using fillConvexPoly
    pts = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)

    return mask


def apply_mask(image, mask):
    """
    Apply the provided mask to the input image.
    Args:
    - image: Input image.
    - mask: Binary mask.
    Returns:
    - masked_image: Image with the applied mask.
    """
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def main():
    # Load your image (replace 'your_image.jpg' with the actual image file path)
    image_path = 'your_image.jpg'
    original_image = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if original_image is not None:
        # Create a mask based on user-selected regions
        mask = create_mask(original_image)

        # Apply the mask to the original image
        masked_image = apply_mask(original_image, mask)

        # Display the original and masked images
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Masked Image", masked_image)

        # Wait for a key event and close the windows when any key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Error: Unable to load the image.")


if __name__ == "__main__":
    main()
