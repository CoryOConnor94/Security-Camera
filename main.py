import cv2
import numpy as np
import subprocess
import time


def apply_mask(image, mask):
    """
    Apply a binary mask to an image.

    Args:
    - image: Input image.
    - mask: Binary mask.

    Returns:
    - masked_image: Image with the applied mask.
    """
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess_image(image):
    """
    Preprocess an image by resizing, converting to grayscale, and blurring.

    Args:
    - image: Input image.

    Returns:
    - preprocessed_image: Preprocessed image.
    """
    resized_image = cv2.resize(image, (200, int(200 / image.shape[1] * image.shape[0])))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image


def capture_images(command, num_images):
    """
    Capture images using the raspistill command.

    Args:
    - command: Raspistill command for image capture.
    - num_images: Number of images to capture.
    """
    for i in range(num_images):
        subprocess.run(command, shell=True)
        time.sleep(1)


def main():
    """
    Main function for capturing and processing images in a loop.
    """
    capture_command = 'raspistill -w 1000 -h 720 -t 1000 -tl 1000 -o test%02d.jpg'
    num_images = 2

    # Load previously determined ROI polygons
    polygons = np.load('roi_polygons.npy', allow_pickle=True)

    while True:
        capture_images(capture_command, num_images)

        for i in range(1, num_images + 1):
            image_path = f'test{i:02d}.jpg'
            original_image = cv2.imread(image_path)

            if original_image is not None:
                preprocessed_image = preprocess_image(original_image)

                mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype="uint8")
                cv2.fillPoly(mask, polygons, 255)

                masked_image = apply_mask(preprocessed_image, mask)

                cv2.imshow(f"Preprocessed Image {i}", preprocessed_image)
                cv2.imshow(f"Masked Image {i}", masked_image)

            else:
                print(f"Error: Unable to load the image {i}.")

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
