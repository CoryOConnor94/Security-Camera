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
    Capture images using the libcamera-jpeg command.

    Args:
    - command: libcamera-jpeg command for image capture.
    - num_images: Number of images to capture.
    """
    for i in range(num_images):
        subprocess.run(command % i, shell=True)  # Use % to insert the image index in the command
        time.sleep(1)


def compare_images(image1, image2, threshold=20):
    """
    Compare two images pixel by pixel and return True if motion is detected.

    Args:
    - image1: First processed image.
    - image2: Second processed image.
    - threshold: Threshold for pixel difference.

    Returns:
    - motion_detected: Boolean indicating whether motion is detected.
    """
    # Ensure images are of the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same size for comparison.")

    # Calculate absolute difference between images
    diff = cv2.absdiff(image1, image2)

    # Create a binary mask where pixel differences exceed the threshold
    _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Check if there are non-zero pixels in the diff_mask
    motion_detected = np.any(diff_mask > 0)

    return motion_detected


def record_video(video_command):
    subprocess.run(video_command, shell=True)


def email_alert():
    pass


def sms_alert():
    pass


def upload_to_cloud():
    pass


def main():
    """
    Main function for capturing, processing, and saving images in a continuous loop.
    """
    # libcamera-jpeg command to capture images for the loop
    capture_command = 'libcamera-jpeg -t 1000 -o test%d.jpg'
    num_images = 2

    # Load previously determined ROI polygons
    polygons = np.load('roi_polygons.npy', allow_pickle=True)

    previous_processed_image = None  # Store the previous processed image for comparison

    while True:
        # Capture two images and process them
        capture_images(capture_command, num_images)

        for i in range(num_images):
            image_path = f'test{i}.jpg'
            original_image = cv2.imread(image_path)

            if original_image is not None:
                mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype="uint8")
                cv2.fillPoly(mask, polygons, 255)

                masked_image = apply_mask(original_image, mask)
                current_processed_image = preprocess_image(masked_image)

                # Compare current processed image with the previous one
                if previous_processed_image is not None:
                    motion_detected = compare_images(previous_processed_image, current_processed_image)
                    if motion_detected:
                        print("Motion Detected!!")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        video_command = f'libcamera-vid -t 9000 --width 1280 --height 720 -o {timestamp}.h264'
                        record_video(video_command)
                        print("Finished Recording")

                        time.sleep(2)  # Add a delay to avoid consecutive recordings for the same motion
                        # Save the original, preprocessed, and masked images
                        cv2.imwrite(f"Original_Image_{i}.jpg", original_image)
                        cv2.imwrite(f"Preprocessed_Image_{i}.jpg", current_processed_image)
                        cv2.imwrite(f"Masked_Image_{i}.jpg", masked_image)
                        email_alert()
                        sms_alert()
                        upload_to_cloud()
                previous_processed_image = current_processed_image

            else:
                print(f"Error: Unable to load the image {i}.")


if __name__ == "__main__":
    main()
