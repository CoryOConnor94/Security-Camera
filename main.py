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
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same size for comparison.")

    diff = cv2.absdiff(image1, image2)
    _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    motion_detected = np.any(diff_mask > 0)

    if motion_detected:
        print("Motion Detected!")

    return motion_detected


def record_video(output_file, duration=15, width=1280, height=720):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"{timestamp}.h264"
    print("")
    # Construct the libcamera-vid command
    command = [
        "libcamera-vid",
        "-t", str(duration * 1000),  # Convert seconds to milliseconds
        "--width", str(width),
        "--height", str(height),
        "-o", output_path
    ]

    try:
        # Run the libcamera-vid command
        subprocess.run(command, check=True)

        # Optionally, convert the video to MP4
        mp4_output_file = output_file.replace(".h264", ".mp4")
        subprocess.run(["MP4Box", "-add", output_path, mp4_output_file])

        print(f"Video saved as {mp4_output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def main():
    capture_command = 'libcamera-jpeg -t 1000 -o test%d.jpg'
    num_images = 2
    polygons = np.load('roi_polygons.npy', allow_pickle=True)

    previous_processed_image = None
    motion_detected = False

    while True:
        capture_images(capture_command, num_images)

        for i in range(num_images):
            image_path = f'test{i}.jpg'
            original_image = cv2.imread(image_path)

            if original_image is not None:
                mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype="uint8")
                cv2.fillPoly(mask, polygons, 255)

                masked_image = apply_mask(original_image, mask)
                current_processed_image = preprocess_image(masked_image)

                cv2.imwrite(f"Original_Image_{i}.jpg", original_image)
                cv2.imwrite(f"Preprocessed_Image_{i}.jpg", current_processed_image)
                cv2.imwrite(f"Masked_Image_{i}.jpg", masked_image)

                if previous_processed_image is not None:
                    motion_detected = compare_images(previous_processed_image, current_processed_image)
                    if motion_detected:
                        print("Motion detected")
                        # timestamp = time.strftime("%Y%m%d_%H%M%S")
                        # video_output_file = f"Motion_Video_{timestamp}.mp4"
                        record_video("output", duration=15, width=1280, height=720)
                        time.sleep(2)  # Add a delay to avoid consecutive recordings for the same motion

                previous_processed_image = current_processed_image

            else:
                print(f"Error: Unable to load the image {i}.")

        print("Images saved successfully. Waiting for the next capture.")


if __name__ == "__main__":
    main()
