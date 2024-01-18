import cv2
import numpy as np

def create_mask(image):
    """
    Create a binary mask based on user-selected regions using polygons.

    Args:
    - image: Input image to create the mask.

    Returns:
    - mask: Binary mask with selected regions.
    """
    mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")

    print("Select ROI and then press SPACE or ENTER button!")
    print("Cancel the selection process by pressing 'c' button!")

    polygons = []
    while True:
        roi = cv2.selectROI("Select ROI (Press Enter to finish)", image, showCrosshair=False)
        if roi[2] == 0 or roi[3] == 0:
            break  # Break if user presses Enter without selecting a region
        # Append vertices of the rectangle to polygons
        polygons.append(np.array([[roi[0], roi[1]], [roi[0] + roi[2], roi[1]], [roi[0] + roi[2], roi[1] + roi[3]], [roi[0], roi[1] + roi[3]]], dtype=np.int32))

    cv2.fillPoly(mask, polygons, 255)

    return mask, polygons

def main():
    """
    Main function for ROI selection.
    """
    image_path = 'your_image.jpg'
    original_image = cv2.imread(image_path)

    if original_image is not None:
        mask, polygons = create_mask(original_image)
        cv2.imshow("Selected ROI", cv2.bitwise_and(original_image, original_image, mask=mask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        np.save('roi_polygons.npy', polygons)  # Save the selected polygons to a file
    else:
        print("Error: Unable to load the image.")

if __name__ == "__main__":
    main()