import cv2
import numpy as np

def analyze_grayscale_image(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load the image.")
        return

    # Display the dimensions of the image
    print(f"Image Dimensions: {image.shape}")

    # Convert image to a matrix
    pixel_matrix = np.array(image)
    print("Pixel Matrix:")
    print(pixel_matrix)

    # Get basic statistics about the pixel values
    min_val = np.min(pixel_matrix)
    max_val = np.max(pixel_matrix)
    mean_val = np.mean(pixel_matrix)
    std_val = np.std(pixel_matrix)

    print(f"\nPixel Value Statistics:")
    print(f"Min Value: {min_val}")
    print(f"Max Value: {max_val}")
    print(f"Mean Value: {mean_val:.2f}")
    print(f"Standard Deviation: {std_val:.2f}")

    # Show the image
    cv2.imshow("Grayscale Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'PCA_non_optimised/data/walk_small.png'  # Replace with the path to your grayscale image
analyze_grayscale_image(image_path)
