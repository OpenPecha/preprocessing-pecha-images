import cv2
import os


def preprocess_image(image_path):
    """
    Preprocesses an image by converting it to grayscale.

    Args:
        image_path (str): The file path to the image to be processed.

    Returns:
        numpy.ndarray: The processed grayscale image.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def process_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_directory, filename)
            processed_image = preprocess_image(image_path)
            base, ext = os.path.splitext(filename)
            processed_image_path = os.path.join(output_directory, f"{base}_processed{ext}")
            cv2.imwrite(processed_image_path, processed_image)
            print(f"Processed image saved at: {processed_image_path}")


def main():
    input_directory = 'data/original_pecing_images/W1KG13126'
    output_directory = 'data/preprocessed_images/W1KG13126'
    process_images_in_directory(input_directory, output_directory)


if __name__ == '__main__':
    main()
