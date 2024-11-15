import cv2
import numpy as np
import pytest
from src.preprocessing_images.preprocess import preprocess_image, process_images_in_directory


@pytest.fixture
def setup_test_images(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = input_dir / "test_image.jpg"
    cv2.imwrite(str(img_path), img)

    return input_dir, output_dir, img_path


def test_preprocess_image(setup_test_images):
    _, _, img_path = setup_test_images
    gray_image = preprocess_image(str(img_path))

    assert gray_image is not None
    assert len(gray_image.shape) == 2


def test_process_images_in_directory(setup_test_images):
    input_dir, output_dir, img_path = setup_test_images
    process_images_in_directory(str(input_dir), str(output_dir))

    processed_image_path = output_dir / "test_image_processed.jpg"
    assert processed_image_path.exists()

    processed_image = cv2.imread(str(processed_image_path), cv2.IMREAD_GRAYSCALE)
    assert processed_image is not None
    assert len(processed_image.shape) == 2
