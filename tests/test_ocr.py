import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from google.cloud import vision
from src.preprocessing_images.ocr import check_google_credentials, google_ocr, apply_ocr_on_image, ocr_images


def create_mock_response():
    mock_response = Mock()

    # Create mock text annotations
    text_annotation = Mock()
    text_annotation.description = "Sample OCR text"
    mock_response.text_annotations = [text_annotation]

    # Create mock full text annotation
    page = Mock()
    block = Mock()
    paragraph = Mock()
    word = Mock()
    symbol = Mock()

    symbol.text = "test"
    word.symbols = [symbol]
    word.confidence = 0.95

    paragraph.words = [word]
    block.paragraphs = [paragraph]
    page.blocks = [block]

    full_text_annotation = Mock()
    full_text_annotation.pages = [page]
    mock_response.full_text_annotation = full_text_annotation

    return mock_response


@pytest.fixture
def mock_vision_client():
    with patch('src.preprocessing_images.ocr.vision_client') as mock_client:
        mock_client.annotate_image.return_value = create_mock_response()
        yield mock_client


@pytest.fixture
def test_data_dir(tmp_path):
    """Create test directory structure with all necessary subdirectories"""
    # Create main directories
    image_dir = tmp_path / "data" / "original_pecing_images" / "test_folder"
    ocr_dir = tmp_path / "data" / "ocr_output" / "original"

    # Create all directories
    image_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    # Create test image
    test_image = image_dir / "test_image.jpg"
    test_image.write_bytes(b"fake image content")

    # Print directory structure for debugging
    print("\nTest directory structure:")
    for path in tmp_path.rglob("*"):
        print(f"Created: {path}")

    return tmp_path


def test_check_google_credentials():
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    with pytest.raises(EnvironmentError):
        check_google_credentials()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/fake/path/credentials.json"
    check_google_credentials()


def test_google_ocr(mock_vision_client, tmp_path):
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image content")

    mock_image = MagicMock()

    with patch('src.preprocessing_images.ocr.vision.Image', return_value=mock_image) as MockImage:
        result = google_ocr(str(image_path))
        assert isinstance(result, dict)
        assert "text" in result
        assert result["text"] == "Sample OCR text"

        binary_content = image_path.read_bytes()
        result = google_ocr(binary_content)
        assert isinstance(result, dict)
        assert "text" in result


def test_apply_ocr_on_image(test_data_dir, mock_vision_client):
    image_path = test_data_dir / "data" / "original_pecing_images" / "test_folder" / "test_image.jpg"
    ocr_dir = test_data_dir / "data" / "ocr_output" / "original" / "test_folder"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    with patch('src.preprocessing_images.ocr.vision.Image'):
        apply_ocr_on_image(image_path, ocr_dir)

        expected_output = ocr_dir / "test_image.json"
        assert expected_output.exists()

        with open(expected_output) as f:
            result = json.load(f)
            assert "text" in result
            assert "confidence_scores" in result


def test_ocr_images(test_data_dir, mock_vision_client):
    # Set up and verify input directory
    images_dir = test_data_dir / "data" / "original_pecing_images"
    assert images_dir.exists(), f"Input directory does not exist: {images_dir}"

    # List contents of input directory for debugging
    print("\nContents of input directory:")
    for path in images_dir.rglob("*"):
        print(f"Found: {path}")

    # Run OCR with mocked Image class
    with patch('src.preprocessing_images.ocr.vision.Image'):
        ocr_images(images_dir)

    # Verify output directory structure
    output_base = test_data_dir / "data" / "ocr_output" / "original"
    output_dir = output_base / "test_folder"
    expected_output = output_dir / "test_image.json"

    print(f"\nVerifying output paths:")
    print(f"Base output directory: {output_base}")
    print(f"Expected output directory: {output_dir}")
    print(f"Expected output file: {expected_output}")

    assert output_base.exists(), f"Base output directory not created: {output_base}"
    assert output_dir.exists(), f"Output directory not created: {output_dir}"
    assert expected_output.exists(), f"Output file not created: {expected_output}"

    # Verify content
    with open(expected_output) as f:
        result = json.load(f)
        assert "text" in result
        assert "confidence_scores" in result


def test_skip_processed_folders(test_data_dir, mock_vision_client):
    output_dir = test_data_dir / "data" / "ocr_output" / "original" / "test_folder"
    output_dir.mkdir(parents=True)

    images_dir = test_data_dir / "data" / "original_pecing_images"

    ocr_images(images_dir)
    mock_vision_client.annotate_image.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
