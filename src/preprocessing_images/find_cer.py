import os
import difflib


def calculate_cer(prediction, ground_truth):
    prediction = prediction.replace(" ", "").lower()
    ground_truth = ground_truth.replace(" ", "").lower()

    edit_distance = 0
    seq_matcher = difflib.SequenceMatcher(None, prediction, ground_truth)
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete':
            edit_distance += i2 - i1
        elif tag == 'insert':
            edit_distance += j2 - j1

    cer = edit_distance / len(ground_truth) if len(ground_truth) > 0 else 0
    return cer


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def check_cer_in_directory(dir_path):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    ground_truth_file = None
    for file in files:
        if 'ground_truth' in file:
            ground_truth_file = file

    ground_truth_path = os.path.join(dir_path, ground_truth_file)
    ground_truth_text = read_file(ground_truth_path)

    for file in files:
        if file != ground_truth_file:
            file_path = os.path.join(dir_path, file)
            file_text = read_file(file_path)

            cer = calculate_cer(file_text, ground_truth_text)
            print(f"CER for {file}: {cer:.4f}")


dir_path = "data/text"
check_cer_in_directory(dir_path)
