import os
import json
from collections import defaultdict


def get_processed_filename(file_name):
    return file_name.split('_')[0] if '_' in file_name else file_name


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def collect_confidence_scores(main_dir):
    json_data = defaultdict(lambda: defaultdict(list))
    for folder_name in os.listdir(main_dir):
        folder_path = os.path.join(main_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    processed_name = get_processed_filename(file_name)
                    file_path = os.path.join(folder_path, file_name)
                    json_content = load_json(file_path)
                    for item in json_content.get("confidence_scores", []):
                        text = item.get("text")
                        confidence = item.get("confidence", 0)
                        json_data[processed_name][text].append((folder_name, confidence))
    return json_data


def print_best_confidence(json_data):
    for json_name, texts in json_data.items():
        print(f"Results for JSON name: {json_name}")
        for text, scores in texts.items():
            best_dir, best_score = max(scores, key=lambda x: x[1])
            print(f"  Text: {text} -> Best confidence: {best_score} in folder: {best_dir}")


def get_best_confidence(main_dir):
    json_data = collect_confidence_scores(main_dir)
    print_best_confidence(json_data)


def main():
    main_directory = 'data/ocr_output'
    get_best_confidence(main_directory)

if __name__ == '__main__':
    main()
