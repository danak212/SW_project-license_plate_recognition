import cv2
import json
import argparse
from pathlib import Path
from processing.utils import process_image
from processing.utils import char_det


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("plates_photos")
    arg_parser.add_argument("output_json")
    arguments = arg_parser.parse_args()

    chars_folder = Path("./data/chars")
    characters = char_det(chars_folder)

    photo_dir = Path(arguments.plates_photos)
    result_file = Path(arguments.output_json)

    photo_paths = sorted(
        [
            photo_path
            for photo_path in photo_dir.iterdir()
            if photo_path.name.endswith(".jpg")
        ]
    )

    results = {}
    for photo_path in photo_paths:
        photo = cv2.imread(str(photo_path))
        results[photo_path.name] = process_image(photo, characters)

    with result_file.open("w") as output:
        json.dump(results, output, indent=4)


if __name__ == "__main__":
    main()
