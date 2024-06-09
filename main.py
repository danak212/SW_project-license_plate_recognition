import cv2
import json
import sys
from pathlib import Path
from processing.utils import process_image
from processing.utils import char_det


def main():
    # Getting the arguments from the command line
    # Pobieranie argumentów z linii poleceń
    plates_photos = sys.argv[1]
    output_json = sys.argv[2]

    # Path to the folder containing character images
    # Ścieżka do folderu zawierającego obrazy znaków
    chars_folder = Path("./data/chars")
    characters = char_det(chars_folder)

    # Path to the directory with photos of plates
    # Ścieżka do katalogu ze zdjęciami tablic rejestracyjnych
    photo_dir = Path(plates_photos)
    result_file = Path(output_json)

    # Sorting photo paths and filtering by .jpg extension
    # Sortowanie ścieżek do zdjęć i filtrowanie po rozszerzeniu .jpg
    photo_paths = sorted(
        [
            photo_path
            for photo_path in photo_dir.iterdir()
            if photo_path.name.endswith(".jpg")
        ]
    )

    # Dictionary to store the results
    # Słownik do przechowywania wyników
    results = {}
    for photo_path in photo_paths:
        # Reading each photo
        # Odczytanie każdego zdjęcia
        photo = cv2.imread(str(photo_path))
        # Processing the photo and storing the result
        # Przetwarzanie zdjęcia i przechowywanie wyniku
        results[photo_path.name] = process_image(photo, characters)

    # Saving results to a JSON file
    # Zapisanie wyników do pliku JSON
    with result_file.open("w") as output:
        json.dump(results, output, indent=4)


if __name__ == "__main__":
    main()
