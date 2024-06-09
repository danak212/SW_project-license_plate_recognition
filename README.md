#### ======= ENGLISH BELOW =======

# SW Project License Plate Recognition - 2024

## Przegląd Projektu

Celem tego projektu jest napisanie programu rozpoznającego tablice rejestracyjne na zdjęciach wykonanych smartfonem lub aparatem fotograficznym zgodnie z określonymi założeniami. Projekt przetwarza zdjęcia tablic rejestracyjnych i wyodrębnia tekst z tych tablic.

## Struktura Projektu

Projekt składa się z następujących plików i skryptów:
- `main.py`: Orkiestruje proces rozpoznawania tablic rejestracyjnych.
- `utils.py`: Zawiera funkcje pomocnicze do przetwarzania obrazów i rozpoznawania znaków.

## Zależności

- Python 3.11
- OpenCV
- NumPy
- Imutils

## Kroki i Implementacja

#### 1. Pobieranie Argumentów

Skrypt `main.py` pobiera argumenty z linii poleceń, które zawierają ścieżkę do folderu ze zdjęciami oraz ścieżkę do pliku wyjściowego JSON.

```python
plates_photos = sys.argv[1]
output_json = sys.argv[2]
```

#### 2. Przetwarzanie Obrazów

Funkcja `process_image` przetwarza obrazy tablic rejestracyjnych, wykonując m.in. filtrację bilateralną, progowanie Otsu oraz dopasowywanie wzorca. Po konwersji obrazu na skalę szarości, funkcja zwiększa kontrast i aplikuje filtr bilateralny, aby zredukować szumy. Następnie lokalizuje potencjalne tablice rejestracyjne i wyodrębnia znaki z tych tablic.

```python
def process_image(image: np.ndarray, chars) -> str:
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    contrast_enhanced = cv.convertScaleAbs(gray_image, alpha=1.1)
    filtered_image = cv.bilateralFilter(contrast_enhanced, 20, 50, 50)
    plate_candidates, candidate_boxes = locate_plate(filtered_image, gray_image)
    plate_numbers = [extract_plate_chars(enhanced_plate, chars, i) for i, enhanced_plate in enumerate(plate_candidates)]
    return max(plate_numbers, key=len, default="P012345")
```

#### 3. Rozpoznawanie Znaków

Funkcja `char_matching` dopasowuje wycięte znaki do wzorców zapisanych w folderze z obrazami znaków, wykorzystując metodę dopasowania szablonów. Dopasowuje wycięte znaki do wzorców zapisanych w folderze z obrazami znaków, znajdując znak o najwyższym wyniku dopasowania.

```python
def char_matching(char, chars):
    highest_score = 0
    best_match = None
    for key in chars:
        match_result = cv.matchTemplate(char, chars[key], cv.TM_CCOEFF_NORMED)
        _, max_value, _, _ = cv.minMaxLoc(match_result)
        if max_value > highest_score:
            best_match = key
            highest_score = max_value
    return str(best_match)
```

#### 4. Zapis Wyników

Wyniki rozpoznawania są zapisywane do pliku JSON, który zawiera mapowanie nazw zdjęć na wykryte tablice rejestracyjne.

```python
with result_file.open("w") as output:
    json.dump(results, output, indent=4)
```

## Dodatkowe Funkcje

#### 5. char_det

Funkcja `char_det` ładuje obrazy znaków z określonego folderu i zapisuje je w słowniku, gdzie kluczami są nazwy plików bez rozszerzeń.

#### 6. extract_plate_chars

Funkcja `extract_plate_chars` wyodrębnia znaki z obrazu tablicy rejestracyjnej, znajduje kontury znaków i dopasowuje je do wzorców znaków.

```python
def extract_plate_chars(plate_img, char_set, idx):
    plate_bgr = cv.cvtColor(plate_img, cv.COLOR_GRAY2BGR)
    contours_found, _ = cv.findContours(
        cv.bitwise_not(plate_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours_found, bboxes = contours.sort_contours(contours_found, method="left-to-right")
    potential_chars = [contour for contour in contours_found if valid_contour(contour, plate_img)]
    roi_characters = [warp_perspective(contour, plate_img) for contour in potential_chars]
    plate_text = [char_matching(cv.resize(roi, (64, 64)), char_set) for roi in roi_characters]
    return "".join(plate_text)
```

#### 7. enhance_plate_image

Funkcja `enhance_plate_image` poprawia jakość obrazu tablicy rejestracyjnej poprzez zastosowanie filtrów bilateralnych, rozmycia gaussowskiego i progowania Otsu.

```python
def enhance_plate_image(plate_img, orig_img, idx):
    plate_img = cv.bilateralFilter(plate_img, 20, 50, 50)
    plate_img = cv.blur(plate_img, (7, 7))
    ret, otsu_thresh = cv.threshold(plate_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    otsu_thresh = cv.erode(otsu_thresh, np.ones((7, 7), np.uint8))
    return otsu_thresh
```

#### 8. locate_plate

Funkcja `locate_plate` używa filtru Canny'ego do wykrywania krawędzi i znajdowania konturów tablic rejestracyjnych na obrazie.

```python
def locate_plate(img: np.ndarray, gray_img):
    edges = cv.Canny(img, 30, 45)
    struct_element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    processed_img = cv.dilate(edges, struct_element, iterations=1)
    contours, _ = cv.findContours(processed_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    plate_candidates = [gray_img[y:y+h, x:x+w] for contour in contours if valid_plate_contour(contour, img)]
    return plate_candidates, [cv.boundingRect(contour) for contour in contours if valid_plate_contour(contour, img)]
```

## Jak Uruchomić

1. Upewnij się, że masz zainstalowany Python 3.11.
2. Zainstaluj wymagane biblioteki używając `pip install -r requirements.txt`.
3. Uruchom skrypt `main.py`, podając jako parametry ścieżkę do folderu ze zdjęciami oraz ścieżkę do pliku wyjściowego JSON:
    ```bash
    python main.py /ścieżka/do/folderu/ze/zdjęciami /ścieżka/do/pliku/wyjściowego.json
    ```

### Wynik

Wynik zostanie zapisany w pliku JSON, zawierającym rozpoznane tablice rejestracyjne dla każdego zdjęcia.

### Podsumowanie

Ten projekt demonstruje, jak wykorzystać przetwarzanie obrazów i algorytmy dopasowania wzorców do rozpoznawania tablic rejestracyjnych. Proces obejmuje przetwarzanie obrazu, ekstrakcję znaków oraz dopasowywanie wzorców, co prowadzi do dokładnego rozpoznawania tablic rejestracyjnych na zdjęciach.

#
#

#### ======= ENGLISH VERSION =======

# SW Project License Plate Recognition - 2024

## Project Overview

The goal of this project is to develop a program that recognizes license plates in photos taken with a smartphone or camera according to specific assumptions. The project processes license plate images and extracts the text from these plates.

## Project Structure

The project consists of the following files and scripts:
- `main.py`: Orchestrates the license plate recognition process.
- `utils.py`: Contains helper functions for image processing and character recognition.

## Dependencies

- Python 3.11
- OpenCV
- NumPy
- Imutils

## Steps and Implementation

#### 1. Argument Parsing

The `main.py` script parses command-line arguments, which include the path to the folder with photos and the path to the output JSON file.

```python
plates_photos = sys.argv[1]
output_json = sys.argv[2]
```

#### 2. Image Processing

The `process_image` function processes license plate images by performing bilateral filtering, Otsu thresholding, and template matching. After converting the image to grayscale, the function enhances the contrast and applies a bilateral filter to reduce noise. It then locates potential license plates and extracts characters from these plates.

```python
def process_image(image: np.ndarray, chars) -> str:
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    contrast_enhanced = cv.convertScaleAbs(gray_image, alpha=1.1)
    filtered_image = cv.bilateralFilter(contrast_enhanced, 20, 50, 50)
    plate_candidates, candidate_boxes = locate_plate(filtered_image, gray_image)
    plate_numbers = [extract_plate_chars(enhanced_plate, chars, i) for i, enhanced_plate in enumerate(plate_candidates)]
    return max(plate_numbers, key=len, default="P012345")
```

#### 3. Character Recognition

The `char_matching` function matches extracted characters to templates stored in a folder of character images, using template matching. It matches the extracted characters to the templates, finding the character with the highest matching score.

```python
def char_matching(char, chars):
    highest_score = 0
    best_match = None
    for key in chars:
        match_result = cv.matchTemplate(char, chars[key], cv.TM_CCOEFF_NORMED)
        _, max_value, _, _ = cv.minMaxLoc(match_result)
        if max_value > highest_score:
            best_match = key
            highest_score = max_value
    return str(best_match)
```

#### 4. Saving Results

The recognition results are saved to a JSON file that maps photo names to detected license plates.

```python
with result_file.open("w") as output:
    json.dump(results, output, indent=4)
```

## Additional Functions

#### 5. char_det

The `char_det` function loads character images from a specified folder and stores them in a dictionary, where the keys are the filenames without extensions.

#### 6. extract_plate_chars

The `extract_plate_chars` function extracts characters from a license plate image, finds character contours, and matches them to character templates.

```python
def extract_plate_chars(plate_img, char_set, idx):
    plate_bgr = cv.cvtColor(plate_img, cv.COLOR_GRAY2BGR)
    contours_found, _ = cv.findContours(
        cv.bitwise_not(plate_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours_found, bboxes = contours.sort_contours(contours_found, method="left-to-right")
    potential_chars = [contour for contour in contours_found if valid_contour(contour, plate_img)]
    roi_characters = [warp_perspective(contour, plate_img) for contour in potential_chars]
    plate_text = [char_matching(cv.resize(roi, (64, 64)), char_set) for roi in roi_characters]
    return "".join(plate_text)
```

#### 7. enhance_plate_image

The `enhance_plate_image` function improves the quality of the license plate image by applying bilateral filters, Gaussian blur, and Otsu thresholding.

```python
def enhance_plate_image(plate_img, orig_img, idx):
    plate_img = cv.bilateralFilter(plate_img, 20, 50, 50)
    plate_img = cv.blur(plate_img, (7, 7))
    ret, otsu_thresh = cv.threshold(plate_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    otsu_thresh = cv.erode(otsu_thresh, np.ones((7, 7), np.uint8))
    return otsu_thresh
```

#### 8. locate_plate

The `locate_plate` function uses the Canny filter to detect edges and find contours of license plates in the image.

```python
def locate_plate(img: np.ndarray, gray_img):
    edges = cv.Canny(img, 30, 45)
    struct_element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    processed_img = cv.dilate(edges, struct_element, iterations=1)
    contours, _ = cv.findContours(processed_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    plate_candidates = [gray_img[y:y+h, x:x+w] for contour in contours if valid_plate_contour(contour, img)]
    return plate_candidates, [cv.boundingRect(contour) for contour in contours if valid_plate_contour(contour, img)]
```

## How to Run

1. Ensure you have Python 3.11 installed.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the `main.py` script, providing the path to the folder with photos and the path to the output JSON file:
    ```bash
    python main.py /path/to/photo/folder /path/to/output.json
    ```

### Result

The result will be saved in a JSON file, containing the recognized license plates for each photo.

### Summary

This project demonstrates how to use image processing and template matching algorithms to recognize license plates. The process includes image processing, character extraction, and template matching, resulting in accurate recognition of license plates in photos.
