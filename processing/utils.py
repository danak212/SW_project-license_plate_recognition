import cv2 as cv
import numpy as np
from imutils import contours


def char_det(source):
    chars = {}
    char_files = list(source.iterdir())

    # Iterating through files and loading images
    # Iterowanie po plikach i wczytywanie obrazów
    for char_file in char_files:
        # Getting the filename without the extension as the character name
        # Pobranie nazwy pliku bez rozszerzenia jako nazwy znaku
        char_name = char_file.stem

        # Reading the character image as a grayscale image
        # Wczytanie obrazu znaku jako obraz w skali szarości
        char_image = cv.imread(str(char_file), cv.IMREAD_GRAYSCALE)

        # Storing the image in a dictionary with the character name as the key
        # Przechowywanie obrazu w słowniku z nazwą znaku jako klucz
        chars[char_name] = char_image

    return chars


def char_matching(char, chars):
    # Function to match an extracted character to a set of characters
    # Funkcja dopasowująca wycięty znak do zestawu znaków
    highest_score = 0
    best_match = None

    # Iterating over the keys in the character dictionary
    # Iterowanie po kluczach w słowniku znaków
    for key in chars:
        # Converting images to uint8 type
        # Przekształcenie obrazów na typ uint8
        chars[key] = cv.convertScaleAbs(chars[key])
        char = cv.convertScaleAbs(char)

        match_result = cv.matchTemplate(char, chars[key], cv.TM_CCOEFF_NORMED)

        # Finding the minimum and maximum values and their locations
        # Znalezienie minimalnej i maksymalnej wartości oraz ich lokalizacji
        min_val, max_value, min_loc, max_loc = cv.minMaxLoc(match_result)

        if max_value > highest_score:
            best_match = key
            highest_score = max_value

    return str(best_match)


def extract_plate_chars(plate_img, char_set, idx):
    # Convert the license plate image to BGR
    # Konwersja obrazu tablicy rejestracyjnej do BGR
    plate_bgr = cv.cvtColor(plate_img, cv.COLOR_GRAY2BGR)

    # Find external contours of the characters on the inverted mask
    # Szukanie konturów zewnętrznych znaków na odwróconej masce
    contours_found, _ = cv.findContours(
        cv.bitwise_not(plate_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    try:
        # Sort contours from left to right
        # Sortowanie konturów od lewej do prawej
        (contours_found, bboxes) = contours.sort_contours(contours_found, method="left-to-right")
    except:
        pass

    potential_chars = []
    for contour in contours_found:
        x, y, w, h = cv.boundingRect(contour)
        # Select contours with appropriate proportions
        # Wybieranie konturów o odpowiednich proporcjach
        if h >= plate_img.shape[0] / 3 and 0.13 <= w / h <= 1.22:
            potential_chars.append(contour)

    # Draw contours on the plate image
    # Rysowanie konturów na obrazie tablicy
    cv.drawContours(plate_bgr, potential_chars, -1, (0, 255, 0), 2)

    roi_characters = []
    # Create a bounding box for each candidate and save it to the list
    # Tworzenie bounding boxa dla każdego kandydata i zapis do listy
    for candidate in potential_chars:
        x, y, w, h = cv.boundingRect(candidate)
        if h < 0.5 * plate_img.shape[0]:
            continue
        # Define source and destination points for perspective transformation
        # Definiowanie punktów źródłowych i docelowych do transformacji perspektywicznej
        src_pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype="float32")
        dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype="float32")
        # Compute the perspective transformation matrix and their apply
        # Wyznaczanie macierzy transformacji perspektywicznej i jej zastosowanie
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv.warpPerspective(plate_img, M, (w, h))
        roi_characters.append(warped)

    plate_text = []
    # Resize the extracted characters to 64x64 and perform template matching
    # Skalowanie wyciętych znaków do rozmiaru 64x64 i dopasowywanie wzorca
    for roi in roi_characters:
        resized_char = cv.resize(roi, (64, 64), interpolation=cv.INTER_AREA)
        if cv.countNonZero(resized_char) / (resized_char.shape[0] * resized_char.shape[1]) > 0.85:
            continue
        plate_text.append(char_matching(resized_char, char_set))

    # Ensure '0' remains '0' for the first three positions
    # Upewnienie się, że '0' pozostaje '0' dla pierwszych trzech pozycji
    plate_text = [char if char != '0' else '0' for i, char in enumerate(plate_text)]

    return "".join(plate_text)


def enhance_plate_image(plate_img, orig_img, idx):
    # Apply bilateral filtering to reduce noise while preserving edges
    # Zastosowanie filtrowania bilateralnego w celu zredukowania szumów przy zachowaniu krawędzi
    plate_img = cv.bilateralFilter(plate_img, 20, 50, 50)

    # Apply Gaussian blur for further smoothing
    # Zastosowanie rozmycia gaussowskiego do dalszego wygładzania obrazu
    plate_img = cv.blur(plate_img, (7, 7))

    # Apply Otsu thresholding
    # Zastosowanie progowania Otsu
    ret, otsu_thresh = cv.threshold(plate_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Erode the image after thresholding to remove small noise
    # Erozja obrazu po progowaniu w celu usunięcia drobnych szumów
    otsu_thresh = cv.erode(otsu_thresh, np.ones((7, 7), np.uint8))

    # Find contours on the thresholded image
    # Wyszukiwanie konturów na progowanym obrazie
    otsu_contours, otsu_hierarchy = cv.findContours(otsu_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Apply adaptive thresholding
    # Zastosowanie adaptacyjnego progowania
    adaptive_thresh = cv.adaptiveThreshold(plate_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 1)

    # Apply morphological closing and opening to improve the image structure
    # Operacje morfologiczne zamykania i otwierania w celu poprawy struktury obrazu
    adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, (7, 7))
    adaptive_thresh = cv.erode(adaptive_thresh, np.ones((7, 7), np.uint8))
    adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_OPEN, (7, 7))

    # Find contours on the adaptively thresholded image
    # Wyszukiwanie konturów na adaptacyjnie progowanym obrazie
    adaptive_contours, adaptive_hierarchy = cv.findContours(adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Convert threshold images to BGR format
    # Konwersja obrazów progowych do formatu BGR
    otsu_thresh_bgr = cv.cvtColor(otsu_thresh, cv.COLOR_GRAY2BGR)

    # Create empty masks for threshold images
    # Tworzenie pustych masek dla obrazów progowych
    mask_otsu = np.zeros_like(otsu_thresh_bgr)
    mask_adaptive = np.zeros_like(otsu_thresh_bgr)
    contours_otsu_list = []
    contours_adaptive_list = []
    combined_mask = np.zeros_like(otsu_thresh_bgr)

    # Find rectangular contours on the Otsu thresholded image
    # Wyszukiwanie prostokątnych konturów na obrazie progowym Otsu
    for i, cnt in enumerate(otsu_contours):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if (2 <= (w / h) <= 7) or (0.1 <= (w / h) <= 0.5):
                contours_otsu_list.append(cnt)

    # Find rectangular contours on the adaptively thresholded image
    # Wyszukiwanie prostokątnych konturów na adaptacyjnie progowanym obrazie
    for i, cnt in enumerate(adaptive_contours):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if (2 <= (w / h) <= 7) or (0.1 <= (w / h) <= 0.5):
                contours_adaptive_list.append(cnt)
    approx = []
    meanTooLow = False

    # If candidates are found in both thresholded images, create a combined mask
    # Jeśli znaleziono kandydatów w obu obrazach progowych, tworzymy wspólną maskę
    if contours_otsu_list and contours_adaptive_list:
        largest_otsu = sorted(contours_otsu_list, key=cv.contourArea)[0]
        largest_adaptive = sorted(contours_adaptive_list, key=cv.contourArea)[0]
        cv.drawContours(mask_otsu, [largest_otsu], -1, (255, 255, 255), -1)
        cv.drawContours(mask_adaptive, [largest_adaptive], -1, (255, 255, 255), -1)
        combined_mask = cv.bitwise_and(mask_adaptive, mask_otsu)

        if cv.countNonZero(cv.cvtColor(combined_mask, cv.COLOR_BGR2GRAY)) / (
                plate_img.shape[0] * plate_img.shape[1]) > 0.3:
            contours_combined_mask, combined_hierarchy = cv.findContours(
                cv.cvtColor(combined_mask, cv.COLOR_BGR2GRAY),
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )
            contours_combined_mask = sorted(
                contours_combined_mask, key=lambda x: cv.contourArea(x), reverse=True
            )
            approx = cv.convexHull(contours_combined_mask[0])
            combined_mask = combined_mask
        else:
            meanTooLow = True
            mask_adaptive = np.zeros_like(otsu_thresh_bgr)

    # If only the Otsu thresholded image has candidates, create a mask based on this image
    # Jeśli tylko obraz progowy Otsu ma kandydatów, tworzymy maskę na podstawie tego obrazu
    if meanTooLow or not contours_adaptive_list and contours_otsu_list:
        largest_otsu = sorted(contours_otsu_list, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(mask_otsu, [largest_otsu], -1, (255, 255, 255), -1)
        contours_combined_mask, combined_hierarchy = cv.findContours(
            cv.cvtColor(mask_otsu, cv.COLOR_BGR2GRAY),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        contours_combined_mask = sorted(
            contours_combined_mask, key=lambda x: cv.contourArea(x), reverse=True
        )
        approx = cv.convexHull(contours_combined_mask[0])
        combined_mask = mask_otsu

    # If only the adaptively thresholded image has candidates, create a mask based on this image
    # Jeśli tylko adaptacyjnie progowany obraz ma kandydatów, tworzymy maskę na podstawie tego obrazu
    if contours_adaptive_list and not contours_otsu_list:
        largest_adaptive = sorted(contours_adaptive_list, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(mask_adaptive, [largest_adaptive], -1, (255, 255, 255), -1)
        contours_combined_mask, combined_hierarchy = cv.findContours(
            cv.cvtColor(mask_adaptive, cv.COLOR_BGR2GRAY),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        contours_combined_mask = sorted(
            contours_combined_mask, key=lambda x: cv.contourArea(x), reverse=True
        )
        cv.drawContours(mask_adaptive, [contours_combined_mask[0]], -1, (0, 255, 0), 5)
        approx = cv.convexHull(contours_combined_mask[0])
        combined_mask = mask_adaptive

    # If no mask could be created, return a black image
    # Jeśli nie udało się utworzyć maski, zwracamy czarny obraz
    if not combined_mask.any():
        combined_mask = np.zeros_like(otsu_thresh_bgr)
    try:
        # Find the corners of the mask
        # Szukanie narożników maski
        LG = [0, 0]
        LD = [0, plate_img.shape[0]]
        PG = [plate_img.shape[1], 0]
        PD = [plate_img.shape[1], plate_img.shape[0]]
        points = [LG, LD, PD, PG]
        mask_corners = []
        for point in points:
            distances = np.linalg.norm(
                approx.reshape(len(approx), -1) - np.array(point), axis=1
            )
            min_index = np.argmin(distances)
            mask_corners.append(
                [
                    approx.reshape(len(approx), -1)[min_index][0],
                    approx.reshape(len(approx), -1)[min_index][1],
                ]
            )
        # Apply the mask to the Otsu thresholded image
        # Nakładanie maski na obraz progowy Otsu
        result = cv.bitwise_and(otsu_thresh_bgr, combined_mask)
        # List of destination points for the license plate size
        # Lista punktów docelowych o rozmiarze tablicy rejestracyjnej
        pts2 = np.array(
            [[0, 0], [0, 112 * 2], [520 * 2, 112 * 2], [520 * 2, 0]], np.float32
        )
        # Perspective transformation
        # Przekształcenie perspektywiczne
        matrix = cv.getPerspectiveTransform(
            np.float32(np.array(mask_corners).reshape(4, 2)), pts2
        )
        result = cv.warpPerspective(
            cv.cvtColor(result, cv.COLOR_BGR2GRAY), matrix, (520 * 2, 112 * 2)
        )
        # Add a border to the image
        # Dodanie ramki do obrazu
        result = cv.copyMakeBorder(result, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, 255)

        # If the white area exceeds 20%, return the result
        # Jeśli biały obszar przekracza 20%, zwracamy rezultat
        if cv.countNonZero(result) / (result.shape[0] * result.shape[1]) > 0.2:
            return result
        else:
            return otsu_thresh
    except:
        return otsu_thresh


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    # Setting the contrast factor
    # Ustawienie współczynnika kontrastu
    alpha = 1.1
    beta = 0

    # Applying contrast adjustment using arithmetic operations on the image matrix
    # Zastosowanie zmiany kontrastu przy użyciu operacji arytmetycznych na macierzy obrazu
    adjusted = cv.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
    return adjusted


def locate_plate(img: np.ndarray, gray_img):
    # Use Canny filter to detect edges
    # Użycie filtru Canny'ego do wykrywania krawędzi
    edges = cv.Canny(img, 30, 45)

    dilate_size = 5
    struct_element = cv.getStructuringElement(cv.MORPH_RECT, (dilate_size, dilate_size))
    processed_img = cv.dilate(edges, struct_element, iterations=1)

    # Find contours in the entire image
    # Wyszukiwanie konturów na całym obrazie
    contours, hierarchy = cv.findContours(processed_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    potential_plates = []
    plate_candidates = []
    plate_bboxes = []
    for i, contour in enumerate(contours):
        # If the contour approximation has 4 points, it is a potential rectangular plate
        # Jeśli aproksymacja konturu ma 4 punkty, to jest potencjalną prostokątną tablicą
        approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(h / w)
            area = cv.contourArea(contour)
            # Rejecting too small candidates with incorrect proportions
            # Odrzucenie zbyt małych kandydatów o złych proporcjach
            if area >= ((img.shape[0] / 3) * (img.shape[1] / 3)) * 0.3 and 0.15 <= ratio <= 0.5:
                potential_plates.append(contour)
                bbox = cv.boundingRect(approx)
                plate_bboxes.append(bbox)
                # Extracting the candidate plate
                # Wycięcie kandydata na tablicę
                x, y, w, h = bbox
                plate_candidates.append(
                    gray_img[
                        int(y * 0.95): int((y + h) * 1.05),
                        int(x * 0.95): int((x + w) * 1.05),
                    ]
                )
    if not plate_candidates:
        # If no plates are found, repeat the operation using adaptive thresholding
        # Jeśli nie znaleziono tablic, powtórz operację używając adaptacyjnego progowania
        adaptive_thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 23, 1)
        adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, (6, 6))
        contours_adaptive, h = cv.findContours(adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours_adaptive):
            approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
            if cv.contourArea(cnt) > 8000 and len(approx) == 4:
                bbox = cv.boundingRect(approx)
                potential_plates.append(contour)
                plate_bboxes.append(bbox)
                x, y, w, h = bbox
                plate_candidates.append(
                    gray_img[
                        int(y * 0.95): int((y + h) * 1.05),
                        int(x * 0.95): int((x + w) * 1.05),
                    ]
                )
    return plate_candidates, plate_bboxes


def process_image(image: np.ndarray, chars) -> str:
    # Convert the image to grayscale
    # Konwersja obrazu na skalę szarości
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Enhance the contrast of the image
    # Zwiększenie kontrastu obrazu
    contrast_enhanced = cv.convertScaleAbs(gray_image, alpha=1.1)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    # Zastosowanie filtru bilateralnego w celu redukcji szumów przy zachowaniu ostrych krawędzi
    filtered_image = cv.bilateralFilter(contrast_enhanced, 20, 50, 50)

    # Locate potential license plates
    # Zlokalizowanie potencjalnych tablic rejestracyjnych
    plate_candidates, candidate_boxes = locate_plate(filtered_image, gray_image)

    plate_numbers = []
    for i, candidate in enumerate(plate_candidates):
        x, y, w, h = candidate_boxes[i]
        # Select an area 5% larger to avoid cropping the plate
        # Wybór obszaru 5% większego, aby uniknąć przycięcia tablicy
        enhanced_plate = enhance_plate_image(
            candidate,
            image[
                int(y * 0.95): int((y + h) * 1.05),
                int(x * 0.95): int((x + w) * 1.05),
            ],
            i,
        )
        # Extract characters from the plate
        # Ekstrakcja znaków z tablicy
        plate_numbers.append(extract_plate_chars(enhanced_plate, chars, i))

    # Filter out overly long sequences and return the longest valid result
    # Odrzucenie zbyt długich sekwencji i zwrócenie najdłuższego poprawnego wyniku
    plate_numbers = [num for num in plate_numbers if len(num) <= 8]
    plate_numbers = sorted(plate_numbers, key=lambda x: len(x))
    result_number = plate_numbers[0] if len(plate_numbers) and len(plate_numbers[0]) else "P012345"

    return result_number
