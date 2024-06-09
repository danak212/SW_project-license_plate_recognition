import cv2 as cv
import numpy as np
import imutils
from imutils import contours


def char_det(source):
    # Funkcja wczytująca zestaw znaków z plików
    chars = {}

    # Listowanie wszystkich plików w katalogu źródłowym
    char_files = list(source.iterdir())

    # Iterowanie po plikach i wczytywanie obrazów
    for char_file in char_files:
        # Pobranie nazwy pliku bez rozszerzenia jako nazwy znaku
        char_name = char_file.stem

        # Wczytanie obrazu znaku jako obraz w skali szarości
        char_image = cv.imread(str(char_file), cv.IMREAD_GRAYSCALE)

        # Przechowywanie obrazu w słowniku z nazwą znaku jako klucz
        chars[char_name] = char_image

    return chars



def char_matching(char, chars):
    # Funkcja dopasowująca wycięty znak do zestawu znaków
    highest_score = 0
    best_match = None
    for key in chars:
        # Przekształcenie obrazów na typ uint8
        chars[key] = chars[key].astype(np.uint8)
        char = char.astype(np.uint8)
        # Dopasowywanie wzorca
        match_result = cv.matchTemplate(char, chars[key], cv.TM_CCOEFF)
        _, max_value, _, _ = cv.minMaxLoc(match_result)
        if max_value > highest_score:
            # Aktualizacja najlepszego dopasowania
            best_match = key
            highest_score = max_value
    return str(best_match)


def extract_plate_chars(plate_img, char_set, idx):
    # Konwersja obrazu tablicy rejestracyjnej do BGR
    plate_bgr = cv.cvtColor(plate_img, cv.COLOR_GRAY2BGR)
    # Szukanie konturów zewnętrznych znaków na odwróconej masce
    contours_found = cv.findContours(
        np.bitwise_not(plate_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contour_list = imutils.grab_contours(contours_found)
    try:
        # Sortowanie konturów od lewej do prawej
        (contour_list, bboxes) = contours.sort_contours(contour_list, method="left-to-right")
    except:
        pass
    potential_chars = []
    for i, contour in enumerate(contour_list):
        x, y, w, h = cv.boundingRect(contour)
        # Wybieranie konturów o odpowiednich proporcjach
        if h >= plate_img.shape[0] / 3 and 0.13 <= w / h <= 1.22:
            potential_chars.append(contour)
        sorted_chars = sorted(
            potential_chars,
            key=lambda a: cv.boundingRect(a)[2],
        )
    cv.drawContours(plate_bgr, potential_chars, -1, (0, 255, 0), 2)
    roi_characters = []
    # Tworzenie bounding boxa dla każdego kandydata i zapis do listy
    for candidate in potential_chars:
        x, y, w, h = cv.boundingRect(candidate)
        if h < 0.5 * plate_img.shape[0]:
            continue
        pts_source = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts_dest = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        transform_matrix = cv.getPerspectiveTransform(pts_source, pts_dest)
        transformed_char = cv.warpPerspective(plate_img, transform_matrix, (w, h))

        roi_characters.append(transformed_char)
    plate_text = []
    # Skalowanie wyciętych znaków do rozmiaru 64x64 i dopasowywanie wzorca
    for i, roi in enumerate(roi_characters):
        resized_char = cv.resize(roi, (64, 64), interpolation=cv.INTER_AREA)
        if cv.countNonZero(resized_char) / (resized_char.shape[0] * resized_char.shape[1]) > 0.85:
            continue
        plate_text.append(char_matching(resized_char, char_set))
    # Zmiana '0' na 'O' dla pierwszych trzech pozycji w stringu tablicy
    for i, char in enumerate(plate_text):
        char = "O" if (char == "0" and i < 3) else char
    return "".join(plate_text)


def enhance_plate_image(plate_img, orig_img, idx):
    # Zastosowanie filtrowania bilateralnego w celu zredukowania szumów przy zachowaniu krawędzi
    plate_img = cv.bilateralFilter(plate_img, 20, 50, 50)
    # Zastosowanie rozmycia gaussowskiego do dalszego wygładzania obrazu
    plate_img = cv.blur(plate_img, (7, 7))

    # Zastosowanie progowania Otsu
    ret, otsu_thresh = cv.threshold(plate_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # Erozja obrazu po progowaniu w celu usunięcia drobnych szumów
    otsu_thresh = cv.erode(otsu_thresh, np.ones((7, 7), np.uint8))
    # Wyszukiwanie konturów na progowanym obrazie
    otsu_contours, otsu_hierarchy = cv.findContours(otsu_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Zastosowanie adaptacyjnego progowania
    adaptive_thresh = cv.adaptiveThreshold(plate_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 1)
    # Operacje morfologiczne zamykania i otwierania w celu poprawy struktury obrazu
    adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, (7, 7))
    adaptive_thresh = cv.erode(adaptive_thresh, np.ones((7, 7), np.uint8))
    adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_OPEN, (7, 7))
    # Wyszukiwanie konturów na adaptacyjnie progowanym obrazie
    adaptive_contours, adaptive_hierarchy = cv.findContours(adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Konwersja obrazów progowych do formatu BGR
    otsu_thresh_bgr = cv.cvtColor(otsu_thresh, cv.COLOR_GRAY2BGR)

    # Tworzenie pustych masek dla obrazów progowych
    mask_otsu = np.zeros_like(otsu_thresh_bgr)
    mask_adaptive = np.zeros_like(otsu_thresh_bgr)
    contours_otsu_list = []
    contours_adaptive_list = []
    combined_mask = np.zeros_like(otsu_thresh_bgr)

    # Wyszukiwanie prostokątnych konturów na obrazie progowym Otsu
    for i, cnt in enumerate(otsu_contours):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if (2 <= (w / h) <= 7) or (0.1 <= (w / h) <= 0.5):
                contours_otsu_list.append(cnt)

    # Wyszukiwanie prostokątnych konturów na adaptacyjnie progowanym obrazie
    for i, cnt in enumerate(adaptive_contours):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if (2 <= (w / h) <= 7) or (0.1 <= (w / h) <= 0.5):
                contours_adaptive_list.append(cnt)
    approx = []
    meanTooLow = False

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
        cv.drawContours(mask_adaptive, [contours_combined_mask[0]], -1, (0.255, 0), 5)
        approx = cv.convexHull(contours_combined_mask[0])
        combined_mask = mask_adaptive

    # Jeśli nie udało się utworzyć maski, zwracamy czarny obraz
    if not combined_mask.any():
        combined_mask = np.zeros_like(otsu_thresh_bgr)
    try:
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
        # Nakładanie maski na obraz progowy Otsu
        result = cv.bitwise_and(otsu_thresh_bgr, combined_mask)
        # Lista punktów docelowych o rozmiarze tablicy rejestracyjnej
        pts2 = np.array(
            [[0, 0], [0, 112 * 2], [520 * 2, 112 * 2], [520 * 2, 0]], np.float32
        )
        # Przekształcenie perspektywiczne
        matrix = cv.getPerspectiveTransform(
            np.float32(np.array(mask_corners).reshape(4, 2)), pts2
        )
        result = cv.warpPerspective(
            cv.cvtColor(result, cv.COLOR_BGR2GRAY), matrix, (520 * 2, 112 * 2)
        )
        # Dodanie ramki do obrazu
        result = cv.copyMakeBorder(result, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, 255)

        # Jeśli biały obszar przekracza 20%, zwracamy rezultat
        if cv.countNonZero(result) / (result.shape[0] * result.shape[1]) > 0.2:
            return result
        else:
            return otsu_thresh
    except:
        return otsu_thresh


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    # Ustawienie współczynnika kontrastu
    contrast_factor = 1.1
    # Zastosowanie konwersji skali dla obrazu z określonym współczynnikiem kontrastu
    enhanced_image = cv.convertScaleAbs(image, alpha=contrast_factor)
    return enhanced_image


def locate_plate(img: np.ndarray, gray_img):
    # Użycie filtru Canny'ego do wykrywania krawędzi
    edges = cv.Canny(img, 30, 45)

    # Zastosowanie dylacji, erozji, operacji otwarcia i zamknięcia w celu poprawy jakości linii i białych obszarów
    dilate_size = 5
    struct_element = cv.getStructuringElement(
        cv.MORPH_RECT,
        (dilate_size, dilate_size),
    )
    processed_img = cv.dilate(edges, struct_element, iterations=1)

    # Wyszukiwanie konturów na całym obrazie
    contours, hierarchy = cv.findContours(processed_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    potential_plates = []
    plate_candidates = []
    plate_bboxes = []
    for i, contour in enumerate(contours):
        # Jeśli aproksymacja konturu ma 4 punkty, to jest potencjalną prostokątną tablicą
        approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(h / w)
            area = cv.contourArea(contour)
            # Odrzucenie zbyt małych kandydatów o złych proporcjach
            if (
                    area >= ((img.shape[0] / 3) * (img.shape[1] / 3)) * 0.3
                    and 0.15 <= ratio <= 0.5
            ):
                potential_plates.append(contour)
                bbox = cv.boundingRect(approx)
                x, y, w, h = bbox
                plate_bboxes.append(bbox)
                # Wycięcie kandydata na tablicę
                plate_candidates.append(
                    gray_img[
                        int(y * 0.95): int((y + h) * 1.05),
                        int(x * 0.95): int((x + w) * 1.05),
                    ]
                )
    if not len(plate_candidates):
        # Jeśli nie znaleziono tablic, powtórz operację używając adaptacyjnego progowania
        adaptive_thresh = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 23, 1
        )
        adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, (6, 6))
        color_thresh = cv.cvtColor(adaptive_thresh, cv.COLOR_GRAY2BGR)
        contours_adaptive, h = cv.findContours(adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        try:
            for i, cnt in enumerate(contours_adaptive):
                approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
                if cv.contourArea(cnt) > 8000 and len(approx) == 4:
                    cv.drawContours(color_thresh, [cnt], -1, (0, 255, 0), 7)
                    potential_plates.append(contour)
                    bbox = cv.boundingRect(approx)
                    x, y, w, h = bbox
                    plate_bboxes.append(bbox)
                    plate_candidates.append(
                        gray_img[
                            int(y * 0.95): int((y + h) * 1.05),
                            int(x * 0.95): int((x + w) * 1.05),
                        ]
                    )
        except:
            pass
    return plate_candidates, plate_bboxes


def process_image(image: np.ndarray, chars) -> str:
    # Konwersja obrazu na skalę szarości
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Zwiększenie kontrastu i zastosowanie filtru bilateralnego w celu redukcji szumów przy zachowaniu krawędzi
    contrast_enhanced = adjust_contrast(gray_image)
    filtered_image = cv.bilateralFilter(contrast_enhanced, 20, 50, 50)

    # Zlokalizowanie potencjalnych kandydatów na tablice
    plate_candidates, candidate_boxes = locate_plate(filtered_image, gray_image)

    plate_numbers = []
    # Dla każdego kandydata, znajdowanie białego obszaru tablicy z numerami
    for i, candidate in enumerate(plate_candidates):
        x, y, w, h = candidate_boxes[i]
        # Wybieranie obszaru 5% większego, aby uniknąć przycięcia białej tablicy
        enhanced_plate = enhance_plate_image(
            candidate,
            image[
                int(y * 0.95): int((y + h) * 1.05),
                int(x * 0.95): int((x + w) * 1.05),
            ],
            i,
        )
        # Dla każdej białej tablicy, znajdowanie numeru i dodawanie do listy potencjalnych znaków
        plate_numbers.append(extract_plate_chars(enhanced_plate, chars, i))

    # Odrzucenie zbyt długich sekwencji, sortowanie i zwracanie najdłuższego poprawnego wyniku
    plate_numbers = [num for num in plate_numbers if len(num) <= 8]
    plate_numbers = sorted(plate_numbers, key=lambda x: len(x))
    result_number = plate_numbers[0] if len(plate_numbers) and len(plate_numbers[0]) else "P012345"

    return result_number
