import cv2 as cv
import numpy as np
import imutils
from imutils import contours


def char_det(source):
    # Loading character set from files
    chars = {}
    for char_file in source.iterdir():
        char_name = char_file.stem
        chars[char_name] = cv.imread(str(char_file), 0)
    return chars



def char_matching(char, chars):
    highest_score = 0
    best_match = None
    # Comparing extracted characters with the character set and selecting the best match
    for key in chars:
        chars[key] = chars[key].astype(np.uint8)
        char = char.astype(np.uint8)
        match_result = cv.matchTemplate(char, chars[key], cv.TM_CCOEFF)
        _, max_value, _, _ = cv.minMaxLoc(match_result)
        if max_value > highest_score:
            best_match = key
            highest_score = max_value
    return str(best_match)


def extract_plate_chars(plate_img, char_set, idx):
    # Convert the plate image to BGR color space
    plate_bgr = cv.cvtColor(plate_img, cv.COLOR_GRAY2BGR)
    # Find contours on the inverted mask of the plate image
    contours_found = cv.findContours(
        np.bitwise_not(plate_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contour_list = imutils.grab_contours(contours_found)
    try:
        # Sort contours from left to right
        (contour_list, bboxes) = contours.sort_contours(contour_list, method="left-to-right")
    except:
        pass
    potential_chars = []
    for i, contour in enumerate(contour_list):
        x, y, w, h = cv.boundingRect(contour)
        # Select contours with height greater than 33% of the plate height and appropriate aspect ratio
        if h >= plate_img.shape[0] / 3 and 0.15 <= w / h <= 1.3:
            potential_chars.append(contour)
        sorted_chars = sorted(
            potential_chars,
            key=lambda a: cv.boundingRect(a)[3],
            reverse=False,
        )
    cv.drawContours(plate_bgr, potential_chars, -1, (0, 255, 0), 5)
    roi_characters = []
    # Create bounding boxes for each candidate character and store them
    for candidate in potential_chars:
        x, y, w, h = cv.boundingRect(candidate)
        if h < 0.5 * plate_img.shape[0]:
            continue
        pts_source = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts_dest = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        transform_matrix = cv.getPerspectiveTransform(pts_source, pts_dest)
        transformed_char = cv.warpPerspective(plate_img, transform_matrix, (w, h))

        roi_characters.append(transformed_char)
        rect = cv.minAreaRect(candidate)
        box_pts = cv.boxPoints(rect)
        box_pts = np.int0(box_pts)
    plate_text = []
    # Resize extracted characters to 64x64 and perform template matching
    for i, roi in enumerate(roi_characters):
        resized_char = cv.resize(roi, (64, 64), interpolation=cv.INTER_AREA)
        if cv.countNonZero(resized_char) / (resized_char.shape[0] * resized_char.shape[1]) > 0.85:
            continue
        plate_text.append(char_matching(resized_char, char_set))
    # Replace '0' with 'O' for the first three positions in the plate string
    for i, char in enumerate(plate_text):
        char = "O" if (char == "0" and i < 3) else char
    return "".join(plate_text)


def enhance_plate_image(plate_img, orig_img, idx):
    # Double blurring
    candidate_index = idx
    plate_img = cv.bilateralFilter(plate_img, 20, 50, 50)
    plate_img = cv.blur(plate_img, (7, 7))

    # Otsu thresholding
    ret, otsu_thresh = cv.threshold(plate_img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    otsu_thresh = cv.erode(otsu_thresh, np.ones((7, 7), np.uint8))
    otsu_contours, otsu_hierarchy = cv.findContours(
        otsu_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # Adaptive thresholding
    adaptive_thresh = cv.adaptiveThreshold(
        plate_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 1
    )
    adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, (7, 7))
    adaptive_thresh = cv.erode(adaptive_thresh, np.ones((7, 7), np.uint8))
    adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_OPEN, (7, 7))
    adaptive_contours, adaptive_hierarchy = cv.findContours(
        adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # Convert threshold images to BGR
    otsu_thresh_bgr = cv.cvtColor(otsu_thresh, cv.COLOR_GRAY2BGR)
    adaptive_thresh_bgr = cv.cvtColor(adaptive_thresh, cv.COLOR_GRAY2BGR)

    # Create empty masks
    mask_otsu = np.zeros_like(otsu_thresh_bgr)
    mask_adaptive = np.zeros_like(otsu_thresh_bgr)
    otsu_hierarchy = otsu_hierarchy[0]
    contours_otsu_list = []
    contours_adaptive_list = []
    combined_mask = np.zeros_like(otsu_thresh_bgr)

    # Search for rectangular contours in Otsu threshold image
    for i, cnt in enumerate(otsu_contours):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if (2 <= (w / h) <= 7) or (0.1 <= (w / h) <= 0.5):
                contours_otsu_list.append(cnt)

    for i, cnt in enumerate(adaptive_contours):
        # Search for rectangular contours in Adaptive threshold image
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if (2 <= (w / h) <= 7) or (0.1 <= (w / h) <= 0.5):
                contours_adaptive_list.append(cnt)
    approx = []
    meanTooLow = False

    # If candidates found in both threshold images, create a combined mask
    if contours_otsu_list and contours_adaptive_list:
        largest_otsu = sorted(contours_otsu_list, key=cv.contourArea, reverse=True)[0]
        largest_adaptive = sorted(contours_adaptive_list, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(mask_otsu, [largest_otsu], -1, (255, 255, 255), -1)
        cv.drawContours(mask_adaptive, [largest_adaptive], -1, (255, 255, 255), -1)
        combined_mask = cv.bitwise_and(mask_adaptive, mask_otsu)

        if cv.countNonZero(cv.cvtColor(combined_mask, cv.COLOR_BGR2GRAY)) / (plate_img.shape[0] * plate_img.shape[1]) > 0.3:
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

    if not combined_mask.any():
        combined_mask = np.zeros_like(otsu_thresh_bgr)
    try:
        LT = [0, 0]
        LB = [0, plate_img.shape[0]]
        RT = [plate_img.shape[1], 0]
        RB = [plate_img.shape[1], plate_img.shape[0]]
        points = [LT, LB, RB, RT]
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
        result = cv.bitwise_and(otsu_thresh_bgr, combined_mask)
        pts2 = np.array(
            [[0, 0], [0, 114 * 2], [520 * 2, 114 * 2], [520 * 2, 0]], np.float32
        )
        matrix = cv.getPerspectiveTransform(
            np.float32(np.array(mask_corners).reshape(4, 2)), pts2
        )
        result = cv.warpPerspective(
            cv.cvtColor(result, cv.COLOR_BGR2GRAY), matrix, (520 * 2, 114 * 2)
        )
        result = cv.copyMakeBorder(result, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, 255)

        if cv.countNonZero(result) / (result.shape[0] * result.shape[1]) > 0.2:
            return result
        else:
            return otsu_thresh
    except:
        return otsu_thresh


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    contrast_factor = 1.1
    brightness_offset = 1

    enhanced_image = cv.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_offset)
    return enhanced_image


def locate_plate(img: np.ndarray, gray_img):
    # Using Canny filter to detect edges
    edges = cv.Canny(img, 30, 45)

    # Applying dilation, erosion, opening, and closing to improve line quality and white areas
    dilate_size = 5
    struct_element = cv.getStructuringElement(
        cv.MORPH_RECT,
        (dilate_size, dilate_size),
    )
    processed_img = cv.erode(edges, np.ones((7, 7), np.uint8))
    processed_img = cv.morphologyEx(processed_img, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    processed_img = cv.morphologyEx(
        processed_img, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    processed_img = cv.dilate(edges, struct_element, iterations=1)

    # Finding contours in the entire image
    contours, hierarchy = cv.findContours(processed_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    potential_plates = []
    plate_candidates = []
    plate_bboxes = []
    for i, contour in enumerate(contours):
        # If the contour approximation has 4 points, it's a potential rectangular plate
        approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(h / w)
            area = cv.contourArea(contour)
            # Rejecting too small candidates with incorrect proportions
            if (
                    area >= ((img.shape[0] / 3) * (img.shape[1] / 3)) * 0.3
                    and 0.15 <= ratio <= 0.5
            ):
                potential_plates.append(contour)
                bbox = cv.boundingRect(approx)
                x, y, w, h = bbox
                plate_bboxes.append(bbox)
                # Extracting the candidate plate
                plate_candidates.append(
                    gray_img[
                        int(y * 0.94): int((y + h) * 1.06),
                        int(x * 0.94): int((x + w) * 1.06),
                    ]
                )
    if not len(plate_candidates):
        # If no plates found, repeat the process using Adaptive Threshold, steps are similar to Canny
        adaptive_thresh = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 23, 1
        )
        adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, (7, 7))
        adaptive_thresh = cv.morphologyEx(adaptive_thresh, cv.MORPH_OPEN, (7, 7))
        color_thresh = cv.cvtColor(adaptive_thresh, cv.COLOR_GRAY2BGR)
        contours_adaptive, h = cv.findContours(adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        try:
            for i, cnt in enumerate(contours_adaptive):
                approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
                # (x, y), (w, h), angle = cv.minAreaRect(cnt)
                if cv.contourArea(cnt) > 8000 and len(approx) == 4:
                    cv.drawContours(color_thresh, [cnt], -1, (0, 255, 0), 7)
                    potential_plates.append(contour)
                    bbox = cv.boundingRect(approx)
                    x, y, w, h = bbox
                    plate_bboxes.append(bbox)
                    plate_candidates.append(
                        gray_img[
                            int(y * 0.94): int((y + h) * 1.06),
                            int(x * 0.94): int((x + w) * 1.06),
                        ]
                    )
        except:
            pass
    return plate_candidates, plate_bboxes


def process_image(image: np.ndarray, chars) -> str:
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Enhance contrast and apply bilateral filter to reduce noise while preserving edges
    contrast_enhanced = adjust_contrast(gray_image)
    filtered_image = cv.bilateralFilter(contrast_enhanced, 20, 50, 50)

    # Locate potential plate candidates
    plate_candidates, candidate_boxes = locate_plate(filtered_image, gray_image)

    plate_numbers = []
    # For each candidate, find the white area of the plate with the numbers
    for i, candidate in enumerate(plate_candidates):
        x, y, w, h = candidate_boxes[i]
        # Select an area 4% larger to avoid cropping the white plate
        enhanced_plate = enhance_plate_image(
            candidate,
            image[
                int(y * 0.96): int((y + h) * 1.04),
                int(x * 0.96): int((x + w) * 1.04),
            ],
            i,
        )
        # For each white plate, find the number and add it to the list of potential characters
        plate_numbers.append(extract_plate_chars(enhanced_plate, chars, i))

    # Filter out overly long sequences, sort, and return the longest valid result
    plate_numbers = [num for num in plate_numbers if len(num) <= 8]
    plate_numbers = sorted(plate_numbers, key=lambda x: len(x))
    result_number = plate_numbers[0] if len(plate_numbers) and len(plate_numbers[0]) else "P012345"

    return result_number