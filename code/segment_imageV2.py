import cv2
import numpy as np
import imutils

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square


def get_bounding_box_from_mask(y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(y)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


def order_points_old(pts):
    # WARNING: UNUSED
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform_old(image, pts):
    # WARNING: UNUSED
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    maxHeight = 70
    maxWidth = 180
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [0, maxHeight]],
        [maxWidth, maxHeight], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def segmented_image(image, prediction):
    """
    Recibe una imagen y una máscara, procesa, obtiene la perspectiva
    aplica Canny y Hough 

    02/07/2022: En esta V2 vamos a intetar aplicar los filtros antes de la obtención de la perspectiva

    WARNING: UNUSED
    """
    image = image.astype(np.uint8)
    prediction = prediction.astype(np.uint8)

    img = cv2.resize(image, (720, 1160), interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize(prediction, (720, 1160), interpolation=cv2.INTER_NEAREST)

    img = img.astype(np.uint8)
    pred = pred.astype(np.uint8)

    cnts = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    box = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            box = np.squeeze(approx, axis=1)
            break

    height = 70
    width = 180

    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    if box[0][0] < box[2][0]:
        pt12 = np.float32([box[0], box[3], box[1], box[2]])
    else:
        pt12 = np.float32([box[1], box[0], box[2], box[3]])

    f = cv2.getPerspectiveTransform(pt12, pt2)
    f2 = cv2.warpPerspective(img, f, (width, height))

    gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # Create structure element for extracting vertical lines through morphology operations
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    # Apply morphology operations
    vertical = cv2.erode(bw, vertical_structure)
    vertical = cv2.dilate(vertical, horizontal_structure)

    horizontal = cv2.erode(bw, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    ver_hor = vertical | horizontal
    canny = cv2.Canny(ver_hor, 40, 200)

    lines = cv2.HoughLinesP(canny, 0.001, np.pi / 180, 3, minLineLength=40, maxLineGap=180)

    hough = canny.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough, (x1, y1), (x2, y2), (255), 1)

    hough = cv2.erode(hough, vertical_structure)
    hough = cv2.dilate(hough, vertical_structure)

    kernel = np.ones((int(height * 0.1)), dtype=np.uint8) * 255

    opening = cv2.morphologyEx(hough, cv2.MORPH_OPEN, kernel)

    boxop = get_bounding_box_from_mask(opening).astype(int)

    result = f2[boxop[0]:boxop[2], boxop[1]:boxop[3]]

    if not result.any():
        result = f2.copy()
    result_seg = result.copy()

    he, wi, dm = result_seg.shape
    div = int(wi / 7)

    divs = [i * div + 5 for i in range(1, 7)]
    pad = 2
    divs[0] = divs[0] - pad
    divs[1] = divs[1] - pad
    divs[-2] = divs[-2] - pad
    divs[-1] = divs[-1] - pad

    for i in divs:
        result_seg[:, i - 1:i + 1] = 0

    return [result_seg, divs, f2, canny, opening]


def segmented_image_warped_old(image):
    # WARNING: UNUSED

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    # Apply morphology operations
    vertical = cv2.erode(bw, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    horizontal = cv2.erode(bw, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    verHor = vertical | horizontal
    canny = cv2.Canny(verHor, 40, 200)
    # lines = cv2.HoughLinesP(canny,1,np.pi/180,10, maxLineGap=150)

    lines = cv2.HoughLinesP(canny, 0.001, np.pi / 180, 3, minLineLength=40, maxLineGap=180)

    # if lines is None:
    #     break
    height = 70
    width = 180

    hough = np.zeros((height, width), np.uint8)
    hough = canny.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough, (x1, y1), (x2, y2), (255), 1)

    hough = cv2.erode(hough, verticalStructure)
    hough = cv2.dilate(hough, verticalStructure)

    kernel = np.ones((int(height * 0.1)), dtype=np.uint8) * 255

    opening = cv2.morphologyEx(hough, cv2.MORPH_OPEN, kernel)

    boxop = get_bounding_box_from_mask(opening).astype(int)

    result = image[boxop[0]:boxop[2], boxop[1]:boxop[3]]

    if not result.any():
        result = image.copy()
    resultSeg = result.copy()

    he, wi, dm = resultSeg.shape
    div = int(wi / 7)

    divs = [i * div + 5 for i in range(1, 7)]
    pad = 2
    divs[0] = divs[0] - pad
    divs[1] = divs[1] - pad
    divs[-2] = divs[-2] - pad
    divs[-1] = divs[-1] - pad

    for i in divs:
        resultSeg[:, i - 1:i + 1] = 0

    return [resultSeg, divs, canny, opening]


def segmented_image_warped_skimage(image_rgb):
    # WARNING: UNUSED

    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # apply threshold
    thresh = threshold_otsu(image)
    thresh = thresh
    bw = closing(image > thresh, square(1))

    bw[:10, :] = 0
    bw[60:, :] = 0
    # bw[:,:10] = 0
    bw[:, 170:] = 0

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    [h, w] = label_image.shape
    h2 = int(h / 2)
    numbers = label_image[h2 - 5:h2 + 5, :].flatten()
    numbers = set(numbers)
    uniq = np.unique(label_image)
    uniq = [x if x not in numbers else 0 for x in uniq]
    for x in uniq:
        label_image[label_image == x] = 0

    rec = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 50:
            minr, minc, maxr, maxc = region.bbox

            rec.append(minc)
            rec.append(maxc)
    rec.sort()

    res = []
    for index in range(1, len(rec) - 1, 2):
        sep = int((rec[index + 1] - rec[index]) * 0.5)
        sep = rec[index] + sep
        res.append(sep)

    result_seg = image_rgb.copy()

    divs = res

    for i in divs:
        result_seg[:, i - 1:i + 1] = 0

    bw = bw * 255
    bw = bw.astype(np.uint8)

    cleared = cleared * 255
    cleared = cleared.astype(np.uint8)
    return [result_seg, divs, bw, cleared]


def segmented_image_warped_skimage_v2(image_rgb):
    """
    A partir de la imagen de la perspectiva se obtiene la segmentacion
    de cada uno de los carácteres
    :param image_rgb: Imagen RGB de la perspectiva h = 70, w = 180
    :return: [result_seg, divs, bw, cleared] donde:
    result_seg -> imagen perspectiva segmentada
    divs -> Array con los números de las divisiones (cols)
    bw -> Imagen binarizada de la perspectiva con thresh
    cleared -> imagen bw con el ruido filtrado
    """
    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # apply threshold
    thresh = threshold_otsu(image)
    thresh = thresh
    bw = closing(image > thresh, square(1))

    bw[:10, :] = 0
    bw[65:, :] = 0
    bw[:, :5] = 0
    bw[:, 170:] = 0

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    [h, w] = label_image.shape
    h2 = int(h / 2)
    numbers = label_image[h2 - 5:h2 + 5, :].flatten()
    numbers = set(numbers)
    uniq = np.unique(label_image)
    uniq = [x if x not in numbers else 0 for x in uniq]
    for x in uniq:
        label_image[label_image == x] = 0

    rec = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 50:
            minr, minc, maxr, maxc = region.bbox

            rec.append(minc)
            rec.append(maxc)
    rec.sort()

    res = []
    for index in range(1, len(rec) - 1, 2):
        sep = int((rec[index + 1] - rec[index]) * 0.5)
        sep = rec[index] + sep
        res.append(sep)

    result_seg = image_rgb.copy()
    if len(res) > 6:
        res = res[-6:]
    if len(res) == 5:
        res.insert(0, 0)
    if len(res) != 6:  # Si no encuentro segmentacion correcta, recurro a la manual
        div = int(w / 7)
        res = [i * div + 5 for i in range(1, 7)]
        pad = 5
        res[0] = res[0] - pad
        res[1] = res[1] - pad
        res[-2] = res[-2] - pad
        res[-1] = res[-1] - pad
    divs = res

    for i in divs:
        result_seg[:, i - 1:i + 1] = 0

    bw = bw * 255
    bw = bw.astype(np.uint8)

    cleared = cleared * 255
    cleared = cleared.astype(np.uint8)
    return [result_seg, divs, bw, cleared]


def plate_classificarion(perspective, divs, class_model):
    """

    :param perspective: Imagen RGB de la perspectiva h = 70, w = 180
    :param divs: Array de int con las divisiones de las columnas de cada carácter
    de la matrícula
    :param class_model: Modelo tensorflow clasificación VGG16 entrenado
    :return: Array de char con
    """
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    perspective = cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB)
    seg_img = []
    for ind in range(0, len(divs)):
        if ind == len(divs) - 1:
            seg_img.append(perspective[:, divs[ind]:, :])
        else:
            seg_img.append(perspective[:, divs[ind]:divs[ind + 1], :])

    seg_letters = []
    if len(seg_img) == 0:
        return []

    for letter in seg_img:
        img = cv2.resize(letter, (32, 32), interpolation=cv2.INTER_NEAREST)
        seg_letters.append(img)
    letters = np.stack(seg_letters, axis=0)
    characters = class_model.predict(letters)

    result_characters = []
    for i in characters:
        result_characters.append(classes[np.argmax(i)])

    return result_characters


def plate_segmentation(batch, seg_model):
    """
    Predice en el modelo de Segmentacion UNET del batch pasado por parámetro
    :param batch: Batch de tipo tf.dataset
    :param seg_model: Modelo Tensorflow UNET
    :return: Batch imagenes predecidas
    """
    pred = seg_model.predict(batch[0])

    for item in pred:
        item[item >= 0.5] = 255
        item[item < 0.5] = 0

    return pred


def corners_by_countours(pred):
    """
    A partir de una imagen binaria [0 255] np.uint8 obtiene las coordenadas de los 4 vertices
    """
    screen_cnt = []
    cnts = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.07 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screen_cnt = np.squeeze(approx, axis=1)
            break
    return screen_cnt


def order_points(pts):
    """
    Ordena los puntos de forma: tl, tr, bl, br
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts, rct=None):
    """
    A patir de una imagen y las coordenadas de sus vertices, obtiene la perspectiva
    geómetrica de esta y la devuelve
        image -> Imagen BGR o RGB con valores [0 255] np.uint8
        pts -> Coordenadas de los 4 vertices, formato: [(F,C),(F,C),(F,C),(F,C)]
    return:
        warped -> Perspectiva geométrica obtenida
        rect -> array de coordenadas ordennado []
    """
    rect = order_points(pts)
    # (tl, tr, bl, br) = rect
    if rct is not None:
        rect = rct

    max_height = 70
    max_width = 180

    dst = np.array([
        [0, 0],
        [max_width, 0],
        [0, max_height],
        [max_width, max_height]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    # return the warped image
    return warped, rect


def perspective_image(image, prediction):
    """
    A partir de una imagen y una máscara binaria obtiene la perspectiva del
    ROI de esta.
    Recibe como argumentos
        image -> Imagen BGR  o RGB con valores [0, 255] np.uint8
        prediction -> Máscara binaria del ROI de image con valores [0 255] np.uint8
    Devuelve: points, drawLines, perspective
    points -> Imagen binaria con los 4 puntos de los vértices
    drawLines -> ROI de image con los contornos obtenidos
    perspective -> Imagen aplicando la perspectiva geométrica del ROI de image
    """
    pred = cv2.GaussianBlur(prediction, (3, 3), 0)
    kernel = np.ones((1, 1), 'uint8')
    pred = cv2.dilate(pred, kernel, iterations=2)
    # Evitar que no detecte contornos en los bordes
    pred = cv2.copyMakeBorder(pred, 0, 1, 1, 0, cv2.BORDER_CONSTANT, None, value=0)

    corners = []
    block_size = range(10, 35)
    for block in block_size:
        corners = cv2.goodFeaturesToTrack(pred, maxCorners=4, qualityLevel=0.05, minDistance=10, blockSize=block,
                                          useHarrisDetector=True)
        corners = np.int0(corners)
        corners = np.squeeze(corners, axis=1)
        if len(corners) == 4:
            break

    # Second try with all border Mask
    if len(corners) != 4:
        # pred = predRes[0].astype(np.uint8)
        pred = cv2.GaussianBlur(prediction, (3, 3), 0)
        kernel = np.ones((1, 1), 'uint8')
        pred = cv2.dilate(pred, kernel, iterations=2)
        # Evitar que no detecte contornos en los bordes
        pred = cv2.copyMakeBorder(pred, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)
        for block in block_size:
            corners = cv2.goodFeaturesToTrack(pred, maxCorners=4, qualityLevel=0.05, minDistance=10, blockSize=block,
                                              useHarrisDetector=True)
            corners = np.int0(corners)
            corners = np.squeeze(corners, axis=1)
            if len(corners) == 4:
                break

    if len(corners) != 4:
        corners = corners_by_countours(pred)

    perspective, _ = four_point_transform(image.copy(), corners)

    draw = image.copy()
    cv2.drawContours(draw, [corners], -1, (0, 0, 1), 2)

    points = pred.copy()
    for i in range(0, 4):
        cv2.circle(points, (corners[i][0], corners[i][1]), 5, (i + 1) * 50, -1)

    return points, draw, perspective
