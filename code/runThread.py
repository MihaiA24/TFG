from tensorflow.keras.models import load_model
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
import glob
import csv

sys.path.append('segmentation-cnn')
sys.path.append('classification-cnn')

from model import jacard_coef, jacard_coef_loss, DiceLoss
from dataset import tf_dataset
from segment_imageV2 import perspective_image, four_point_transform, plate_classificarion, plate_segmentation, \
    segmented_image_warped_skimage_v2

CHARACTER_DIVSION_NAME = 'characterDivision.csv'

model_seg_list = glob.glob('../data/segmentation-cnn/models/*')
model_seg_list.sort()
# Load Model Classification List
model_clas_list = glob.glob('../data/classification-cnn/models/*')
model_clas_list.sort()

model = load_model(model_seg_list[11], custom_objects={'jacard_coef_loss': jacard_coef_loss, 'jacard_coef': jacard_coef,
                                                       'DiceLoss': DiceLoss})
model_classification = load_model(model_clas_list[-2])


def read_image(image_path, mode, shape=None, interpolation=None):
    x = cv2.imread(image_path, mode)
    if shape is not None:
        x = cv2.resize(x, shape, interpolation=interpolation)
    return x


# Only for mask
def bounding_box(img):
    bbox = []
    img_split = img.split('-')[3].split('_')
    for coor in img_split:
        bbox.append((int(coor.split('&')[0]), int(coor.split('&')[1])))
    return np.float32([bbox[2], bbox[3], bbox[1], bbox[0]])


col_names = ['img', 'mask']
df = pd.read_csv('segmentation-cnn/original csv/test_dataset.csv', sep=',', header=None, names=col_names)
images = df['img'].tolist()
masks = df['mask'].tolist()

test_dataset = tf_dataset(images, masks, batch_size=32, buffer_size=1000, shuffle=False)


def copy_image_and_mask(output_path):
    """
    Se crea el directorio de la imagen con formatos:
    output pred/
        1/
            1. [name].jpg
            2. mask.png
        2/
            1. [name].jpg
            2. mask.png
        ...
    Copia la imagen original y la máscara al directorio correspondiente de la imagen.
    """
    for index, item in enumerate(tqdm(images)):
        new_dir = output_path + str((index + 1))
        os.mkdir(new_dir)
        new_dir = new_dir + '/'

        image = cv2.imread(item, cv2.IMREAD_COLOR)
        mask = cv2.imread(masks[index], cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(new_dir + '1. ' + item.split('/')[-1], image)
        cv2.imwrite(new_dir + '2. mask.png', mask)


def run_segmentation_model(output_path):
    """
    Para cada una de las imagenes del directorio con el modelo de segmentacion 
    seleccionado, predice cada una de las imágenes del directorio y guarda su predicción
    con la resolución original: 3.1.pred_original_size.png (720,1160) y 
    la predicción reducida: 3.2.pred.png (288,448)
    """
    index = 0
    for batch in tqdm(test_dataset):
        # Carga Imagenes 

        pred = plate_segmentation(batch, model)
        # Obtencion perspectiva

        for item in pred:
            pred = item * 255
            pred = pred.astype(np.uint8)

            pred_original_size = cv2.resize(pred, (720, 1160), interpolation=cv2.INTER_NEAREST)

            new_dir = output_path + str((index + 1)) + '/'

            cv2.imwrite(new_dir + '3.1. ' + 'pred_original_size.png', pred_original_size * 255)
            cv2.imwrite(new_dir + '3.2. ' + 'pred.png', item)
            index = index + 1


def obtain_perspective_mask_and_seg_letters(output_path):
    """
    A partir del dataset de test, obtiene a partir de la máscara
    la perspectiva geometrica (6.perspective.jpg) a partir del bounding box de la máscara,
    y la imagen correcta de segmentación (7.segmented.jpg), usando los valores fijos de segmentación
    Tambien guarda los valores numéricos de la segmentación(characterDivision.csv)
    """

    for index, item in enumerate(tqdm(images)):
        new_dir = output_path + str((index + 1)) + '/'

        image = cv2.imread(item)
        box = bounding_box(item.split('/')[-1])

        perspective, _ = four_point_transform(image.copy(), box)

        he, wi, dm = perspective.shape
        div = int(wi / 7)

        divs = [i * div + 5 for i in range(1, 7)]
        divs[0] = divs[0] - 5
        divs[1] = divs[1] - 5
        divs[-2] = divs[-2] - 5
        divs[-1] = divs[-1] - 5

        result_seg = perspective.copy()
        for i in divs:
            # divImgList.append(resultSeg)
            result_seg[:, i - 1:i + 1] = 0

        cv2.imwrite(new_dir + '6. ' + 'perspective.jpg', perspective)
        cv2.imwrite(new_dir + '7. ' + 'segmented.jpg', result_seg)
        np.savetxt(new_dir + CHARACTER_DIVSION_NAME, divs, fmt='%i', delimiter=',')


def obtain_perspective_image(output_path):
    """
    A partir de la prediccion de la imagen obtenida con la red de segmentacion
    obtenemos los puntos de la perspectiva (4.points.jpg), las lineas obtenidas (5.draw.jpg) y
    la perspectiva geometrica (6.perspective.jpg).
    """
    for index in tqdm(range(0, len(images))):
        new_dir = output_path + str((index + 1)) + '/'

        image = read_image(images[index], cv2.IMREAD_COLOR, (288, 448), cv2.INTER_CUBIC)
        pred = read_image(new_dir + '3.2. pred.png', cv2.IMREAD_GRAYSCALE)

        points, draw, perspective = perspective_image(image, pred)

        cv2.imwrite(new_dir + '4. ' + 'points.jpg', points)
        cv2.imwrite(new_dir + '5. ' + 'draw.jpg', draw)
        cv2.imwrite(new_dir + '6. ' + 'perspective.jpg', perspective)


def obtain_segmented_letters(output_path):
    """
    A partir de la perspectiva geometrica obtiene la división de cada una de las letras
    y las guarda en el fichero characterDivision.csv y guarda la imagen con las segmentaciones
    (7. segmented. jpg)
    """
    for index in tqdm(range(0, len(images))):
        new_dir = output_path + str((index + 1)) + '/'

        perspective = read_image(new_dir + '6. perspective.jpg', cv2.IMREAD_COLOR)
        result_seg, divs, bw, cleared = segmented_image_warped_skimage_v2(perspective)

        cv2.imwrite(new_dir + '7. ' + 'segmented.jpg', result_seg)
        np.savetxt(new_dir + CHARACTER_DIVSION_NAME, divs, fmt='%i', delimiter=',')


def run_classification_model(output_path):
    """
    A partir de la perspectiva de la imagen y de las coordenadas de segmentación,
    se clasifica cada uno de los caracteres individuales de imangen y se guarda la imagen
    de la perspectiva con el numbre de la prediccion de las letras
    """
    for index in tqdm(range(0, len(images))):
        # Carga Imagenes 
        new_dir = output_path + str((index + 1)) + '/'

        perspective = read_image(new_dir + '6. perspective.jpg', cv2.IMREAD_COLOR)

        with open(new_dir + 'characterDivision.csv', newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
            divs = [int(x[0]) for x in reader]
        result_characters = plate_classificarion(perspective, divs, model_classification)

        cv2.imwrite(new_dir + '10. ' + "".join(result_characters) + '.jpg', perspective)


def execute_pred():
    output_path = 'output pred/'

    lst = os.listdir(output_path)
    lst.sort()

    print('Deleting precv generated images:')
    for item in lst:
        gb = glob.glob(output_path + item + '/10.*')
        for g in gb:
            os.remove(g)

    # Complete Run Full Pipeline
    print('Copy image and mask:')
    copy_image_and_mask(output_path)

    print('Runing seg model:')
    run_segmentation_model(output_path)

    print('Obtain perspective:')
    obtain_perspective_image(output_path)

    print('Obtaing seg letters:')
    obtain_segmented_letters(output_path)

    print('Runing class model:')
    run_classification_model(output_path)
    # F Complete Run Full Pipeline


def exectute_mask():
    output_path = 'output mask/'

    lst = os.listdir(output_path)
    lst.sort()

    print('Deleting precv generated images:')
    for item in lst:
        gb = glob.glob(output_path + item + '/10.*')
        for g in gb:
            os.remove(g)
    # Full Pipeline Mask
    print('Copy image and mask:')
    copy_image_and_mask(output_path)
    print('obtainPerspectiveMaskAndSegLetters: ')
    obtain_perspective_mask_and_seg_letters(output_path)
    print('Run class model: ')
    run_classification_model(output_path)
    # F Full Pipeline Mask


def main():
    execute_pred()
    exectute_mask()


if __name__ == "__main__":
    main()

