from tqdm import tqdm
import pandas as pd
import glob
from natsort import natsorted 
import collections  
import sys
import cv2

"""
Comparo los resultados de las imagenes obtenidas. Obtengo los nombres de las imagenes obtenidas con el script runThread.py
de cada una de las carpetas que comienzan con 10.* . En este caso vamos a comprobar cada uno de los caracteres de las matriculas
"""

test_length = 25000 * 26
# Leo en el array las imagenes originales y saco los indices correctos

col_names = ['img', 'mask']
df = pd.read_csv('segmentation-cnn/original csv/test_dataset.csv',sep=',', header=None, names=col_names)
images = df['img'].tolist()

args = sys.argv

path = args[1]

origImg = [None] * test_length 

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for index, item in enumerate(tqdm(images)):
    img_name = item.split('/')[-1]
    splt = img_name.split('-')[-3].split('_')[1:]
    arr = [ads[int(x)] for x in splt]
    origImg.extend(arr)


resultImg = []

lst = glob.glob(path + '*/10.*')
lst = natsorted(lst)

print('Total processed: ', len(lst))
for index, item in enumerate(tqdm(lst)):
    resultImg.extend(item.split('10. ')[-1].split('.jpg')[0])


print('Original size: ', len(origImg))
print('Prediction size: ', len(resultImg))

iguales = [None] * test_length

for index, item in enumerate(tqdm(origImg)):
    if origImg[index] == resultImg[index]:
        iguales[index] = True
    else:
        iguales[index] = False


counter = collections.Counter(iguales)

print(counter)
