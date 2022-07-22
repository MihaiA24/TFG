from tqdm import tqdm
import pandas as pd
import glob
from natsort import natsorted 
import collections  
import sys

# Comparo los resultados

test_length = 25000
# Leo en el array las imagenes originales y saco los indices correctos

col_names = ['img','mask']
df = pd.read_csv ('segmentation-cnn/original csv/test_dataset.csv',sep=',',header=None,names=col_names)
images = df['img'].tolist()

args = sys.argv

path = args[1]

origImg = [None] * test_length 

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for index,item in enumerate(tqdm(images)):
    img_name = item.split('/')[-1]
    splt = img_name.split('-')[-3].split('_')[1:]
    arr = [ads[int(x)] for x in splt]
    origImg[index] = arr


resultImg = dict()

lst = glob.glob(path + '*/10.*')
lst = natsorted(lst)
print('Total processed: ',len(lst))
for index, item in enumerate(tqdm(lst)):
    resultImg[int(item.split('/')[-2])] = list(item.split('10. ')[-1].split('.jpg')[0])


iguales = [None] * test_length 
maxIndex = 0
for index,item in enumerate(tqdm(origImg)):
    if (index + 1) in resultImg:
        if resultImg[(index + 1)] == item:
            iguales[index] = True
        else:
            iguales[index] = False
    maxIndex = index + 1


counter=collections.Counter(iguales)

print(counter)
