import csv
import os
import random
from tqdm import tqdm


print('Total images length: ',len(os.listdir('../ccpd_base')))
print('Total masks length:  ',len(os.listdir('../ccpd_base')))

def write_csv(csv_name,data,image_path,mask_path):
    
    if('.csv' not in csv_name):
        csv_name = csv_name + '.csv'
        
    with open(csv_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in tqdm(data):
            csv_writer.writerow([image_path + '/' + item,mask_path + '/' + item.replace('jpg','png')])


# Divido el dataset en 3 partes
# 25.000 imagenes test
# 10.000 imagenes validation
# 275.482 (Resto) imagenes train

test_length = 25000
val_length = 10000

images_dir = '../ccpd_base'
mask_dir = '../ccpd_mask'

image_full_path = os.path.abspath(images_dir)
mask_full_path = os.path.abspath(mask_dir)

images_list = os.listdir(images_dir)

random.shuffle(images_list)

test_data = images_list[:test_length]
val_data = images_list[test_length:(test_length + val_length)]
train_data = images_list[(test_length + val_length):]
print('Test length: ',len(test_data))
print('Val length: ',len(val_data))
print('Train length: ',len(train_data))

print('Creating test_dataset.csv: ')
write_csv('test_dataset.csv',test_data,image_full_path,mask_full_path)
print('Creating val_dataset.csv: ')
write_csv('val_dataset.csv',val_data,image_full_path,mask_full_path)
print('Creating train_dataset.csv: ')
write_csv('train_dataset.csv',train_data,image_full_path,mask_full_path)

print('Done!')


