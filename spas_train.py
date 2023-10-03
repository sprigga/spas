from gc import callbacks
import numpy as np
import os
import time
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import TensorBoard

# print(tf.__version__)
epochs = 10       #訓練的次數
img_rows = None   #驗證碼影像檔的高
img_cols = None   #驗證碼影像檔的寬
digits_in_img = 4 #驗證碼影像檔中有幾位數
x_list = list()   #存所有驗證碼數字影像檔的array
y_list = list()   #存所有的驗證碼數字影像檔array代表的正確數字
x_train = list()  #存訓練用驗證碼數字影像檔的array
y_train = list()  #存訓練用驗證碼數字影像檔array代表的正確數字
x_test = list()   #存測試用驗證碼數字影像檔的array
y_test = list()   #存測試用驗證碼數字影像檔array代表的正確數字
img_filenames = os.listdir(r'label_captcha_tool-master/captcha')
labeled = open(r'label_captcha_tool-master/label.csv', 'r')
run_id = time.strftime('Run_Time_%Y_%m_%d_%H_%M_%S')
tensorboard_path = os.path.join('captcha_code','logs', run_id)
#abcdefgnmpwxy2345678
dict_captcha = {'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'a':8,'b':9,'c':10
               ,'d':11,'e':12,'f':13,'g':14,'n':15,'m':16,'p':17,'w':18,'x':19,'y':20} 

def sort_key(s):
    # 排序關鍵字匹配
    # 匹配開頭數字序號
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)
def to_onehot(c):
        # split_label = labeled.readline()
        for dict in dict_captcha:
            if dict==c:
                return dict_captcha[dict]
                # print(dict,dict_captcha[dict])

def split_digits_in_img(img_array, x_list, y_list):
    split_label = labeled.readline()
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        # y_list.append(img_filename[i])
        # print(split_label[i])
        onehot = to_onehot(split_label[i])
        y_list.append(onehot)
       
def train_dataset(x_list,y_list):
    y_list = keras.utils.to_categorical(y_list, num_classes=21)
    x_train, x_test, y_train, y_test = train_test_split(x_list, y_list)
    return x_train, x_test, y_train, y_test 

def train_model():
    tensorBoard = TensorBoard(log_dir =tensorboard_path,histogram_freq=1)
    callbacks_list = [tensorBoard]
    if os.path.isfile('spas_cnn_model.h5'):
        model = models.load_model('spas_cnn_model.h5')
        print('Model loaded from file.')
    else:
        model = models.Sequential()
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols // digits_in_img, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(rate=0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(21, activation='softmax'))
        print('New model created.')
 
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(np.array(x_train), np.array(y_train), batch_size=digits_in_img, epochs=epochs, verbose=1, 
                validation_data=(np.array(x_test), np.array(y_test)),callbacks=callbacks_list)
 
    loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
 
    model.save('spas_cnn_model.h5')


if __name__ == '__main__': 

    for img_filename in sorted(img_filenames,key=sort_key):
        # print(sorted(img_filenames,key=sort_key))
        if '.jpg' not in img_filename:
            continue
        img = load_img('label_captcha_tool-master/captcha/{0}'.format(img_filename), color_mode='grayscale')
        img_array = img_to_array(img)
        img_rows, img_cols, _ = img_array.shape
        split_digits_in_img(img_array, x_list, y_list)

# print(len(x_list),x_list)
# print(len(y_list),y_list)
# to_onehot()
        x_train, x_test, y_train, y_test = train_dataset(x_list,y_list)
        train_model()

# img_filename="001314"
# img = load_img('captcha_code/training/001314.png', color_mode='grayscale')
# img_array = img_to_array(img)
# img_rows, img_cols, _ = img_array.shape
# split_digits_in_img(img_array, x_list, y_list)    
labeled.close()
