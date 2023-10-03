import cv2
from cv2 import Mat
import numpy as np
import os
import sys
import re
import time
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

img_rows = None
img_cols = None
digits_in_img = 4
model = None
np.set_printoptions(suppress=True, linewidth=150, precision=3, formatter={'float': '{: 0.3f}'.format})
img_filenames = os.listdir(r'test_captcha')
dict_captcha = {'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'a':8,'b':9,'c':10
            ,'d':11,'e':12,'f':13,'g':14,'n':15,'m':16,'p':17,'w':18,'x':19,'y':20} 

def processImg(img_filename):
    img_cv = cv2.imread(img_filename)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    x,y,w,h = [102,0,88,50]
    dst_resize = img_gray[y:y+h,x:x+w]
    print("procssing captcha")
    time.sleep(0.5)   
    cv2.imwrite(r"test_captcha/processed_captcha.jpg",dst_resize)

def processBatchImg():
     for i in range(1201,1301):
        img_cv = cv2.imread(r"selenium/getKaptchaImg/getKaptchaImg"+str(i)+".jpeg")
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        x,y,w,h = [102,0,88,50]
        dst_resize = img_gray[y:y+h,x:x+w]
        print("procssing captcha_"+str(i))
        time.sleep(0.5)   
        cv2.imwrite(r"test_captcha/"+str(i)+".jpg",dst_resize)

def revers_list(c):
       # split_label = labeled.readline()
       for dict in dict_captcha:
            if dict_captcha[dict] == c:
                   return dict

def sort_key(s):
       # 排序關鍵字匹配
       # 匹配開頭數字序號
       if s:
            try:
                c = re.findall('^\d+', s)[0]
            except:
                c = -1
            return int(c)


def split_digits_in_img(img_array):
        x_list = list()
        for i in range(digits_in_img):
            step = img_cols // digits_in_img
            x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
            # print(x_list)
        return x_list

def cnn_model_predict(predict_img):
    global img_rows
    global img_cols
    # global result_class
    if os.path.isfile('spas_cnn_model.h5'):
        model = models.load_model('spas_cnn_model.h5')
    else:
        print('No trained model found.')
        exit(-1)
    
    img = load_img(predict_img,color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    # print(img_array.shape, img_rows, img_cols)
    x_list = split_digits_in_img(img_array)

    varification_code = list()
    for i in range(digits_in_img):
        confidences = model.predict(np.array([x_list[i]]), verbose=0)
        result_class = model.predict_classes(np.array([x_list[i]]), verbose=0)
        result_value = revers_list(result_class[0])
        varification_code.append(result_value)
        print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(
                i + 1, np.squeeze(confidences), np.squeeze(result_class)))
    print('Predicted varification code:', varification_code)
    print('\r\n')
    return varification_code


def cnn_model_batch_predict():
    global img_rows
    global img_cols
    global result_class
    if os.path.isfile('spas_cnn_model.h5'):
        model = models.load_model('spas_cnn_model.h5')
    else:
        print('No trained model found.')
        exit(-1)

    for img_filename in sorted(img_filenames, key=sort_key):
        print(img_filename)
        if '.jpg' not in img_filename:
            continue
            # img_filename = input('Varification code img filename: ')
        img = load_img("test_captcha/" + img_filename,
                        color_mode='grayscale')
        img_array = img_to_array(img)
        img_rows, img_cols, _ = img_array.shape
        # print(img_array.shape, img_rows, img_cols)
        x_list = split_digits_in_img(img_array)

        varification_code = list()
        for i in range(digits_in_img):
            confidences = model.predict(np.array([x_list[i]]), verbose=0)
            result_class = model.predict_classes(np.array([x_list[i]]), verbose=0)
            result_value = revers_list(result_class[0])
            varification_code.append(result_value)
            print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(
                    i + 1, np.squeeze(confidences), np.squeeze(result_class)))
        print('Predicted varification code:', varification_code)
        print('\r\n')

def captcha_code(img_filename,predict_img):
    processImg(img_filename)
    time.sleep(1)
    captcha_list = cnn_model_predict(predict_img)
    # print("captcha list: ",captcha_list)
    captcha_str=""
    for ca in captcha_list:
        captcha_str = captcha_str + ca
    print("captcha code: ", captcha_str)
    return captcha_str