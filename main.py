import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets.base import Bunch
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.svm import SVC

import brightnessNormalize as bn
import calorieCalculator as cc
import cnn
import coinManager as cm
import datasetReader as reader
import imageSegmentation as imSeg


def append_k_means(X_train, X_test, y_train):
    n_clusters = len(np.unique(y_train))
    clf = KMeans(n_clusters = n_clusters, random_state=42)
    clf.fit(X_train)
    y_labels_train = clf.labels_
    y_labels_test = clf.predict(X_test)
    #print("kmeans result = ", y_labels_test)
    #X_train = y_labels_train[:, np.newaxis]
    #X_test = y_labels_test[:, np.newaxis]
    X_train['km_clust'] = y_labels_train
    X_test['km_clust'] = y_labels_test
    return X_train, X_test

def apply_model(X_train, X_test, y_train, y_test, model=LogisticRegression(random_state=42)):
    #X_train, X_test, y_train, y_test = train_test_split(data_frame, target, test_size=0.3, random_state=42)
    X_train, X_test = append_k_means(X_train, X_test, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred))
    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
    return label_encoder.inverse_transform(y_pred)

def build_filters():
    filters = []
    ksize = 21
    for theta in np.arange(0, np.pi, np.pi / 20):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def image_resize(image, height, width, inter = cv2.INTER_AREA):
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def getMaskedCropedImage(image_cropped):
    filters = build_filters()
    image_cropped_textured = process(image_cropped,filters)

    mask = np.zeros(image_cropped_textured.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,image_cropped_textured.shape[1]-1,image_cropped_textured.shape[0]-1)
    cv2.grabCut(image_cropped_textured,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    final_image = image_cropped*mask2[:,:,np.newaxis]
    final_image_gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    return final_image_gray

def getMaskedFoodFromPlate(image_cropped):
    filters = build_filters()
    image_cropped_textured = process(image_cropped,filters)
    #cv2.imshow("image_cropped_textured", image_cropped_textured)
    mask = np.zeros(image_cropped_textured.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (200,200,int(image_cropped_textured.shape[1]-350),int(image_cropped_textured.shape[0]-350))
    cv2.grabCut(image_cropped_textured,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    final_image = image_cropped*mask2[:,:,np.newaxis]
    return final_image

def put_to_corner(image):       
    final_picture = np.zeros((x_to_crop,y_to_crop),np.uint8)

    for x in range(final_picture.shape[0]):
        if(image.shape[0]<=x):
            break    
        for y in range(final_picture.shape[1]):      
            if(image.shape[1]<=y):
                break
            final_picture[x][y] +=image[x][y]

    return final_picture

def transform_to_array(croppedMaskedCornered):
    element_list = []
    for x in range(croppedMaskedCornered.shape[0]):
        for y in range(croppedMaskedCornered.shape[1]):
            element_list.append(croppedMaskedCornered[x][y])
    
    return element_list
cc.read_food_features()

label_encoder =  LabelEncoder()
label_binarizer = LabelBinarizer()

#full_dataset_data_frame = reader.read_dataset_cv("sveTop")
full_dataset_data_frame = reader.read_dataset_cv("testCNN")

Y = full_dataset_data_frame['foodType']
X = full_dataset_data_frame.drop(['food2_ymax', 'food2_xmax', 'food2_ymin', 'food2_xmin', 'foodType2', 'foodType'], axis='columns')
temp_binarised = label_binarizer.fit_transform(np.array(Y))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

x_train.reset_index(inplace=True, drop=True)
x_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)

dataset = x_train['fullImage']
x_test_dataset = x_test['fullImage']
    
coin_ymax = x_train['coin_ymax']
coin_xmax = x_train['coin_xmax']
coin_ymin = x_train['coin_ymin']
coin_xmin = x_train['coin_xmin']

food_ymax = x_train['food_ymax']
food_xmax = x_train['food_xmax']
food_ymin = x_train['food_ymin']
food_xmin = x_train['food_xmin']

foodType = y_train
"""
food2_ymax = x_train['food2_ymax']
food2_xmax = x_train['food2_xmax']
food2_ymin = x_train['food2_ymin']
food2_xmin = x_train['food2_xmin']

foodType = x_train['foodType']
foodType2 = x_train['foodType2']
"""
my_coin_radius_pixels = []
dataset_coin_radius_pixels = []

final_croped_masked_images = []

full_image_x_dim = 612
full_image_y_dim = 816
croped_image_dividing_const = 12
full_image_dividing_const  = 18

x_to_crop = int(full_image_x_dim/full_image_dividing_const)
y_to_crop = int(full_image_y_dim/full_image_dividing_const)

kmeans_x_train = pd.DataFrame(columns=range(x_to_crop*y_to_crop))
kmeans_y_train = []

cnn_x_train = []
cnn_y_train = []

for i in range(len(dataset)):
    #original_image = full_dataset_data_frame['fullImage'][i]
    original_image = dataset[i]
    image = original_image.copy()

    #CNN
    cnn_x_train.append(cv2.resize(image, None, fx=0.15, fy=0.15))
    cnn_y_train.append([])
    cnn_y_train[i].append(y_train[i])

    # cropping food from image
    image_cropped = image[int(food_ymin[i]):int(food_ymax[i]), int(food_xmin[i]):int(food_xmax[i])]
    #cv2.imshow("imshow"+str(i), image_cropped)
    #cv2.waitKey(0)

    #mask return croped image in bw format
    cropedMasked = getMaskedCropedImage(image_cropped)
    #cv2.imshow("imshow"+str(i), cropedMasked)
    #cv2.waitKey(0)

    resized = image_resize(cropedMasked, int(cropedMasked.shape[0]/croped_image_dividing_const), int(cropedMasked.shape[1]/croped_image_dividing_const))
    #cv2.imshow("imshow"+str(i), resized)
    #cv2.waitKey(0)

    cropedMasked_cornered = put_to_corner(resized)
    #cv2.imshow("imshow"+str(i), cropedMasked_cornered)
    #cv2.waitKey(0)

    image_as_array = transform_to_array(cropedMasked_cornered)
    pds = pd.DataFrame(data = [image_as_array])
    #KMeans
    kmeans_x_train = kmeans_x_train.append(pds, ignore_index=True)
    kmeans_y_train.append(y_train[i])
    """
    if(foodType2[i] != ""):
        image_cropped2 = image[int(food2_ymin[i]):int(food2_ymax[i]), int(food2_xmin[i]):int(food2_xmax[i])]
        cropedMasked2 = getMaskedCropedImage(image_cropped2)
        resized = image_resize(cropedMasked, cropedMasked.shape[0]/croped_image_dividing_const, cropedMasked.shape[1]/croped_image_dividing_const)
        cropedMasked_cornered2 = put_to_corner(resized)
        image_as_array2 = transform_to_array(cropedMasked_cornered2)
        pds = pd.DataFrame(data = [image_as_array2])
        kmeans_x_train = kmeans_x_train.append(pds, ignore_index=True)
        kmeans_y_train.append(foodType2[i])
    """   
    print(int(i+1),"/", len(dataset))

kmeans_x_test = pd.DataFrame(columns=range(x_to_crop*y_to_crop))
kmeans_y_test = []

cnn_x_test = []
cnn_y_test = []

food_index_to_features = dict()

for i in range(len(x_test_dataset)):  
    
    min_x, min_y = full_image_x_dim, full_image_y_dim
    max_x = max_y = 0
    
    image = x_test_dataset[i].copy()    
    
    cnn_x_test.append(cv2.resize(image, None, fx=0.15, fy=0.15))
    cnn_y_test.append([])
    cnn_y_test[i].append(y_test[i])  

    #normal brigthness
    image = bn.brightness_normalize(image)
    
    # separated plate from background with food
    plate_image = imSeg.separate_plate_from_background(image)
    #cv2.imshow('plateImage%d' %(i), plate_image)
    #cv2.waitKey(0)

    food_image, food_area, food_contour = imSeg.get_food(plate_image)

    #croping image to stay only food
    #x,y center of contour
    (x,y,w,h) = cv2.boundingRect(food_contour)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    image_cropped = food_image[min_y:max_y, min_x:max_x]
    #cv2.imshow('image_cropped%d' %(i), image_cropped)
    #cv2.waitKey(0)
    
    #croped food and transform in bw image
    cropedMasked = getMaskedCropedImage(image_cropped)
    #cv2.imshow('image_cropped%d' %(i), cropedMasked)
    #cv2.waitKey(0)

    #resize and put to corner
    resized = image_resize(cropedMasked, int(cropedMasked.shape[0]/croped_image_dividing_const), int(cropedMasked.shape[1]/croped_image_dividing_const))
    cropedMasked_cornered = put_to_corner(resized)

    image_as_array = transform_to_array(cropedMasked_cornered)
    pds = pd.DataFrame(data = [image_as_array])
    kmeans_x_test = kmeans_x_test.append(pds, ignore_index=True)
    kmeans_y_test.append(y_test[i])

    coin_radius = cm.getCoinSize(image)
    coin_area, pix_to_cm_multiplier = cc.get_coin_info(coin_radius)
    
    food_index_to_features[i] = [food_area, food_contour, coin_area, pix_to_cm_multiplier] 
    #cv2.waitKey(0)

y_pred_cnn = cnn.apply_cnn(cnn_x_train, cnn_x_test, cnn_y_train, cnn_y_test)

y_train_encoded = label_encoder.fit_transform(kmeans_y_train)
y_test_encoded = label_encoder.fit_transform(kmeans_y_test)

y_pred_kmeans = apply_model(kmeans_x_train, kmeans_x_test, y_train_encoded, y_test_encoded)

for i in range(len(y_pred_kmeans)):
    volume = cc.get_volume(y_pred_kmeans[i], food_index_to_features[i][0], food_index_to_features[i][1], food_index_to_features[i][2], food_index_to_features[i][3])
    mass, calories = cc.get_calorie(y_pred_kmeans[i], volume)
    print(str(y_pred_kmeans[i])+': '+str(round(volume,2))+'cm^3, '+str(round(mass,2))+'g, '+str(round(calories,2))+'kcal.')


cv2.waitKey(0)
