import glob
import pandas as pd
import cv2
import xml.etree.ElementTree as et

def read_dataset_cv(path_to_folder):
    files = [file for file in glob.glob(path_to_folder+ "/JPEGImages/*.jpg")]
    images = [cv2.imread(file) for file in files]
    file_names = [file.split('\\')[-1] for file in files]
    
    metadata_dataset_filenames = []
    metadata_dataset_food_type = []
    metadata_dataset_food_type2 = []

    metadata_dataset_bound_box_xmin = []
    metadata_dataset_bound_box_ymin = []
    metadata_dataset_bound_box_xmax = []
    metadata_dataset_bound_box_ymax = []

    metadata_dataset_bound_box2_xmin = []
    metadata_dataset_bound_box2_ymin = []
    metadata_dataset_bound_box2_xmax = []
    metadata_dataset_bound_box2_ymax = []

    metadata_dataset_coin_bound_box_xmin = []
    metadata_dataset_coin_bound_box_ymin = []
    metadata_dataset_coin_bound_box_xmax = []
    metadata_dataset_coin_bound_box_ymax = []


    for file_name in file_names:
        file_path = path_to_folder + "/Annotations/"+ file_name.split('.')[0]+ ".xml"

        xml_tree = et.parse(file_path)
        root = xml_tree.getroot()

        if(root[6][4][0].text == ""): # if xml is empty continue
            continue

        if(root[4][0].text != str(816)): # if resolution of image is not appropriate
            continue

        if(root[4][1].text != str(612)):
            continue

        metadata_dataset_filenames.append(root[1].text)

        metadata_dataset_bound_box_xmin.append(int(root[6][4][0].text))
        metadata_dataset_bound_box_ymin.append(int(root[6][4][1].text))
        metadata_dataset_bound_box_xmax.append(int(root[6][4][2].text))
        metadata_dataset_bound_box_ymax.append(int(root[6][4][3].text))

        if(root[7][0].text == "coin"):
            metadata_dataset_food_type.append(root[6][0].text)
            metadata_dataset_food_type2.append("")

            metadata_dataset_bound_box2_xmin.append("")
            metadata_dataset_bound_box2_ymin.append("")
            metadata_dataset_bound_box2_xmax.append("")
            metadata_dataset_bound_box2_ymax.append("")
            
            metadata_dataset_coin_bound_box_xmin.append(int(root[7][4][0].text))
            metadata_dataset_coin_bound_box_ymin.append(int(root[7][4][1].text))
            metadata_dataset_coin_bound_box_xmax.append(int(root[7][4][2].text))
            metadata_dataset_coin_bound_box_ymax.append(int(root[7][4][3].text))

        elif(len(root) >= 8):
            try:
                metadata_dataset_food_type.append(root[6][0].text)
                metadata_dataset_food_type2.append(root[7][0].text)
    
                metadata_dataset_bound_box2_xmin.append(int(root[7][4][0].text))
                metadata_dataset_bound_box2_ymin.append(int(root[7][4][1].text))
                metadata_dataset_bound_box2_xmax.append(int(root[7][4][2].text))
                metadata_dataset_bound_box2_ymax.append(int(root[7][4][3].text))
                
                metadata_dataset_coin_bound_box_xmin.append(int(root[8][4][0].text))
                metadata_dataset_coin_bound_box_ymin.append(int(root[8][4][1].text))
                metadata_dataset_coin_bound_box_xmax.append(int(root[8][4][2].text))
                metadata_dataset_coin_bound_box_ymax.append(int(root[8][4][3].text))
            except:
                print(file_name)
                
    image_data_frame = pd.DataFrame(index=range(len(file_names)), data ={'fileName' : file_names,'fullImage': images})
    image_meta_data =  {"fileName": metadata_dataset_filenames, "foodType": metadata_dataset_food_type, "foodType2": metadata_dataset_food_type2, 
                        "food_xmin": metadata_dataset_bound_box_xmin, "food_ymin": metadata_dataset_bound_box_ymin, 
                        "food_xmax": metadata_dataset_bound_box_xmax, "food_ymax": metadata_dataset_bound_box_ymax,
                        "food2_xmin": metadata_dataset_bound_box2_xmin, "food2_ymin": metadata_dataset_bound_box2_ymin, 
                        "food2_xmax": metadata_dataset_bound_box2_xmax, "food2_ymax": metadata_dataset_bound_box2_ymax, 
                        "coin_xmin": metadata_dataset_coin_bound_box_xmin, "coin_ymin": metadata_dataset_coin_bound_box_ymin, 
                        "coin_xmax": metadata_dataset_coin_bound_box_xmax, "coin_ymax": metadata_dataset_coin_bound_box_ymax}
    image_meta_data_frame = pd.DataFrame(index = range(len(metadata_dataset_bound_box_ymax)), data = image_meta_data)
    full_dataset_data_frame = pd.merge(image_meta_data_frame, image_data_frame, left_on='fileName', right_on='fileName', how='left')

    return full_dataset_data_frame