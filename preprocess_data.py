import os
from os import path
import xml.etree.ElementTree as ET
import cv2

import numpy as np
import shutil as sh

import matplotlib.pyplot as plt

from utils import pad_image_labels, display_image_and_box_list, display_image_and_box, xmlbox2str, listbox2str, calculate_slice_bboxes, filebbox2list

import albumentations as A 

import yaml
import fnmatch
import json


def coffee_data_convert_xml_to_coco(source='coffee_raw', output_path='coffee_raw/coco_labels'):
    """
    The bounding boxes in the original dataset are stored in XML file and coordinates in format (corner_x, corner_y, width, height)
    Coco format is (center_x, center_y, width, height) need to convert each XML file into a txt, containing for each image, all bounding boxes.
    """

    # The file name correspond to its bounding box : Etiquetas/1.xml correspond to the Imagenes/1.jpg 
    label_paths = path.join(source, "Etiquetas")
    image_paths = path.join(source, "Imagenes")

    if not os.path.exists(output_path):
        # Create a new directory because it does not exist
        os.makedirs(output_path)
    else:
        print("labels were already created")
        return

    total_images = dict()

    # Not used yet, in the future, will hold the classes
    class_to_index = dict()

    for label_path in os.listdir(label_paths):
        label_full_path = path.join(label_paths, label_path)
        img_name, _ = label_path.split('.')

        img_full_path = path.join(image_paths, img_name) + ".jpg"
        img = cv2.imread(img_full_path)

        if img.shape[2] != 3:
            
            raise NotImplementedError(f"Something is wrong with those images, expected format is (width, height, channel) is : {img.shape}")
            print(img_full_path)
            print(img.shape, img_full_path, img)
            #new_img = np.swapaxes(img.copy(), 0,2)#.swapaxes(1,2)
            #print(img.shape)
            assert cv2.imwrite(img_full_path, new_img) 
            img = new_img.copy()
            break

        img_true_height, img_true_width, depth = img.shape
        assert depth == 3, f"depth = {depth}"
        
        # Parse XML using ET
        xml_tree = ET.parse(label_full_path)
        root = xml_tree.getroot()

        # Width and Height vary
        img_size = root.find("size")
        img_width = int(img_size.find("width").text)
        img_height = int(img_size.find("height").text)

        # Image size describe in the XML should be the same as true image size (not always the case)
        if img_true_width != img_width or img_true_height != img_true_height :
            print(f"""Width and height of the image and XML info differs :
                True image size: {img_true_width} {img_true_height}
                XML info : {img_width} {img_height}
                for img {label_full_path}""")
            
        img_width = img_true_width
        img_height = img_true_height

        # Counting images per size
        if (img_width, img_height) not in total_images:
            total_images[(img_width, img_height)] = 0       
        total_images[(img_width, img_height)] += 1
 
        # Create txt containing all boxes for the current img
        bbox_str_path = img_name + ".txt"
        bbox_str_path = path.join(output_path, bbox_str_path)

        annotations = root.findall("object")
        str_full = xmlbox2str(annotations=annotations,
                              img_width=img_width,
                              img_height=img_height)

        with open(bbox_str_path, "w") as f:
            f.write(str_full)


    print(f"Number of images per size {total_images}")


def adapt_classes(labels_path, classes_json_path, convert_all=True, to_ignore=None):

    annotations = json.load(open(classes_json_path, 'r'))
    annotations = [i["name"] for i in annotations["categories"]]

    old_annotations = annotations.copy()

    if not convert_all and to_ignore is None:
        return
    
    if to_ignore is not None:
        for annotation in to_ignore:
            index_to_remove = annotations.index(annotation)
            annotations.pop(index_to_remove)
    else:
        to_ignore = []

    anot2index = dict([(key, value) for key, value in enumerate(annotations)])

    for bbox_txt in os.listdir(labels_path):

        bbox_full_path = os.path.join(labels_path, bbox_txt)
        bbox_file = open(bbox_full_path, 'r')
        new_bbox_str = ""

        for bbox in bbox_file.readlines():
            bbox_list = bbox[:-1].split(" ")
            label = bbox_list[0]

            if label not in to_ignore:
                if convert_all: 
                    bbox_list[0] = str(0)
                else:
                    bbox_list[0] = anot2index[old_annotations[bbox_list[0]]] 

                new_bbox_str += " ".join(bbox_list) + "\n"

        bbox_file.close()
        new_bbox_file = open(bbox_full_path, 'w')
        new_bbox_file.write(new_bbox_str)
        new_bbox_file.close()

    return



def preprocess_image(raw_images_dir, raw_label_dir, output_path="coffee_coco", preprocess='both', downsize=(768,1024)):
    """
    Preprocess all images located in raw_images_dir
    Two options for larger images are : 
        - Cropping the image to smaller patches
        - pad and downsizing heavily

    preprocess can be : crop, pad or both

    You can apply both here.
    """

    count_image = 0
    total_crop = 0

    if not os.path.exists(output_path):
        # Create a new directory because it does not exist
        os.makedirs(output_path)

    problematic_images = {"1615315525581"} # Labels are fucked up for those, ignore them

    for i, image_path in enumerate(os.listdir(raw_images_dir)):

        ignore = False # Flag used to ignore padding if the crop already contains all the boxes

        if image_path[:-4] in problematic_images:
           continue
        
        # Open image
        image_full_path = path.join(raw_images_dir, image_path)
        current_img = cv2.imread(image_full_path)
        img_shape = current_img.shape
        
        ####### WEIRD CASES ####################################
        if image_path[:-4] == "1615385610373": 
          current_img = cv2.rotate(current_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if image_path[:-4] == "1615581114578": 
          current_img = cv2.rotate(current_img, cv2.ROTATE_180)
        ###########################################

        # Load bboxes
        label_path = path.join(raw_label_dir, image_path)[:-3] + "txt"

        # Crop big images into smaller patches
        if preprocess in ['crop', 'both']:
            label_file = open(label_path)
            bboxes, labels = filebbox2list(label_file)
            total_bbox_full = len(bboxes)

            if total_bbox_full == 0:
                print("No bbox found on ", image_path)
                continue
                
            subimgs = calculate_slice_bboxes(image_width=img_shape[1], image_height=img_shape[0],
                                             slice_width=768, slice_height=1024, overlap_height_ratio=0.05, overlap_width_ratio=0.05)

            n_crop = 0
            invalid_bbox = False

            for subimg in subimgs:
                total_crop += 1
                crop_transform = A.Compose(
                    [A.Crop(*subimg),],
                    bbox_params=A.BboxParams(format='yolo',
                                             label_fields=['labels'],
                                             min_visibility=0.1,
                                             min_area=0.1)
                )

                try:
                    transformed = crop_transform(image=current_img, bboxes=bboxes, labels=labels)
                except ValueError:
                    invalid_bbox = True
                    break

                # Crop that contains no box or less, remove
                if len(transformed['bboxes']) > 0:

                    img_new_path = image_path[:-4] + f"c{n_crop}" + ".jpg"
                    img_new_path = path.join(output_path, img_new_path)

                    label_new_path = img_new_path[:-3] + "txt"

                    bbox_str = ""
                    for label, bbox in zip(transformed['labels'], transformed['bboxes']):
                        bbox_str += str(label[0]) + " " + " ".join(map(str, bbox)) + "\n"
                    
                    n_crop += 1
                    count_image += 1

                    with open(label_new_path, "w") as f:
                        f.write(bbox_str)

                    # Resize and save image 
                    img = cv2.resize(transformed['image'], dsize=downsize)
                    cv2.imwrite(img_new_path, img)

                    # If the number of bbox within a crop is the same as the original one, can ignore others.
                    if len(transformed['bboxes']) == total_bbox_full:
                        ignore = True
                        break

            if invalid_bbox:
                print("Problem in bbox value, contains a value bigger than 1")
                continue
            
            assert n_crop > 0, "problem no bbox in this image crop"


        
        # Pad and dowsize
        if preprocess in ["pad", "both"] and not ignore:

            img_new_path = path.join(output_path, image_path)
            label_new_path = img_new_path[:-3] + "txt"
            
            label_file.seek(0) # just in case, rewind the file
            padded_image, bbox_list = pad_image_labels(current_img, label_file)
            downsized_img = cv2.resize(padded_image, dsize=downsize)            
            cv2.imwrite(img_new_path, downsized_img)
            
            bbox_str = listbox2str(bbox_list=bbox_list)
            with open(label_new_path, "w") as f:
                f.write(bbox_str)

    print(f"Total images reformat : {count_image}")


def split_train_val(datadir, proportion_trainval=0.85):
    """
    At the moment, no split are predefinied, so we split by hand 
    proportion_trainval defines the proportion of images going to the train set (around 80-85% usually)

    subname allows to define multiple datasets
    """

    # Count the number of images in the dir
    count_image = len(fnmatch.filter(os.listdir(datadir), '*.jpg'))
    print(f"Number of images in {datadir} : {count_image}")

    proportion_trainval = 0.80
    n_train = int(count_image * proportion_trainval)

    count_copied_image = 0
    subname = datadir

    train_dir = f"datasets/{subname}/train"
    val_dir = f"datasets/{subname}/val"

    if not os.path.exists("datasets"):
        os.mkdir(f"datasets")

    if not os.path.exists(f"datasets/{subname}"):
        os.mkdir(f"datasets/{subname}")

        os.mkdir(train_dir)
        os.mkdir(val_dir)

        os.mkdir(path.join(train_dir, "images"))
        os.mkdir(path.join(train_dir, "labels"))

        os.mkdir(path.join(val_dir, "images"))
        os.mkdir(path.join(val_dir, "labels"))

    for image_path in fnmatch.filter(os.listdir(datadir), '*.jpg'):

        image_full_path = path.join(datadir, image_path)

        if count_copied_image < n_train:
            image_new_full_path = path.join(train_dir, "images", image_path)
            label_new_full_path = path.join(train_dir, "labels", image_path)

        else:
            image_new_full_path = path.join(val_dir, "images", image_path)
            label_new_full_path = path.join(val_dir, "labels", image_path)


        # Copy image AND label txt to path
        sh.copy(image_full_path, image_new_full_path)
        sh.copy(image_full_path[:-3]+"txt", label_new_full_path[:-3]+"txt")

        count_copied_image += 1

    print("Finished Splitting")

if __name__ == "__main__":

    import argparse

    #coffee_data_convert_xml_to_coco()

    sh.rmtree("croppieV2/labels")
    sh.copytree("/home/mseurin/Téléchargements/labels/", "croppieV2/labels")

    #preprocess_image(raw_images_dir="croppieV2/images", raw_label_dir="croppieV2/labels", output_path="coffee_coco1024", downsize=(768,1024))
    adapt_classes(labels_path="croppieV2/labels", classes_json_path="croppieV2/notes.json", convert_all=True, to_ignore=["low_visibility_unsure"])
    preprocess_image(raw_images_dir="croppieV2/images", raw_label_dir="croppieV2/labels", output_path="coffee_coco640", downsize=(480,640))

    split_train_val(datadir="coffee_coco640")
    #split_train_val(datadir="coffee_coco1024", subname="coffee1024")


    wheet_yaml640 = dict(
    train ='coffee640/train',
    val ='coffee640/val',
    nc =1,
    names =["cafe verde"]
    )

    with open('coffee640.yaml', 'w') as outfile:
        yaml.dump(wheet_yaml640, outfile, default_flow_style=True)


    # wheet_yaml1024 = dict(
    # train ='coffee1024/train',
    # val ='coffee1024/val',
    # nc =1,
    # names =["cafe verde"]
    # )

    # with open('coffee1024.yaml', 'w') as outfile:
    #     yaml.dump(wheet_yaml1024, outfile, default_flow_style=True)