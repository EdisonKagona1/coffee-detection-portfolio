import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches
from matplotlib.patches import Rectangle
import cv2

def display_image_and_box(img, bbox_file, name, save=False):

    height, width = img.shape[0], img.shape[1]

    # Display the image

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img)
    ax2.imshow(img)

    ax2 = plt.gca()

    for bbox_str in bbox_file.readlines():
        object_class, center_x, center_y, width_x, width_y = tuple(map(float, bbox_str.split(" ")))
        anchor_x = (center_x - width_x / 2) * width
        anchor_y = (center_y - width_y / 2) * height
        width_x *= width
        width_y *= height

        ax2.add_patch(
            patches.Rectangle(
                (anchor_x, anchor_y),
                width_x,
                width_y,
                fill=True
            ) ) 
    
    plt.title(name)

    if save:
        plt.savefig(f"{name}bbox_test.jpg")
        cv2.imwrite(f"{name}original_image.jpg", img)

    plt.show()



def display_image_and_box_list(img, bbox_list, name):

    height, width = img.shape[0], img.shape[1]

    # Display the image

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img)
    ax2.imshow(img)

    ax2 = plt.gca()

    for object_class, center_x, center_y, width_x, width_y in bbox_list:
        
        anchor_x = (center_x - width_x / 2) * width
        anchor_y = (center_y - width_y / 2) * height
        width_x *= width
        width_y *= height

        ax2.add_patch(
            patches.Rectangle(
                (anchor_x, anchor_y),
                width_x,
                width_y,
                fill=True
            ) ) 
    
    plt.title(name)
    #plt.savefig(f"testout_{name}.png")
    plt.show()


def pad_image_labels(current_img, label_file):
    
    #print(current_img.shape)
    true_width, true_height = current_img.shape[0], current_img.shape[1]

    modify_bbox = False

    if true_width > true_height:
        ratio = true_width / true_height
        
        if ratio > 4/3: # Pad Height
            new_height = int(true_width * 0.75)
            color = (0,0,0)
            padded_image = np.full((true_width, new_height, 3), color, dtype=np.uint8)

            padded_image[:, :true_height] = current_img

            whoisbeingpad = "height"
            modify_bbox = True

            #fig, (ax1, ax2) = plt.subplots(1, 2)
            #ax1.imshow(current_img)
            #ax2.imshow(result)

        elif ratio == 4/3:
            padded_image = current_img

        else:           # Pad Width
            print(true_width, true_height, ratio)
            raise NotImplementedError("Fuck those images")
        
    elif true_width < true_height:
        ratio = true_height / true_width

        if ratio > 4/3:  #Pad width
            new_width = int(true_height * 0.75)
            color = (0,0,0)
            padded_image = np.full((new_width, true_height, 3), color, dtype=np.uint8)

            padded_image[:true_width, :] = current_img

            whoisbeingpad = "width"
            modify_bbox = True
        elif ratio < 4/3:
            print("ratio < 4/3")
            raise NotImplementedError("Fuck those images")
        else:
            padded_image = current_img


    elif true_width == true_height:

        new_width = int(true_width / 0.75)
        color = (0,0,0)
        padded_image = np.full((new_width, true_height, 3), color, dtype=np.uint8)

        padded_image[:true_width, :] = current_img
        
        whoisbeingpad = "width"
        modify_bbox = True

        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #ax1.imshow(current_img)
        #ax2.imshow(result)

    new_bboxes = []
    for bbox_str in label_file.readlines():
        object_class, center_x, center_y, width_x, width_y = tuple(map(float, bbox_str.split(" ")))

        if modify_bbox:
            if whoisbeingpad == "height":
                oldnew_ratio = new_height / true_height
                center_x /= oldnew_ratio
                width_x /= oldnew_ratio

            else:
                oldnew_ratio = new_width / true_width
                center_y /= oldnew_ratio
                width_y /= oldnew_ratio

        new_bboxes.append([object_class, center_x, center_y, width_x, width_y])

    return padded_image, new_bboxes


def xmlbox2str(annotations, img_width, img_height):

    str_full = ""

    # Reformat each bounding box to one info per line
    for annotation in annotations:

        # Retrieve coordinates
        bndbox = annotation.find("bndbox")
        xmin, ymin = int(bndbox.find("xmin").text), int(bndbox.find("ymin").text)
        xmax, ymax = int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)


        # Convert [rectangle dimensions] to [center and rectangle width and height] in percentage of the total image
        width_x, width_y = xmax - xmin, ymax - ymin
        center_x, center_y = (xmax - xmin) / 2 + xmin, (ymax - ymin) / 2 + ymin

        # Retrieve class
        object_class = annotation.find("name").text

        # Create new index if the class does not exist, allow flexible number of cafe class in training
        ################## NOT AVAILABLE AT THE MOMENT ##########
        #########################################################
        #if object_class in class_to_index:
        #    index = class_to_index[object_class]
        #else:
        #    index = len(class_to_index)
        #    class_to_index[object_class] = index

        # Only one class at the moment
        index = 0

        str_full += " ".join(map(str, [index,
                                        center_x / img_width, center_y / img_height,
                                        width_x / img_width, width_y / img_height]))
        str_full += "\n"

    return str_full


def listbox2str(bbox_list):
    bbox_str = ""

    # oneliner challenge
    for bbox in bbox_list:
        bbox_str += " ".join(map(str, bbox))
        bbox_str += "\n"

    # oneliner challenge
    # bbox_str = "\n".join([" ".join(map(str, bbox)) for bbox in bbox_list])

    return bbox_str



######################## NOT USED AT THE MOMENT #########

def split_image(current_img, label_file):
    
    true_width, true_height = current_img.shape[0], current_img.shape[1]

    kept_bbox_count = 0
    bboxs = []

    for bbox_str in label_file.readlines():
        object_class, center_x, center_y, width_x, width_y = tuple(map(float, bbox_str.split(" ")))
        
        width_x *= true_width
        width_y *= true_height
        center_x *= true_width
        center_y *= true_height
        corner_x = center_x - width_x // 2
        corner_y = center_y - width_y // 2

        bboxs.append([corner_x, corner_y, width_x, width_y])

    if true_width > true_height :

        #(4128, 2322, 3) example
        need_to_reduce = true_height - true_width // 2

        if need_to_reduce < 0:
            assert NotImplementedError("Warning, image not dealt with")

        bboxs.sort(key=lambda x:x[1])
        print(bboxs)

        bboxs = 0 


        image1 = current_img[:width//2, :]
        image2 = current_img[width//2:, :]
    else:
        image1 = current_img[:, :height//2]
        image2 = current_img[:, height//2:]
        

        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #ax1.imshow(image1)
        #ax2.imshow(image2)