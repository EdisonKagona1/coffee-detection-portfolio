import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches
from matplotlib.patches import Rectangle
import cv2

from matplotlib.pyplot import figure
import subprocess

def display_image_and_box(img, bbox_file, name, save=False, display=True):

    height, width = img.shape[0], img.shape[1]

    # Display the image
    #figure(figsize=(16, 12), dpi=160)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=160, figsize=(16, 12))

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
        plt.savefig(f"ALL_LABELS/{name}")
        #cv2.imwrite(f"{name}original_image.jpg", img)
    
    if display:
        plt.show()
    
    fig.clear()
    plt.close()



def display_image_and_box_list(img, bbox_list, name, label_present=True):

    height, width = img.shape[0], img.shape[1]

    # Display the image

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img)
    ax2.imshow(img)

    ax2 = plt.gca()

    for bbox in bbox_list:

        if label_present:
            object_class, center_x, center_y, width_x, width_y = bbox
        else:
            center_x, center_y, width_x, width_y = bbox

        
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
            new_width = int(true_height * 4/3)
            color = (0,0,0)
            padded_image = np.full((new_width, true_height, 3), color, dtype=np.uint8)

            padded_image[:true_width, :] = current_img

            whoisbeingpad = "width"
            modify_bbox = True
        
    elif true_width < true_height:
        ratio = true_height / true_width

        if ratio > 4/3:  #Pad width
            new_width = int(true_height * 0.75)
            color = (0,0,0)
            padded_image = np.full((new_width, true_height, 3), color, dtype=np.uint8)

            padded_image[:true_width, :] = current_img

            whoisbeingpad = "width"
            modify_bbox = True

        elif ratio < 4/3: # Pad Height
            new_height = int(true_width * 4/3)
            color = (0,0,0)
            padded_image = np.full((true_width, new_height, 3), color, dtype=np.uint8)

            padded_image[:, :true_height] = current_img

            whoisbeingpad = "height"
            modify_bbox = True
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

def filebbox2list(label_file):
    bboxes = []
    labels = []
    for bbox_str in label_file.readlines():
        class_and_bbox = tuple(map(float, bbox_str.split(" ")))
        labels.append([0])
        bboxes.append(class_and_bbox[1:])
    return bboxes, labels


def calculate_slice_bboxes(image_height, image_width, slice_height=768, slice_width=1024, overlap_height_ratio=0.1, overlap_width_ratio=0.1):
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.

    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param slice_height: Height of each slice
    :param slice_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format
    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass
    
    # true_width, true_height = current_img.shape[0], current_img.shape[1]

    # kept_bbox_count = 0
    # bboxs = []

    # for bbox_str in label_file.readlines():
    #     object_class, center_x, center_y, width_x, width_y = tuple(map(float, bbox_str.split(" ")))
        
    #     width_x *= true_width
    #     width_y *= true_height
    #     center_x *= true_width
    #     center_y *= true_height
    #     corner_x = center_x - width_x // 2
    #     corner_y = center_y - width_y // 2

    #     bboxs.append([corner_x, corner_y, width_x, width_y])

    # if true_width > true_height :

    #     #(4128, 2322, 3) example
    #     need_to_reduce = true_height - true_width // 2

    #     if need_to_reduce < 0:
    #         assert NotImplementedError("Warning, image not dealt with")

    #     bboxs.sort(key=lambda x:x[1])
    #     print(bboxs)

    #     bboxs = 0 


    #     image1 = current_img[:width//2, :]
    #     image2 = current_img[width//2:, :]
    # else:
    #     image1 = current_img[:, :height//2]
    #     image2 = current_img[:, height//2:]
        

    #     #fig, (ax1, ax2) = plt.subplots(1, 2)
    #     #ax1.imshow(image1)
    #     #ax2.imshow(image2)