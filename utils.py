import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches
from matplotlib.patches import Rectangle

def display_image_and_box(img, bbox_file, name):

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
    #plt.savefig("test.jpg")
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
    plt.show()


def pad_image_labels(current_img, label_file):
    
    true_width, true_height = current_img.shape[0], current_img.shape[1]
    print(current_img.shape)

    modify_bbox = False

    # (4128, 2322, 3)
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

    elif true_width == true_height:

        new_width = int(true_width / 0.75)
        color = (0,0,0)
        padded_image = np.full((new_width, true_height, 3), color, dtype=np.uint8)

        padded_image[:true_width, :] = current_img
        whoisbeingpad = "width"

        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #ax1.imshow(current_img)
        #ax2.imshow(result)

    else:

        raise NotImplementedError("Fuck those images")


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