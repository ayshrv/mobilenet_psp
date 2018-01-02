import os
import glob

# Put this file in Cityscapes dataset folder which contains leftImg8bit, gtCoarse & gtFine Folders

CASE = 0

params = [ ['train', 'coarse', 'coarse_train'],
           ['train_extra', 'coarse', 'coarse_train_extra'],
           ['train', 'fine', 'train_list'],
           ['val', 'fine', 'eval_list'] ]

DIR = params[CASE][0]
TYPE = params[CASE][1]
LIST = params[CASE][2]


def image_annotation_lists(dir, type): #type: fine/coarse

    image_list = []
    label_list = []

    file_glob = os.path.join("leftImg8bit", dir, "*",  "*_leftImg8bit." + "png")
    image_list.extend(glob.glob(file_glob))

    if not image_list:
        print('No images found')
        return None

    for i in image_list:
        filename = os.path.splitext(i.split('/')[-1])[0][:-len("_leftImg8bit")]
        foldername = i.split('/')[-2]
        if type == 'fine':
            annotation_file = os.path.join("gtFine", dir, foldername, filename + "_gtFine_labelTrainIds" + ".png")
        elif type == 'coarse':
            annotation_file = os.path.join("gtCoarse", dir, foldername, filename + "_gtCoarse_labelTrainIds" + ".png")
        else:
            print("type not found.")
            return None
        if os.path.exists(annotation_file) == False:
            print("Annotation file not found for %s - Error!" % filename)
            return None
        label_list.append(annotation_file)

    return image_list, label_list

def write_to_file(image_list, label_list, list_name):
    file = open(list_name+'.txt', 'w')
    for image, label in zip(image_list, label_list):
        s = "{0} {1}\n".format(image, label)
        file.write(s)
    file.close()




if image_annotation_lists(DIR, TYPE) != None:
    image_list, label_list = image_annotation_lists(DIR, TYPE)
    assert(len(image_list) == len(label_list))
    write_to_file(image_list, label_list, LIST)
else:
    print("Error!")
