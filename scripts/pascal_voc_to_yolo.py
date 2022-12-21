import glob
import os
import argparse
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import shutil


dirs = ['train', 'validation']
sub_dirs = ["images", "annotations"]
classes = []

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(input_ann_path):

    tree = ET.parse(input_ann_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    ann_list = []

    for obj in root.iter('object'):
        obj_class = obj.find('name').text
        if obj_class not in classes:
            classes.append(obj_class)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)

        ann_list.append(
            {
                "class": obj_class,
                "bbox": bb
            }
        )

    return ann_list
        # output_ann_path.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def main(dataset_dir: str):
    yolo_dataset = os.path.join(
        os.path.dirname(dataset_dir),
        os.path.basename(f"{dataset_dir}-yolo")
    )
    for dir in dirs:
        dir_path = os.path.join(
            yolo_dataset,
            dir
        )
        os.makedirs(dir_path, exist_ok=True)

        for sub_dir in sub_dirs:
            os.makedirs(
                os.path.join(
                    dir_path,
                    sub_dir
                ),
                exist_ok=True
            )
        
    train_anns = {}
    validation_anns = {}

    for dir in dirs:
        dir_path = os.path.join(
            dataset_dir,
            dir
        )

        images = [file for file in os.listdir(
            os.path.join(dir_path, "images")
        ) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")]

        annotations = [file for file in os.listdir(
            os.path.join(dir_path, "annotations")
        ) if file.endswith(".xml")]

        for image, annotation in zip(images, annotations):
            shutil.copy(
                os.path.join(
                    dataset_dir,
                    dir,
                    "images",
                    image
                ),
                os.path.join(
                    yolo_dataset,
                    dir,
                    "images",
                    image
                )
            )

            ann_list = convert_annotation(
               os.path.join(
                    dataset_dir,
                    dir,
                    "annotations",
                    annotation
                ) 
            )
            if dir == "train":
                train_anns[annotation] = ann_list
            elif dir == "validation":
                validation_anns[annotation] = ann_list
    
    all_classes = sorted(classes)

    for k,v in {"train": train_anns, "validation": validation_anns}.items():
        for anns_k, anns_v in v.items():
            output_ann_path = os.path.join(
                yolo_dataset, k, "annotations", anns_k.replace(".xml", ".txt")
            )
            anns_str = ""
            for ann in anns_v:
                class_idx = all_classes.index(ann["class"])
                bbox = [str(f) for f in ann["bbox"]]
                anns_str += f"{class_idx} {' '.join(bbox)}\n"
            
            with open(output_ann_path, "w") as ann_writer:
                ann_writer.write(anns_str)
        
        with open(os.path.join(
            yolo_dataset, k, "annotations", "classes.txt"
        ), "w") as classes_writer:
            classes_writer.write("\n".join(all_classes))
    


            
        
        

            


# for dir_path in dirs:
#     full_dir_path = cwd + '/' + dir_path
#     output_path = full_dir_path +'/yolo/'

#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     image_paths = getImagesInDir(full_dir_path)
#     list_file = open(full_dir_path + '.txt', 'w')

#     for image_path in image_paths:
#         list_file.write(image_path + '\n')
#         convert_annotation(full_dir_path, output_path, image_path)
#     list_file.close()

#     print("Finished processing: " + dir_path)

if __name__ == "__main__":

    parse = argparse.ArgumentParser(
        description="Convert Pascal VOC dataset to YOLO format")
    parse.add_argument(
        "--dataset_dir",
        help="Dataset directory",
        type=str,
        required=True,
    )
    args = parse.parse_args()
    main(args.dataset_dir)