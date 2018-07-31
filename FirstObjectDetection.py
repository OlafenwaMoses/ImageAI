from imageai.Detection import ObjectDetection
import os
import time
import argparse
'''
python FirstObjectDetection.py \
    --model_name yolo3 \
    --model_path  'D:/data_public/ImageAI/yolo.h5' \
    --input ./images/1.jpg \
    --output result.jpg

retinanet : 使用GPU要1秒
yolo3 : 使用GPU 650毫秒

分别测试不同speed下的速度，依次是 “normal”(default), “fast”, “faster” , “fastest” and “flash”
model           “normal”(default),      “fast”,     “faster” ,      “fastest”,   “flash”
retinanet       1.02                    0.678       0.630            0.592           0.590   
yolo3           0.65                    0.63        0.61            0.619           0.619

注意：之前在mac用py2存在问题，画的框几乎无限多。而py3执行则正常
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='retinanet')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--input', default='./images/1.jpg')
    parser.add_argument('--output', default='result.jpg')
    parser.add_argument('--speed', default='normal', help='“normal”(default), “fast”, “faster” , “fastest” and “flash”')
    return parser.parse_args()

def main(args):
    detector = ObjectDetection()
    if args.model_name == 'retinanet':
        detector.setModelTypeAsRetinaNet()
        if args.model_path is None:
            args.model_path = 'D:/data_public/ImageAI/retinanet/resnet50_coco_best_v2.0.1.h5'
    elif args.model_name == 'yolo3':
        detector.setModelTypeAsYOLOv3()
        if args.model_path is None:
            args.model_path = 'D:/data_public/ImageAI/yolo.h5'
    elif args.model_name == 'yolo3-tiny':
        detector.setModelTypeAsTinyYOLOv3()
        if args.model_path is None:
            args.model_path = 'D:/data_public/ImageAI/yolo.h5'
    else:
        exit('unknown model_name !!!')
    # 只检测车
    custom_objects = detector.CustomObjects(car=True)
    detector.setModelPath(args.model_path)
    detector.loadModel(detection_speed=args.speed)

    t0 = time.time()
    detections = detector.detectObjectsFromImage(input_image=args.input, output_image_path=args.output, minimum_percentage_probability=30)
    print('time :', time.time() - t0)
    for i in range(10):
        t0 = time.time()
        detections = detector.detectObjectsFromImage(input_image=args.input, output_image_path=args.output, minimum_percentage_probability=30)
        print('time :', time.time() - t0)
    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

if __name__ == '__main__':
    main(get_args())

