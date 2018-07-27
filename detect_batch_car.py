from imageai.Detection.Detector import ObjectDetection
import os
import time
import argparse
import cv2
import logging
import shutil
import matplotlib.image as pltimage
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

def _resize_short_side(img, short_side=500):
    '''
    原图宽高为(w,h), 短边大于short_side的，长宽同比例缩放短边至short_side
    '''
    h, w = img.shape[:2]
    if h >= w and w > short_side:
        new_h = int(h * short_side / w + 0.5)
        img = cv2.resize(img, (short_side, new_h))
    elif w > h and h > short_side:
        new_w = int(w * short_side / h + 0.5)
        img = cv2.resize(img, (new_w, short_side))
    return img

def init_model(args):
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
    detector.setModelPath(args.model_path)
    detector.loadModel(detection_speed=args.speed)
    return detector

def test_image(detector, input, output, prob=30):
    t0 = time.time()
    detections = detector.detectObjectsFromImage(input_image=input, output_image_path=output, minimum_percentage_probability=prob)
    print('time :', time.time() - t0)
    avg_time = 0
    for i in range(10):
        t0 = time.time()
        detections = detector.detectObjectsFromImage(input_image=input, output_image_path=output, minimum_percentage_probability=prob)
        t = time.time() - t0
        avg_time += t
        print('time :', t)
    print('average time : ', avg_time / 10)
    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")


def test_dir(detector, in_dir, out_dir, prob=20, short_size=300, process_id=0,process_num=10):
    # 只检测车
    counter = 0
    custom_objects = detector.CustomObjects(car=True)
    # if not os.path.isdir(out_dir):
    #     os.makedirs(out_dir)
    for dirname in os.listdir(in_dir):
        if dirname[0] == '.':
            continue
        in_path1 = os.path.join(in_dir, dirname)
        out_path1 = os.path.join(out_dir, dirname)
        if not os.path.isdir(out_path1):
            os.makedirs(out_path1)
        for filename in os.listdir(in_path1):
            if filename[0] == '.':
                continue
            in_path2 = os.path.join(in_path1, filename)
            out_path2 = os.path.join(out_path1, filename)
            if os.path.isfile(out_path2) and os.path.getsize(out_path2) > 10240:
                logging.info(out_path2 + ' file exists !')
                continue
            print(counter)
            print(in_path2)
            print(out_path2)
            t0 = time.time()
            img_obj, name, pp, point = detector.detectCustomObjectsFromImage(custom_objects, in_path2, output_type='array', minimum_percentage_probability=prob)
            t1 = time.time() - t0
            if img_obj is None:
                logging.info(in_path2 + ' get no car')
                # shutil.copy2(in_path2, out_path2)
            else:
                logging.info(in_path2 + ' %s %6.3f %s  time:%6.3f' % (name, pp, str(point), t1))
                # pltimage.imsave(out_path2, img_obj)
                img_obj = _resize_short_side(img_obj, short_size)
                cv2.imwrite(out_path2, img_obj)
            # break
            counter += 1
        # break





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='yolo3')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--input', default='d:/data_autohome/autoimg1') # ./images/1.jpg
    parser.add_argument('--output', default='d:/data_autohome/autoimg1_det_yolo3') # result.jpg
    parser.add_argument('--speed', default='normal', help='“normal”(default), “fast”, “faster” , “fastest” and “flash”')
    parser.add_argument('--prob', default=20, type=float)
    parser.add_argument('--short_size', default=300, type=int)
    parser.add_argument('--process_id', default=0, type=int)
    parser.add_argument('--process_num', default=10, type=int)
    args = parser.parse_args()
    # args = choose_model(args)
    return args


def main(args):
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logging.basicConfig(
        filename='%s_%d-%d_%s_detect_car.log' % (str_time, args.process_num, args.process_id, args.model_name),
        level=logging.INFO,
        format='[%(levelname)s] (%(process)d) (%(threadName)-10s) %(message)s',
    )

    detector = init_model(args)

    if os.path.isfile(args.input):
        test_image(detector, args.input, args.output)
    elif os.path.isdir(args.input):
        test_dir(detector, args.input, args.output, args.prob, args.short_size, args.process_id, args.process_num)
    else:
        exit(' wrong input')

if __name__ == '__main__':
    main(get_args())

