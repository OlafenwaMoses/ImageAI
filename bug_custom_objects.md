```
Traceback (most recent call last):
  File "D:\github\ImageAI\imageai\Detection\Detector.py", line 682, in detectCustomObjectsFromImage
    display_percentage_probability, display_object_name)
  File "D:\github\ImageAI\imageai\Detection\Detector.py", line 977, in detectCustomObjectsFromImage_yolo_simplify
    if (custom_objects[predicted_class] == "invalid"):
KeyError: 'donut'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File ".\detect_batch_car.py", line 168, in <module>
    main(get_args())
  File ".\detect_batch_car.py", line 163, in main
    test_dir(detector, args.input, args.output, args.prob, args.short_size, args.process_id, args.process_num)
  File ".\detect_batch_car.py", line 111, in test_dir
    img, objs = detector.detectCustomObjectsFromImage(custom_objects, in_path2, output_type='array', minimum_percentage_probability=prob)
  File "D:\github\ImageAI\imageai\Detection\Detector.py", line 685, in detectCustomObjectsFromImage
    "Ensure you specified correct input image, input type, output type and/or output image path ")
ValueError: Ensure you specified correct input image, input type, output type and/or output image path
(tensorflow_py3.5) PS D:\github\ImageAI>
```




```
d:/data_autohome/autoimg1\32.172.2839\1004284.1.100092776.jpg
d:/data_autohome/autoimg1_det_yolo3\32.172.2839\1004284.1.100092776.jpg
Traceback (most recent call last):
  File "D:\github\ImageAI\imageai\Detection\Detector.py", line 682, in detectCustomObjectsFromImage
    display_percentage_probability, display_object_name)
  File "D:\github\ImageAI\imageai\Detection\Detector.py", line 925, in detectCustomObjectsFromImage_yolo_simplify
    image = Image.open(input_image)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\PIL\Image.py", line 2622, in open
    % (filename if filename else fp))
OSError: cannot identify image file 'd:/data_autohome/autoimg1\\32.172.2839\\1004284.1.100092776.jpg'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File ".\detect_batch_car.py", line 168, in <module>
    main(get_args())
  File ".\detect_batch_car.py", line 163, in main
    test_dir(detector, args.input, args.output, args.prob, args.short_size, args.process_id, args.process_num)
  File ".\detect_batch_car.py", line 111, in test_dir
    img, objs = detector.detectCustomObjectsFromImage(custom_objects, in_path2, output_type='array', minimum_percentage_probability=prob)
  File "D:\github\ImageAI\imageai\Detection\Detector.py", line 685, in detectCustomObjectsFromImage
    "Ensure you specified correct input image, input type, output type and/or output image path ")
ValueError: Ensure you specified correct input image, input type, output type and/or output image path
```