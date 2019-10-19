# Python后端说明文档
基于深度学习的图像处理 2019年python小学期大作业部分

## 0 注意事项

(1)请安装好运行本代码的基本库，见requirment.txt
(2)项目mysite运行时可能存在路径的问题，一些代码有严格的路径要求才能正确实现，如：

> mysite\image\torch_face_recognition\src\get_nets.py
> mysite\image\object_detection_image_classify\od_image_classify.py

Django项目在我的电脑上运行时默认的os路径是mysite（项目本身的root路径），即为当前项目的路径。若存在图片无法读入、项目无法运行，可能是项目的路径设置不对。
(3)无法处理过大的图片，请提交小于2MB的图片，否则有可能程序死机
(4)Tensorflow暂无基于python=3.7的发布，所以本项目无法在3.7的项目上运行。
本机Python信息为
> Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47)
> 
[MSC v.1916 64 bit (AMD64)] on win32 64bit的3.6.8版本，还未再其他环境下测试。

(5)image文件夹以及image下的object_detection_image_classify文件夹不可更改，因为object_detection_image_classify中的代码import需要路径正确，否则无法导入。

## 1.人脸识别

### 1.1 OpenCV中基于Haar特征和级联分类器的人脸检测

```python
def opencv_detect(image_in, caspath=cascadeclassfier_path):
	face_casecade = cv2.CascadeClassifier(caspath)
	img = cv2.imread(image_in, cv2.IMREAD_COLOR)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_casecade.detectMultiScale(gray_img, 1.2, 5)
	#运用opencv自带的cascadeclassifer来进行处理数据的
```

其中caspath为级联分类器的地址，cv2提供了许多个分类器，这里运用的是frontface的default的xml作为我们用的
优点：实现简单
缺点：只能识别正面脸部，且识别正确率较低
python库的要求：OpenCV-python=4.1.1.26 以及OpenCV-python依赖的库 

### 1.2 PyTorch实现人脸检测算法MTCNN 

```python
for s in scales:
	boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
	bounding_boxes.append(boxes)
	#从不同的范围找到可能的人脸box,offsets,和scores
bounding_boxes = [i for i in bounding_boxes if i is not None]
bounding_boxes = np.vstack(bounding_boxes)
#PNET 运用NMS进行再次训练，校准
keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
bounding_boxes = bounding_boxes[keep]
#使用PNET层得到的offsets来训练bounding boxes
bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
#shape [n_boxes, 5]
bounding_boxes = convert_to_square(bounding_boxes)
bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
#PNET层的训练(非极大值抑制NMS校准)，后面还有RNET和ONET的训练以及校准
#'通过三层的训练能够得到较好的box集，即为人脸框。接着运用PIL的ImageDraw画出人脸框, 返回画好的图 
```

优点：识别率很高，且数据处理较快，相对于第一个人脸识别的跟好。
缺点：暂无
Pillow 6.1.0 , numpy 1.17.2 , torch 1.2.0+cpu , 以及其他上面的库所依赖的库 

## 2 图像分类 

### 2.1 基于深度学习库Keras的图像分类
Keras 是为人类而非机器设计的 API。Keras 遵循减少认知困难的最佳实践: 它提供一致且简单的 API，所以极易上手和运用

```python
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

#调用ImageNet模型
model = ResNet50(weights='imagenet')
#导入图片数据进行预测
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
return decode_predictions(preds, top=top_n)[0]
#由于这一部分非常简单，所以我实现了对返回的预测结果进行可视化的操作，以分类结果和分类得到的配对率为坐标返回坐标系的图片
```

优点：实现简单
缺点:暂无，速度相对较慢
Keras 2.2.5, Keras-Applications 1.0.8, Keras-Preprocessing 1.1.0, 以及keras依赖的Tensorflow等库

备注：如果没安装Keras，可进入项目文件夹image下的image_manager.py中隐去使用Keras的部分，不影响测试
Warning: 需要指出的是Keras不是基于最新的Tensorflow库，运行会报一些Warning，不影响测试

### 2.2 基于SSD,mobilenet的深度学习进行图像分类
（object_detection）

```python
#ssd_mobilenet有训练好的模型，github/tensorflow/models中下载调用
MODEL_NAME = BASIC_PATH + 'ssd_mobilenet_v1_coco_2017_11_17'
#frozen_detection_graph的路径，是真正处理的模型，T
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
#标签的加载.这里我们运用mscoco_label_map作为label集，models/object_detection中还含有许多其他的Label集，可更换其文件名调用。
#此次项目我们仅保留了两个pbtxt，如mscoco_complete_label_map
PATH_TO_LABELS = os.path.join(BASIC_PATH + 'object_detection/data', 'mscoco_label_map.pbtxt')
```

加载模型以及训练的部分可查看代码，代码加载模型以及实现分类的代码比较复杂，这里不展示了。
有点：识别率较好，可选择模型训练集范围广
缺点：自带的代码无法直接使用，需要修改许多地方才能完整运行，运行时间较长 
