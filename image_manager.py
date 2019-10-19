"""
    在导入ofr时，请确保能正确给出当前电脑分类器文件的路径
    也就是安装了OpenCv后，找到Python路径下的Lib/site-packages
    进入cv2，找到data/haarcascade_frontalface_default.xml
"""
import os

print(os.getcwd())


def face_recognition(image_in, image_out=''):
    ofr_img_src = ''
    tfr_img_src = ''
    from image import opencv_face_recognition as ofr
    from image.torch_face_recognition import torch_face_re as tfr
    ofr_img_src = ofr.opencv_face_recognition(image_in, image_out)
    tfr_img_src = tfr.torch_face_recognition(image_in, image_out)
    print(ofr_img_src, tfr_img_src)
    return ofr_img_src, tfr_img_src


# face_recognition(r'userdata\123\83\liu1.jpg')

def image_classify(image_in, image_out=''):
    kic_img_src = ''
    kic_result = []
    oic_img_src = ''
    oic_result = []
    # 如果未安装相关库可以直接隐去下面的任意一部分代码，不会报错
    from image import keras_image_classify as kic
    kic_img_src, kic_result = kic.keras_image_classify(image_in, image_out)
    print(kic_img_src, kic_result)

    from image.object_detection_image_classify import od_image_classify as oic
    oic_img_src, oic_result = oic.object_detection(image_in, image_out)
    print(oic_img_src, oic_result)
    return kic_img_src, oic_img_src, str(kic_result), str(oic_result)
#
# image_classify(r'userdata\test\test\bear.jpg')
