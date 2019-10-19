import cv2
import os

cascadeclassfier_path = r'image/haarcascade_frontalface_default.xml'


def opencv_detect(image_in, caspath=cascadeclassfier_path):
    face_casecade = cv2.CascadeClassifier(caspath)

    img = cv2.imread(image_in, cv2.IMREAD_COLOR)
    # print(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray_img)
    faces = face_casecade.detectMultiScale(gray_img, 1.2, 5)  # 默认参数为1.1和3
    # print(faces)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.namedWindow('Face_Detected')
    # cv2.imshow('Face_Detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def opencv_face_recognition(image_in, image_out=''):
    # image_in= r'C:\code\python\myopencv\timg.jpg'
    if os.path.isfile(image_in) is False:
        print('Error: opencv image path not exist!', image_in)
        return ''
    img = opencv_detect(image_in)
    try:
        if image_out == '':
            if os.path.isfile(image_in):
                image_out = image_in[:image_in.rfind('.')] + '_ofc_' + image_in[image_in.rfind('.'):]
                cv2.imwrite(image_out, img)
                return image_out
        else:
            cv2.imwrite(image_out, img)
            return image_out
    except Exception as e:
        print("Please ensure your path is available!")

# opencv_face_recognition(r'..\userdata\123\87\liu1.jpg')
