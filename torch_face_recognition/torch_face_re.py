import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import os

from image.torch_face_recognition.src.get_nets import PNet, RNet, ONet
from image.torch_face_recognition.src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from image.torch_face_recognition.src.first_stage import run_first_stage
from image.torch_face_recognition.src.visualization_utils import show_bboxes


def face_get(image):
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()
    # if this value is too low the algorithm will use a lot of memory
    min_face_size = 15.0
    # for probabilities
    thresholds = [0.6, 0.7, 0.8]
    # for NMS
    nms_thresholds = [0.7, 0.7, 0.7]
    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)
    # scales for scaling the image
    scales = []
    width, height = image.size
    min_length = min(height, width)
    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size / min_face_size  # 12/15
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    # P-Net
    bounding_boxes = []
    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)
    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    # PNET NMS + calibration
    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]
    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])  # shape [n_boxes, 5]
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # R-NET
    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    with torch.no_grad():
        img_boxes = Variable(torch.FloatTensor(img_boxes))
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]
    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    # RNET NMS + calibration
    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # O-Net
    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    with torch.no_grad():
        img_boxes = Variable(torch.FloatTensor(img_boxes))
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    # NMS + calibration
    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return show_bboxes(image, bounding_boxes, landmarks)


def torch_face_recognition(image_in, image_out=''):
    """
    filepath is the path of image, ensuer it's safe to PIL.open
    :return a path that store the image, so we should:
    """
    # import os
    # print(os.getcwd(), image_in)
    if os.path.isfile(image_in) is False:
        print('Error:torch image path not exist!', image_in)
        return ''
    image = Image.open(image_in)
    img = face_get(image)
    # img.show()
    try:
        if image_out == '':
            if os.path.isfile(image_in):
                image_out = image_in[:image_in.rfind('.')] + '_tfc_' + image_in[image_in.rfind('.'):]
                img.save(image_out)
                return image_out
        else:
            img.save(image_out)
            return image_out
    except Exception as e:
        print("Please ensure your path is available!")

# torch_face_recognition(r'..\..\userdata\123\83\liu1.jpg')
