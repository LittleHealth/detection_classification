import numpy as np
import os
# import sys
# print(os.getcwd(), sys.path)
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image

from image.object_detection_image_classify.object_detection.utils import ops as utils_ops

from image.object_detection_image_classify.object_detection.utils import label_map_util

from image.object_detection_image_classify.object_detection.utils import visualization_utils as vis_util

BASIC_PATH = 'image/object_detection_image_classify/'

# What model to download.
MODEL_NAME = BASIC_PATH + 'ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(BASIC_PATH + 'object_detection/data', 'mscoco_label_map.pbtxt')


def detect_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            # print('all_tensor_names', all_tensor_names, ops)
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image
                # size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    # print("output_dict:", output_dict)
    return output_dict


def ssd_mobile_detection(image_in, image_out, detection_graph):
    image = Image.open(image_in)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
    scores = output_dict['detection_scores']
    classes = output_dict['detection_classes']
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    min_score_thresh = 0.6  # 匹配率低于0.6的筛选掉
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        classes,
        scores,
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=True,
        line_thickness=8)
    # print(
    #     'output_dict[detection_boxes]',output_dict['detection_boxes'],
    #     'CLASSES',output_dict['detection_classes'],
    #     'SCORES', output_dict['detection_scores'],
    #     category_index,)
    result = []

    for index in range(len(scores)):
        if scores[index] > min_score_thresh:
            class_index = classes[index]
            name = category_index[class_index]['name']
            result.append([name, scores[index]])
            # print(index, class_index, name, result)
    # print("odimage", result, str(result))

    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.savefig(image_out)
    return result


def object_detection(image_in, image_out=''):
    # print('object_detection', image_in)
    detection_graph = detect_graph()
    # print('over detection')
    if os.path.isfile(image_in) is False:
        print('Error: image path not exist!')
        return image_out, None
    if image_out == '':
        if os.path.isfile(image_in):
            image_out = image_in[:image_in.rfind('.')] + '_oic_' + image_in[image_in.rfind('.'):]
            # print('image_out',image_out)
        else:
            print('Error:Can not find the image')
            return '', None
    result = ssd_mobile_detection(image_in, image_out, detection_graph)
    return image_out, result


# print(os.getcwd())
object_detection(r'..\..\userdata\test\test\dog.jpg')
