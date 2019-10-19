import os
# print("keras:", os.getcwd())
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')
target_size = (224, 224)


def predict(model, img, target_size, top_n=3):
    """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
    if img.size != target_size:
        img = img.resize(target_size)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        return decode_predictions(preds, top=top_n)[0]


def plot_preds(image, preds, image_out):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
    """
    # print(preds)
    plt.imshow(image)
    plt.axis('off')

    plt.figure()
    order = list(reversed(range(len(preds))))
    bar_preds = [pr[2] for pr in preds]
    labels = (pr[1] for pr in preds)
    # print(order, bar_preds, labels)
    plt.barh(order, bar_preds, alpha=0.5)
    plt.yticks(order, labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    # plt.show()
    plt.savefig(image_out)


def keras_image_classify(image_in, image_out=''):
    # print('keras_image_classify')
    if os.path.isfile(image_in) is False:
        print('Error: image path not exist!')
        return image_out, None
    image = Image.open(image_in)
    preds = predict(model, image, target_size)
    result = []
    for pr in preds:
        result.append([pr[1], pr[2]])
    # print(result)
    try:
        if image_out == '':
            if os.path.isfile(image_in):
                image_out = image_in[:image_in.rfind('.')] + '_kic_' + image_in[image_in.rfind('.'):]
                plot_preds(image, preds, image_out)
                return image_out, result
        else:
            plot_preds(image, preds, image_out)
            return image_out, result
    except Exception as e:
        print("Please ensure your path is available!")

# keras_image_classify(r'..\userdata\1\63\office2.jpg')
