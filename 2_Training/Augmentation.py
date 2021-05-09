import numpy as np
import imgaug.augmenters as iaa

from PIL import Image
from Train_Utils import data_generator_wrapper


def __sometimes(aug):
    return iaa.Sometimes(0.5, aug)

__seq = iaa.Sequential([
    # gaussian blur
    __sometimes(iaa.GaussianBlur(sigma=(0, 3))),

    # average blur
    __sometimes(iaa.AverageBlur(k=(2, 7))),

    # median blur
    __sometimes(iaa.MedianBlur(k=(3, 11))),

    # Sharpen each image, overlay the result with the original
    __sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),

    # Same as sharpen, but for an embossing effect.
    __sometimes(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),

    # Strengthen or weaken the contrast in each image.
    __sometimes(iaa.LinearContrast((0.75, 1.5))),

    # Search in some images either for all edges or for
    # directed edges. These edges are then marked in a black
    # and white image and overlayed with the original image
    # using an alpha of 0 to 0.7.
    __sometimes(iaa.OneOf([
        iaa.EdgeDetect(alpha=(0, 0.7)),
        iaa.DirectedEdgeDetect(
            alpha=(0, 0.7), direction=(0.0, 1.0)
        ),
    ])),

    # Add gaussian noise.
    __sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),

    # Invert each image's channel with 5% probability.
    # This sets each pixel value v to 255-v.
    iaa.Invert(0.05, per_channel=True), # invert color channels

    iaa.Add((-10, 10), per_channel=0.5),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.4, 1.6), per_channel=0.2),

    iaa.Grayscale(alpha=(0.0, 1.0)),

    # Convert some images into their superpixel representation,
    # sample between 20 and 200 superpixels per image, but do
    # not replace all superpixels with their average, only
    # some of them (p_replace).
    # _sometimes(
    #     iaa.Superpixels(
    #         p_replace=(0, 1.0),
    #         n_segments=(20, 200)
    #     )
    # ),
], random_order=True)  # apply augmenters in random order

def augmented_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    data = data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes)

    files_saved = False

    for datum in data:

        ### Augment images BEGIN ###

        shape = datum[0][0].shape
        img_list = [(img * 255).astype(np.uint8) for img in datum[0][0]]
        img_list = __seq(images=img_list)
        datum[0][0] = np.array(img_list, dtype=np.float32).reshape(shape) / 255.0

        ### Augment images END ###

        #save images in the root if not saved already (to showcase augmentation)
        if not files_saved:
            files_saved = True
            for j, img in enumerate(img_list):

                img = Image.fromarray(img)
                img.save(f"{j+1}.jpeg")

        yield datum