#!/usr/bin/python
from data.UMDdatabase import UMDDatabase
from PIL import Image, ImageDraw
import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from random import shuffle, randint
from os.path import join, isdir
from os import makedirs
import numpy as np
import tensorflow as tf
from tqdm import tqdm

label_dic = {"negative": [0, 0, 0, 0],
             "left_eye": [1, 0, 0, 0],
             "right_eye": [0, 1, 0, 0],
             "nose": [0, 0, 1, 0],
             "mouth": [0, 0, 0, 1],
}

def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def to_example(image, label):
    image_buffer = np.array(image).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(label)
    }))
    return example


def intersection(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1+w1, x2+w2)
    bot = min(y1+h1, y2+h2)
    width = max(0, right-left)
    height = max(0, bot-top)
    return width*height


def generate_crops(x, y , w, n_elem):
    crops = []
    for a in range(int(-w/4), int(w/4)+1):
        for b in range(int(-w/6), int(w/6)+1):
            crops.append([x-w/2+a, y-w/3+b, x+w/2+a, y+w/3+b])
    shuffle(crops)
    return crops[:n_elem]


def generate_negative_crops(width, height, bbox, n_elem):
    crops = []
    while len(crops) < n_elem:
        try:
            w = randint(3, width)
            h = int(w / 1.5)
            x = randint(0, width - w)
            y = randint(0, height - h)
            if intersection(bbox, [x, y, w, h]) == 0.0:
                crops.append([x, y, x+w, y+h])
        except ValueError:
            pass
    return crops


def generate_data(annotation, tf_writer, visibility_threshold=0.5, count=[5, 10]):
    im = Image.open(annotation.path).convert("L")
    w = annotation.bbox[2] / 3.5
    crops = {}
    x, y, vis = annotation.left_eye
    if vis > visibility_threshold:
        crops["left_eye"] = generate_crops(x, y, w, count[0])
    x, y, vis = annotation.right_eye
    if vis > visibility_threshold:
        crops["right_eye"] = generate_crops(x, y, w, count[0])
    x, y, vis = annotation.nose
    if vis > visibility_threshold:
        crops["nose"] = generate_crops(x, y, w, count[0])
    x1, y1, vis1 = annotation.left_mouth
    x2, y2, vis2 = annotation.right_mouth
    if vis1 > visibility_threshold and vis2 > visibility_threshold:
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        w = math.sqrt((x2-x1)**2 + (y2-y1)**2) * 1.3
        crops["mouth"] = generate_crops(x, y, w, count[0])
    crops["negative"] = generate_negative_crops(im.width, im.height, annotation.bbox, count[1])
    for key in crops:
        for crop in crops[key]:
            cropped_im = im.crop(crop).resize((9, 6), resample=Image.BILINEAR)
            label = label_dic[key]
            example = to_example(cropped_im, label)
            tf_writer[key].write(example.SerializeToString())


def main(umd_path, output_dir):
    db = UMDDatabase(umd_path)

    tf_writer = {t: tf.python_io.TFRecordWriter(join(output_dir, t + ".tfrecord")) for t in label_dic}

    for annotation in tqdm(db.faces(), total=db.size):
        generate_data(annotation, tf_writer)

    for t in tf_writer:
        tf_writer[t].close()


if __name__ == "__main__":
    parser = ArgumentParser(description="This scripts generates negatives, "
                                        "positives and partial positives faces",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            epilog="")
    parser.add_argument("-u", "--umd", required=True, help="UMD Database path")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    main(args.umd, args.output_dir)
