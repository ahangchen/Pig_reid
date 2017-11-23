import operator

from keras.preprocessing import image

import cuda_util
import os
from os import remove, path

import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from utils.file_helper import write


def test_data_prepare(data_list_path, train_dir_path):
    test_imgs = list()
    test_ids = list()
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img_name = line
            test_id = img_name.split('.')[0]
            img = image.load_img(os.path.join(train_dir_path, img_name), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img).reshape(224, 224, 3)
            test_imgs.append(img)
            test_ids.append(test_id)
    return test_imgs, test_ids


def pig_generator(test_imgs, batch_size):
    cur_batch = -1
    while True:
        cur_batch += 1
        yield np.array(test_imgs[cur_batch * batch_size : (cur_batch + 1) * batch_size])


if __name__ == '__main__':
    batch_size = 64
    model = load_model('pig_softmax_pretrain.h5')
    # model = load_model('pig_pair_pretrain.h5')
    # single_model = Model(inputs=model.layers[0].input, outputs=[model.layers[9].output])
    # model = single_model

    test_imgs, test_ids = test_data_prepare('../dataset/pig_test.list', '../data/test_a/test_A')
    y = model.predict_generator(pig_generator(test_imgs, batch_size), 3000/batch_size + 1, use_multiprocessing=True)
    # ys = np.sum(y,axis=1)
    # y_i = ys > 0
    y/=1.001
    predict_path = 'predict.csv'
    if path.exists(predict_path):
        remove(predict_path)

    np.savetxt(predict_path, y, fmt='%6f', delimiter='\t')
    # y = np.genfromtxt('predict.csv', delimiter='\t')
    y = y.reshape(-1)
    ids = np.array(test_ids)
    cls = np.arange(1, 31)
    ids = ids.repeat(30)
    cls = np.tile(cls, 3000)

    rst = np.append(ids, cls)
    rst = np.append(rst, y).reshape(3, 90000)
    rst = np.rot90(rst)
    rst_str = ''
    for line in rst:
        rst_str += '%d,%d,%5f\n' % (int(line[0]), int(line[1]), float(line[2]))

    rst_path = 'pig_rst.csv'
    if path.exists(rst_path):
        remove(rst_path)
    write(rst_path, rst_str)
    # np.savetxt('pig_rst.csv', rst, fmt='%d,%d,%6f', delimiter=',')


