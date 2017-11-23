import cv2
import numpy
import os


def video2img():
    for i in range(1, 31):
        pref_i = '%04d' % i
        os.system('ffmpeg -i ../data/train/train/' + str(
            i) + '.mp4 -r 10 -f image2 ../data/train/images/' + pref_i + '_%05d.jpg')


def video_slice():
    for i in range(1, 31):
        for j in range(10):
            os.system('ffmpeg -i ../data/train/train/%d.mp4 -ss %d -t %d -acodec copy -vcodec copy ../data/train/s_train/%d.mp4' % (
            i, j * 6, (j + 1) * 6, i * 100 + j))


def s_video2img():
    for i in range(1, 31):
        for j in range(10):
            s_i = i * 100 + j
            pref_i = '%04d' % s_i
            os.system('ffmpeg -i ../data/train/s_train/' + str(
                s_i) + '.mp4 -r 10 -f image2 ../data/train/s_images/' + pref_i + '_%05d.jpg')

if __name__ == '__main__':
    # video_slice()
    s_video2img()