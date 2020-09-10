# -*- coding:utf-8 -*-
import datetime
import math
import time
from math import *
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import glob
import numpy as np
import os
import cv2
import alphabet

alphabet_str = alphabet.alphabet


# 从文字库中随机选择n个字符

def sto_choice_from_info_str(quantity=7):
    random_str = ""
    chinese = "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新港澳"
    english = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    numbers = "0123456789"
    random_str += random.choice(alphabet_str)
    random_str += random.choice(alphabet_str)


    for i in range(2, quantity):
        if random.random() > 0.5:
            random_str += random.choice(alphabet_str)
        else:
            random_str += random.choice(alphabet_str)

    # print(random_str)
    return random_str


def random_word_color(random_back_color):
    if random_back_color == 0 or random_back_color == 1:
        font_color = (255, 255, 255)
        noise = np.array([random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)])
        font_color = (np.array(font_color) - noise).tolist()

    else:
        font_color = (0, 0, 0)
        noise = np.array([random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)])
        font_color = (np.array(font_color) + noise).tolist()
    return tuple(font_color)


def cut_img(img, x, y):
    """
    函数功能：进行图片裁剪（从中心点出发）
    :param img: 要裁剪的图片
    :param x: 需要裁剪的宽度
    :param y: 需要裁剪的高
    :return: 返回裁剪后的图片
    """
    x_center = random.randint(x // 2, img.size[0] - x // 2)
    y_center = random.randint(y // 2, img.size[1] - y // 2)
    new_x1 = x_center - x//2
    new_y1 = y_center - y//2
    new_x2 = x_center + x//2
    new_y2 = y_center + y//2
    new_img = img.crop((new_x1, new_y1, new_x2, new_y2))
    return new_img


# 生成一张图
def create_an_image(bground_path, random_back_color):
    bground_list = os.listdir(bground_path)
    bground_list.remove("noise")
    noise_path = bground_path + "noise/"
    noise_list = os.listdir(noise_path)
    noise_choice = random.choice(noise_list)
    bground = Image.open(bground_path + bground_list[random_back_color]).convert("RGB")
    noise = Image.open(noise_path + noise_choice).convert("RGB")
    bground = bground.resize((140, 32))
    noise1 = cut_img(noise, 140, 32)
    bground1 = cut_img(bground, 140, 32)
    final_img2 = Image.blend(bground1, noise1, 0.3)
    return final_img2


# 选取作用函数
def random_choice_in_process_func():
    pass


# 模糊函数
def darken_func(image):
    # .SMOOTH
    # .SMOOTH_MORE
    # .GaussianBlur(radius=2 or 1)
    # .MedianFilter(size=3)
    # 随机选取模糊参数

    filter_ = random.choice(
        [
            ImageFilter.BLUR,
            ImageFilter.MedianFilter(size=3),
            ImageFilter.SMOOTH,
            ImageFilter.SMOOTH_MORE]
    )

    image = image.filter(filter_)

    return image


# 随机选取文字贴合起始的坐标 根据背景的尺寸和字体的大小选择
def random_x_y_1(bground_size, font_size):
    width, height = bground_size
    # 为防止文字溢出图片，x，y要预留宽
    x = random.randint(0, max(int(width - font_size * 7), 0))
    y = random.randint(0, int((height - font_size) / 2))
    return x, y


def random_font_size():
    font_size = random.randint(17, 22)
    return font_size


def random_font():
    font_path = './font/Chinese/'

    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)

    return font_path + random_font


def rotate_img(img, angle_range=10):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(angle_range))) + height * fabs(cos(radians(angle_range))))
    widthNew = int(height * fabs(sin(radians(angle_range))) + width * fabs(cos(radians(angle_range))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle_range, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation


def sp_noise(image,prob):
  '''
  添加椒盐噪声
  prob:噪声比例
  '''
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output


def crop_img(img):
    pts1 = np.float32([[0, 0], [140, 0], [0, 32], [140, 32]])
    x1 = random.randint(-3, 10)
    y1 = random.randint(-3, 3)
    x2 = random.randint(-3, 10)
    y2 = random.randint(-3, 3)
    x3 = random.randint(-3, 10)
    y3 = random.randint(-3, 3)
    x4 = random.randint(-3, 10)
    y4 = random.randint(-3, 3)
    pts2 = np.float32([[0 + x1, 0 + y1], [140 - x2, 0 + y2], [0 + x3, 32 - y3], [140 - x4, 32 - y4]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_crop = cv2.warpPerspective(img, M, (140, 32))
    return img_crop

def main(save_path, num):
    random_word = sto_choice_from_info_str()
    # print(random_word)
    # 生成一张背景图片，已经剪裁好，宽高32*280
    random_back_color = random.randint(0, 4)
    # 4 黄色
    # 3 绿色
    # 2 绿色
    # 1 蓝色
    # 0 黑色
    # print(random_back_color)
    raw_image = create_an_image('./background/', random_back_color)

    # 随机选取字体大小
    font_size = random_font_size()
    # print(font_size)
    # 随机选取字体
    font_name = random_font()
    font_color = random_word_color(random_back_color)

    # 随机选取文字贴合的坐�?x,y
    draw_x, draw_y = random_x_y_1(raw_image.size, font_size)
    # 将文本贴到背景图
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(raw_image)
    draw.text((draw_x, draw_y), random_word, fill=font_color, font=font)

    # 随机选取作用函数和数量作用于图片

    if draw_x < 40 and font_size < 15 and random.randint(1, 10) < 10:
        params = [1 - float(random.randint(1, 2)) / 51,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 5)) / 50,
                  float(random.randint(1, 4)) / 1200,
                  0.00005,
                  float(random.randint(1, 4)) / 1150
                  ]
        raw_image = raw_image.transform((140, 32), Image.PERSPECTIVE, params)

    if draw_x < 40 and font_size < 20 and random.randint(1, 10) < 7:
        params = [1 - float(random.randint(1, 2)) / 51,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 4)) / 50,
                  float(random.randint(1, 4)) / 1150,
                  0.00005,
                  float(random.randint(1, 4)) / 1150
                  ]
        raw_image = raw_image.transform((140, 32), Image.PERSPECTIVE, params)

    # raw_image = raw_image.rotate(0.3)
    # 保存文本信息和对应图片名  #with open(save_path[:-1]+'.txt', 'a+', encoding='utf-8') as file:
    img = cv2.cvtColor(np.asarray(raw_image), cv2.COLOR_RGB2BGR)
    if random.random() < 0.4:
        rotate_angle = random.randint(-5, 5)
        img = rotate_img(img, rotate_angle)
    if random.random() < 0.4:
        img = crop_img(img)
    if random.random() < 0.4:
        img = sp_noise(img, 0.01)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if random.random() < 0.4:
        random_p = random.random() + 0.5
        image = image.point(lambda p: p * random_p)
    if random.random() < 0.7:
        random_Gauss = random.random() + 0.5
        image = image.filter(ImageFilter.GaussianBlur(radius=random_Gauss))
    if random.randint(1, 10) < 5:
        image = darken_func(image)
    # file.write('train_set/' + str(num) + '.png ' + random_word + '\n')
    image.save(save_path + '/img/img_{}.jpg'.format(num))
    savename = save_path + '/txt/img_{}.txt'.format(num)
    savef = open(savename, 'w', encoding="utf-8")
    savef.write(random_word + "\n" + str(random_back_color))
    savef.close()


if __name__ == '__main__':

    # 图片标签
    total = 100000
    prev_time = time.time()
    for num in range(0, total):
        # print(num)
        main('./train_set', num)
        time_left = datetime.timedelta(seconds=(total - num) * (time.time() - prev_time) / (num + 1))
        if num % 1000 == 0:
            print('[%d/%d], [time_left: %s]' % (num, total, time_left))
