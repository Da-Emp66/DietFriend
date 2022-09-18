import argparse
import os
from os import listdir
import time
import numpy as np
import easyocr
import glob
import datetime
import sys
import cv2
from cv2 import connectedComponentsWithStats
import psycopg2 as psycopg2
import pytesseract
import torch
import torchvision
import torchaudio
import transformers
import PIL.Image
import shutil

import kivy
from kivy.uix.popup import Popup
from kivymd.uix.button import MDFillRoundFlatIconButton, MDFloatingActionButton
from numpy.random import random
from six import BytesIO
from autocorrect import Speller
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.config import Config
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivy.uix.floatlayout import FloatLayout
import matplotlib as plt
import matplotlib.pyplot
import scipy.ndimage as inter
from kivy.garden import GardenImporter
from kivy_garden.graph import Graph, LinePlot
from kivy.clock import Clock
from kivy.base import runTouchApp
from kivy.factory import Factory
import requests
import io
from io import BytesIO
from kivy.config import Config
from kivy.cache import Cache
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivymd.icon_definitions import md_icons
from kivymd.uix.list import OneLineIconListItem, IconLeftWidget
from kivymd.theming import ThemeManager
# from kivymd.toast import toast
from kivy.properties import ObjectProperty
from kivy_garden.mapview import MapView, MapMarker, MapMarkerPopup
from kivy.uix.image import Image
from kivymd.uix.menu import MDDropdownMenu
from psycopg2 import sql
from configparser import ConfigParser
# ########## Downgrade to Python 3.8 if you want keras to work
# from tensorflow.keras.preprocessing import image as tensorimage
# ##########

Config.set('graphics', 'resizable', 0)


class Day:
    def __init__(self, date_time, totalsfoodlist, usedfoodsincount, rejectsfoodlist_basedonfoodattributeindex,
                 session_id):
        self.date_time = date_time
        self.totalsfoodlist = totalsfoodlist
        self.usedfoodsincount = usedfoodsincount
        self.rejectsfoodlist_basedonfoodattributeindex = rejectsfoodlist_basedonfoodattributeindex
        self.session_id = session_id


class Food:
    def __init__(self, food_name, food_datetime, serving, calories, total_fat, saturated_fat, trans_fat, cholesterol,
                 sodium, total_carb, fiber, total_sugars, added_sugars, protein, calcium, iron, potassium, vitamin_a,
                 vitamin_b, vitamin_c, vitamin_d):
        self.food_name = food_name
        self.food_datetime = food_datetime
        self.serving = serving
        self.calories = calories
        self.total_fat = total_fat
        self.saturated_fat = saturated_fat
        self.trans_fat = trans_fat
        self.cholesterol = cholesterol
        self.sodium = sodium
        self.total_carb = total_carb
        self.fiber = fiber
        self.total_sugars = total_sugars
        self.added_sugars = added_sugars
        self.protein = protein
        self.calcium = calcium
        self.iron = iron
        self.potassium = potassium
        self.vitamin_a = vitamin_a
        self.vitamin_b = vitamin_b
        self.vitamin_c = vitamin_c
        self.vitamin_d = vitamin_d


def adaptive_binarization(img):
    # Whole Image Binarization
    # ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    # Partial Image (Adaptive) Thresholding
    """ UNCOMMENT AND REMOVE pass FOR FULL FUNCTIONALITY
    img = tensorimage.img_to_array(img, dtype='uint8')
    imgf = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return imgf
    """
    pass


def second_binarization(img_name):
    # # #
    # input_file = sys.argv[1]
    # img = PIL.Image.open(input_file)
    # OR
    input_file = img_name
    img = PIL.Image.open(input_file)
    # # #
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    matplotlib.pyplot.imshow(bin_img, cmap='gray')
    matplotlib.pyplot.savefig(img_name)
    return bin_img


def skew_correct(img_name):
    bin_img = second_binarization(img_name)

    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    delta = 1
    limit = 5
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle: {}'.format(best_angle))

    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    imgr = PIL.Image.fromarray((255 * data).astype("uint8")).convert("RGB")
    imgr.save(img_name)
    return imgr


def preprocess_image(img_name):
    global user
    cwd = os.getcwd()
    os.chdir(r"C:\Users\asher\PycharmProjects\DietFriend\DietFriend_Pictures" + user)
    imgf = cv2.imread(img_name)

    print("Step 1: Binarization")
    # Step 1: Binarization
    # imgx = adaptive_binarization(imgf)
    # imgx.save(img_name)

    print("Step 2: Skew Correction")
    # Step 2: Skew Correction
    # # # #
    # imgx = second_binarization(img_name)
    # # # #
    # imgx = skew_correct(img_name)

    print("Step 3: Noise Removal")
    # Step 3: Noise Removal
    dst = cv2.fastNlMeansDenoisingColored(imgf, None, 10, 10, 7, 15)
    # matplotlib.pyplot.subplot(121),
    # matplotlib.pyplot.imshow(imgf)
    # matplotlib.pyplot.subplot(122),
    # matplotlib.pyplot.imshow(dst)
    # matplotlib.pyplot.show()

    print("Step 4: Thinning and Skeletonization")
    # Step 4: Thinning and Skeletonization
    # ###################### imgn = cv2.imread(imgx, 0)
    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(dst, kernel, iterations=1)
    # imgar = PIL.Image.fromarray(erosion)
    # imgar.save(img_name)

    print("x-here-x")
    os.chdir(cwd)
    print("chdir")


def recognize_text(img_path_for_readtext, img_path):
    preprocess_image(img_path)
    print("Pre-Process Complete!")
    reader = easyocr.Reader(lang_list=['en'], gpu=False)
    return reader.readtext(image=img_path_for_readtext)


def onlystrings(read):
    newread = []
    for x in read:
        for subx in x:
            if isinstance(subx, str):
                newread.append(subx)
    return newread


def onlystringstostring(onlystring):
    stringer = ''
    for p in onlystring:
        stringer += p
        stringer += ' '
    return stringer


def autocorrect(for_ltext):
    """AUTO-CORRECT"""
    spell = Speller()
    return spell(for_ltext)


def isin(im, lis):
    for elem in lis:
        if elem == im:
            return True
    return False


def checker(curitm, rcrd, foodattrlisttp):
    """Returns True if element (which is not curitm) is found in rcrd"""
    for attributec in foodattrlisttp:
        if not isin(curitm, attributec):
            for possiblec in attributec:
                anlys = rcrd.find(possiblec)
                if rcrd.find(possiblec) != -1:
                    tplt = (True, anlys)
                    return tplt
    tplf = (False, -1)
    return tplf


def finditem(ptext, item, foodattrlistp):
    pos = ptext.find(item)
    if pos == -1:
        return '???'
    record = ''
    charpos = pos
    curval = (False, -1)
    while (((ptext[charpos - 1] != 'g' and (ptext[charpos - 2] != 'n' and ptext[charpos - 2] != 'u')) or (ptext[charpos - 1] !=
            '9' and (ptext[charpos - 2] != 'n' and ptext[charpos - 2] != 'u')) or ptext[charpos - 1] != '%') and not curval[0]) and (charpos <= len(ptext) - 2):
        record += ptext[charpos]
        print(record)
        curval = checker(curitm=item, rcrd=record, foodattrlisttp=foodattrlistp)
        print(curval)
        if curval[0] or ((ptext[charpos - 1] == 'g' and (ptext[charpos - 2] != 'n' and ptext[charpos - 2] != 'u')) or (ptext[charpos - 1] ==
                         '9' and (ptext[charpos - 2] != 'n' and ptext[charpos - 2] != 'u')) or ptext[charpos - 1] == '%') or (ptext[charpos] == ' ' and (ptext[charpos - 1] == '0' or
                                                                                          ptext[charpos - 1] == '1' or
                                                                                          ptext[charpos - 1] == '2' or
                                                                                          ptext[charpos - 1] == '3' or
                                                                                          ptext[charpos - 1] == '4' or
                                                                                          ptext[charpos - 1] == '5' or
                                                                                          ptext[charpos - 1] == '6' or
                                                                                          ptext[charpos - 1] == '7' or
                                                                                          ptext[charpos - 1] == '8' or
                                                                                          ptext[charpos - 1] == '9')) \
                and (charpos <= len(ptext)):
            if curval[1] != -1:
                # Should always be True
                record = record[0:curval[1]]
            return record
        charpos += 1
    return record


def itemtoserv(recordp):
    exclfound = False
    numstring = ''
    chnum = 0
    while chnum < len(recordp):
        if (recordp[chnum] == '0' or recordp[chnum] == '1' or recordp[chnum] == '2' or recordp[chnum]
                == '3' or recordp[chnum] == '4' or recordp[chnum] == '5' or recordp[chnum] == '6' or recordp[chnum]
                == '7' or recordp[chnum] == '8' or recordp[chnum] == '9' or recordp[chnum] == '.'):
            numstring += recordp[chnum]
        cursearch = recordp[0:chnum]
        if cursearch.find(' ! ') > -1 and not exclfound:
            numstring += '1'
            exclfound = True
        if len(recordp) - 2 >= chnum >= 1:
            if ((recordp[chnum] == 'I' and recordp[chnum + 1] == 'm' and recordp[chnum + 2] == 'g') or
                    (recordp[chnum] == 'I' and recordp[chnum + 1] == 'g') or
                    (recordp[chnum] == 'I' and recordp[chnum + 1] == '%') or
                    (recordp[chnum] == 'I' and (
                            (recordp[chnum - 1] == '0' or recordp[chnum - 1] == '1' or recordp[chnum - 1]
                             == '2' or recordp[chnum - 1] == '3' or recordp[chnum - 1] == '4' or
                             recordp[chnum - 1] == '5' or recordp[chnum - 1] == '6' or recordp[chnum - 1]
                             == '7' or recordp[chnum - 1] == '8' or recordp[chnum - 1] == '9') or
                            (recordp[chnum + 1] == '0' or recordp[chnum + 1] == '1' or recordp[chnum + 1] == '2' or
                             recordp[chnum + 1]
                             == '3' or recordp[chnum + 1] == '4' or recordp[chnum + 1] == '5' or recordp[
                                 chnum + 1] == '6' or
                             recordp[chnum + 1] == '7' or recordp[chnum + 1] == '8' or recordp[chnum + 1] == '9')))):
                recordp = recordp[0:chnum] + '1' + recordp[chnum + 1:len(recordp)]
                numstring += '1'
        if len(recordp) - 2 >= chnum >= 1:
            if ((recordp[chnum] == 'C' and recordp[chnum + 1] == 'm' and recordp[chnum + 2] == 'g') or
                    (recordp[chnum] == 'C' and recordp[chnum + 1] == 'g') or
                    (recordp[chnum] == 'C' and recordp[chnum + 1] == '%') or
                    (recordp[chnum] == 'C' and (
                            (recordp[chnum - 1] == '0' or recordp[chnum - 1] == '1' or recordp[chnum - 1]
                             == '2' or recordp[chnum - 1] == '3' or recordp[chnum - 1] == '4' or
                             recordp[chnum - 1] == '5' or recordp[chnum - 1] == '6' or recordp[chnum - 1]
                             == '7' or recordp[chnum - 1] == '8' or recordp[chnum - 1] == '9') or
                            (recordp[chnum + 1] == '0' or recordp[chnum + 1] == '1' or recordp[chnum + 1] == '2' or
                             recordp[chnum + 1]
                             == '3' or recordp[chnum + 1] == '4' or recordp[chnum + 1] == '5' or recordp[
                                 chnum + 1] == '6' or
                             recordp[chnum + 1] == '7' or recordp[chnum + 1] == '8' or recordp[chnum + 1] == '9')))):
                recordp = recordp[0:chnum] + '0' + recordp[chnum + 1:len(recordp)]
                numstring += '0'
        if len(recordp) - 2 >= chnum >= 1:
            if ((recordp[chnum] == 'c' and recordp[chnum + 1] == 'm' and recordp[chnum + 2] == 'g') or
                    (recordp[chnum] == 'c' and recordp[chnum + 1] == 'g') or
                    (recordp[chnum] == 'c' and recordp[chnum + 1] == '%') or
                    (recordp[chnum] == 'c' and (
                            (recordp[chnum - 1] == '0' or recordp[chnum - 1] == '1' or recordp[chnum - 1]
                             == '2' or recordp[chnum - 1] == '3' or recordp[chnum - 1] == '4' or
                             recordp[chnum - 1] == '5' or recordp[chnum - 1] == '6' or recordp[chnum - 1]
                             == '7' or recordp[chnum - 1] == '8' or recordp[chnum - 1] == '9') or
                            (recordp[chnum + 1] == '0' or recordp[chnum + 1] == '1' or recordp[chnum + 1] == '2' or
                             recordp[chnum + 1]
                             == '3' or recordp[chnum + 1] == '4' or recordp[chnum + 1] == '5' or recordp[
                                 chnum + 1] == '6' or
                             recordp[chnum + 1] == '7' or recordp[chnum + 1] == '8' or recordp[chnum + 1] == '9')))):
                recordp = recordp[0:chnum] + '0' + recordp[chnum + 1:len(recordp)]
                numstring += '0'
        if len(recordp) - 2 >= chnum >= 1:
            if ((recordp[chnum] == 'o' and recordp[chnum + 1] == 'm' and recordp[chnum + 2] == 'g') or
                    (recordp[chnum] == 'o' and recordp[chnum + 1] == 'g') or
                    (recordp[chnum] == 'o' and recordp[chnum + 1] == '%') or
                    (recordp[chnum] == 'o' and (
                            (recordp[chnum - 1] == '0' or recordp[chnum - 1] == '1' or recordp[chnum - 1]
                             == '2' or recordp[chnum - 1] == '3' or recordp[chnum - 1] == '4' or
                             recordp[chnum - 1] == '5' or recordp[chnum - 1] == '6' or recordp[chnum - 1]
                             == '7' or recordp[chnum - 1] == '8' or recordp[chnum - 1] == '9') or
                            (recordp[chnum + 1] == '0' or recordp[chnum + 1] == '1' or recordp[chnum + 1] == '2' or
                             recordp[chnum + 1]
                             == '3' or recordp[chnum + 1] == '4' or recordp[chnum + 1] == '5' or recordp[
                                 chnum + 1] == '6' or
                             recordp[chnum + 1] == '7' or recordp[chnum + 1] == '8' or recordp[chnum + 1] == '9')))):
                recordp = recordp[0:chnum] + '0' + recordp[chnum + 1:len(recordp)]
                numstring += '0'
        if len(recordp) - 2 >= chnum >= 1:
            if ((recordp[chnum] == 'O' and recordp[chnum + 1] == 'm' and recordp[chnum + 2] == 'g') or
                    (recordp[chnum] == 'O' and recordp[chnum + 1] == 'g') or
                    (recordp[chnum] == 'O' and recordp[chnum + 1] == '%') or
                    (recordp[chnum] == 'O' and (
                            (recordp[chnum - 1] == '0' or recordp[chnum - 1] == '1' or recordp[chnum - 1]
                             == '2' or recordp[chnum - 1] == '3' or recordp[chnum - 1] == '4' or
                             recordp[chnum - 1] == '5' or recordp[chnum - 1] == '6' or recordp[chnum - 1]
                             == '7' or recordp[chnum - 1] == '8' or recordp[chnum - 1] == '9') or
                            (recordp[chnum + 1] == '0' or recordp[chnum + 1] == '1' or recordp[chnum + 1] == '2' or
                             recordp[chnum + 1]
                             == '3' or recordp[chnum + 1] == '4' or recordp[chnum + 1] == '5' or recordp[
                                 chnum + 1] == '6' or
                             recordp[chnum + 1] == '7' or recordp[chnum + 1] == '8' or recordp[chnum + 1] == '9')))):
                recordp = recordp[0:chnum] + '0' + recordp[chnum + 1:len(recordp)]
                numstring += '0'
        chnum += 1
    if numstring.find('9') > -1:
        numstring = numprocess(numstring)
    if numstring == '.':
        numstring = ''
    finalstring = afternumprocess(numstring)
    return finalstring


def numprocess(numstringp):
    curchar = 0
    while curchar < len(numstringp) - 1:
        if numstringp[curchar + 1] == 'g' and (numstringp[curchar] == '0' or numstringp[curchar] ==
                                               '1' or numstringp[curchar] == '2' or numstringp[curchar] ==
                                               '3' or numstringp[curchar] == '4' or numstringp[curchar] ==
                                               '5' or numstringp[curchar] == '6' or numstringp[curchar] ==
                                               '7' or numstringp[curchar] == '8' or numstringp[curchar] == '9'):
            numstringp[curchar + 1] = '9'
        if numstringp[curchar] == '9':
            if (numstringp[curchar + 1] != '0' and numstringp[curchar + 1] != '1' and numstringp[curchar + 1] !=
                    '2' and numstringp[curchar + 1] != '3' and numstringp[curchar + 1] != '4' and numstringp[
                        curchar + 1] !=
                    '5' and numstringp[curchar + 1] != '6' and numstringp[curchar + 1] != '7' and numstringp[
                        curchar + 1] !=
                    '8' and numstringp[curchar + 1] != '9'):
                if (numstringp[curchar - 1] == '0' or numstringp[curchar - 1] == '1' or numstringp[curchar - 1] ==
                        '2' or numstringp[curchar - 1] == '3' or numstringp[curchar - 1] == '4' or numstringp[
                            curchar - 1] ==
                        '5' or numstringp[curchar - 1] == '6' or numstringp[curchar - 1] == '7' or numstringp[
                            curchar - 1] ==
                        '8' or numstringp[curchar - 1] == '9'):
                    remover = numstringp[0:curchar] + numstringp[curchar + 1:len(numstringp)]
                    numstringp = remover
                    curchar -= 1
        curchar += 1
    if numstringp[len(numstringp) - 1] \
            == '9' and (numstringp[len(numstringp) - 2] == '0' or numstringp[len(numstringp) - 2] ==
                        '1' or numstringp[len(numstringp) - 2] == '2' or numstringp[len(numstringp) - 2] ==
                        '3' or numstringp[len(numstringp) - 2] == '4' or numstringp[len(numstringp) - 2] ==
                        '5' or numstringp[len(numstringp) - 2] == '6' or numstringp[len(numstringp) - 2] ==
                        '7' or numstringp[len(numstringp) - 2] == '8' or numstringp[len(numstringp) - 2] == '9'):
        newnumstringp = numstringp[0:len(numstringp) - 1]
        return newnumstringp
    return numstringp


def afternumprocess(fnumstring):
    if fnumstring == '' and fnumstring != '0':
        fnumstring = -2000
    if isinstance(fnumstring, str) and fnumstring.find('.') != -1:
        plop = fnumstring[0:fnumstring.find('.')+1] + fnumstring[fnumstring.find('.')+1:len(fnumstring)].replace('.', '')
        as_second_int_double = float(plop)
    else:
        as_second_int_double = int(fnumstring)
    return as_second_int_double


def classer(currecrd, fal):
    for fa in fal:
        for possibl in fa:
            if currecrd.find(possibl) != -1:
                return fa
    return '???'


def fix96prcnt(at):
    while at.find('96') != -1:
        strt = at[0:at.find('96')]
        nd = at[at.find('96') + 2:len(at)]
        at = strt + '%' + nd
    return at


def decrypt(atext):
    serving = ['Serving', 'serving', 'Serv', 'serv', 'Size', 'size']
    calories = ['Calories', 'calories', 'kcal']
    total_fat = ['Total Fat', 'total fat', 'Tot. Fat', 'tot. fat', 'TobalFatzig']
    saturated_fat = ['Saturated Fat', 'saturated fat', 'SaturatedFat', 'saturatedFat', 'saturatedfat',
                     'Sat. Fat', 'sat. fat']
    trans_fat = ['Trans Fat', 'Trans fat', 'TransFat', 'trans fat', 'transfat']
    cholesterol = ['Cholesterol', 'cholesterol', 'Cholest.', 'cholest.']
    sodium = ['Sodium', 'sodium', 'Sod.', 'sod.']
    total_carb = ['Total Carbohydrate', 'Carbohydrate', 'carbohydrate', 'Total Carb', 'total carb', 'Total Car .',
                  'Total Car.', 'Carb', 'carb']
    fiber = ['Fiber', 'fiber', 'Fib.', 'fib.']
    total_sugars = ['Total Sugars', 'total sugars', 'Total Sug.', 'total sug.', 'Tot. Sug.', 'tot. sug.', 'Sugars',
                    'Sugar', 'sugars', 'sugar']
    added_sugars = ['Added Sugars', 'added sugars', 'Added Sug.', 'added sug.', 'Add. Sug.', 'add. sug.']
    protein = ['Protein', 'protein', 'Prot.', 'prot.']
    calcium = ['Calcium', 'calcium']
    iron = ['Iron', 'iron']
    potassium = ['Potassium', 'potas']
    vitamin_a = ['Vitamin A', 'vitamin a', 'vitamin A', 'Vit A', 'vit a']
    vitamin_b = ['Vitamin B', 'vitamin b', 'vitamin B', 'Vit B', 'vit b']
    vitamin_c = ['Vitamin C', 'vitamin c', 'vitamin C', 'Vit C', 'vit c']
    vitamin_d = ['Vitamin D', 'vitamin d', 'vitamin D', 'Vit D', 'vit d']
    food_attr_list = [serving, calories, total_fat, saturated_fat, trans_fat, cholesterol, sodium,
                      total_carb, fiber, total_sugars, added_sugars, protein, calcium, iron, potassium,
                      vitamin_a, vitamin_b, vitamin_c, vitamin_d]
    atxt = fix96prcnt(atext)
    in_ordforertuple = decrypt_substep_a(atxt, food_attr_list)
    in_order = in_ordforertuple[0]
    print('in_order: ')
    print(in_order)
    ffood = decrypt_substep_b(in_ordforertuple, food_attr_list)
    return ffood


def checkifin(itmn, falst):
    tpop = 0
    tplfrrtrn = (False, 0)
    while tpop < len(falst):
        if itmn == falst[tpop]:
            tplfrrtrn = (True, tpop)
            return tplfrrtrn
        tpop += 1
    return tplfrrtrn


def noise(fdatrlst, crlst):
    checkerlist = [False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False]
    up = 0
    while up < len(crlst):
        tpleforcomparison = checkifin(crlst[up][0], fdatrlst)
        if tpleforcomparison[0]:
            checkerlist[tpleforcomparison[1]] = True
        up += 1
    cri = 0
    while cri < len(checkerlist):
        if not checkerlist[cri]:
            toappend = (fdatrlst[cri], -2000)
            crlst.append(toappend)
        cri += 1
    return crlst


def denoise(ii, lst):
    done = False
    p = ii + 1
    while p < len(lst) and not done:
        if lst[p][0] == lst[ii][0]:
            if lst[ii][1] == -2000:
                lst.pop(ii)
                done = True
                p -= 1
        p += 1
    res = (lst, done)
    return res


def denoisebw(ii, lst):
    done = False
    p = ii - 1
    while p >= 0 and not done:
        if lst[p][0] == lst[ii][0]:
            if lst[ii][1] == -2000:
                lst.pop(ii)
                done = True
                break
                # p += 1
        p -= 1
    res = (lst, done)
    return res


def denoisea(ii, lst):
    done = False
    p = ii + 1
    while p < len(lst) and not done:
        if lst[p][0] == lst[ii][0]:
            if lst[ii][1] > 1500:
                lst.pop(ii)
                done = True
                p -= 1
        p += 1
    res = (lst, done)
    return res


def denoisebwa(ii, lst):
    done = False
    p = ii - 1
    while p >= 0 and not done:
        if lst[p][0] == lst[ii][0]:
            if lst[ii][1] > 1500:
                lst.pop(ii)
                done = True
                break
                # p += 1
        p -= 1
    res = (lst, done)
    return res


def decrypt_substep_a(atext, foodattrlist):
    inorder = []
    forerlist = []
    for attribute in foodattrlist:
        for possible in attribute:
            forer = finditem(atext, possible, foodattrlist)
            forerrnum = itemtoserv(forer)
            forerrclass = classer(forer, foodattrlist)
            classnumtuple = (forerrclass, forerrnum)
            if forer != '???':
                forerlist.append(forer)
            if classnumtuple != ('???', -2000):
                inorder.append(classnumtuple)
    inorder = noise(foodattrlist, inorder)
    print('inorder noised:')
    print(inorder)
    frr = 0
    d = 0
    fulllength = len(inorder) - 1
    while frr < fulllength - d:
        scnd = denoise(frr, inorder)
        inorder = scnd[0]
        if scnd[1]:
            frr -= 1
            d += 1
        frr += 1
    frr = len(inorder) - 1
    while frr > 0:
        scnd = denoisebw(frr, inorder)
        inorder = scnd[0]
        if scnd[1]:
            frr += 1
        frr -= 1
    frr = 0
    d = 0
    fulllength = len(inorder) - 1
    while frr < fulllength - d:
        scnd = denoisea(frr, inorder)
        inorder = scnd[0]
        if scnd[1]:
            frr -= 1
            d += 1
        frr += 1
    frr = len(inorder) - 1
    while frr > 0:
        scnd = denoisebwa(frr, inorder)
        inorder = scnd[0]
        if scnd[1]:
            frr += 1
        frr -= 1
    exittuple = (inorder, forerlist)
    return exittuple


def conversion(frn_rdfrrtpl, fdatlt):
    nrdr = []
    d = 0
    while d < len(frn_rdfrrtpl[1]):
        tdcrypt = frn_rdfrrtpl[1][d]
        for q in fdatlt:
            for qq in q:
                if tdcrypt.find(qq) != -1:
                    if tdcrypt.find('%') != -1:
                        ntpl = (q, '%')
                    elif tdcrypt.find('mg') != -1:
                        ntpl = (q, 'mg')
                    elif tdcrypt.find('mcg') != -1:
                        ntpl = (q, 'mcg')
                    elif tdcrypt.find('g') != -1 and (tdcrypt[tdcrypt.find('g')-1:tdcrypt.find('g')] != 'n' and tdcrypt[tdcrypt.find('g')-1:tdcrypt.find('g')] != 'u'):
                        ntpl = (q, 'g')
                        # #######################################FIX THIS: ENSURE 'g' does not have 'u' or 'n' in front of it: DO THIS TO STRING ADDITION STOPPER AS WELL to account for "serving" and "sugars"
                    else:
                        ntpl = (q, '')
                    nrdr.append(ntpl)
        d += 1
    print(nrdr)
    for n in nrdr:
        u = 0
        while u < len(frn_rdfrrtpl[0]):
            if n[0] == frn_rdfrrtpl[0][u][0]:
                frn_rdfrrtpl[0][u] = list(frn_rdfrrtpl[0][u])
                frn_rdfrrtpl[0][u][1] *= getmultiple(n[0], fdatlt, n[1])
                frn_rdfrrtpl[0][u] = tuple(frn_rdfrrtpl[0][u])

            u += 1
    return frn_rdfrrtpl


def dailycount(lmnfdstrlst, cfdatlst):
    dc = [0, 2000, 65, 20, 2, 0.3, 2.4, 335, 25, 50, 24, 50, 1.3, 0.018, 4.7, 0.3, 1.35, 2, 0.01]
    m = 0
    while m < len(cfdatlst):
        if lmnfdstrlst == cfdatlst[m]:
            return dc[m]
        m += 1
    return 1


def getmultiple(eleminfdatrlst, fdatlst, ndsign):
    if ndsign == '%':
        return dailycount(eleminfdatrlst, fdatlst) / 100
    elif ndsign == 'mg':
        return 0.001
    elif ndsign == 'mcg':
        return 0.0001
    elif ndsign == 'g':
        return 1
    else:
        return 1


def decrypt_substep_b(forin_ordforertuple, fd_attr_lst):
    print('forin_ordforertuple')
    print(forin_ordforertuple)
    forin_ordforertuple = conversion(forin_ordforertuple, fd_attr_lst)
    newfood = Food('', '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    curtple = 0
    while curtple < len(forin_ordforertuple[0]):
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[0]:
            newfood.serving = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[1]:
            newfood.calories = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[2]:
            newfood.total_fat = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[3]:
            newfood.saturated_fat = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[4]:
            newfood.trans_fat = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[5]:
            newfood.cholesterol = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[6]:
            newfood.sodium = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[7]:
            newfood.total_carb = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[8]:
            newfood.fiber = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[9]:
            newfood.total_sugars = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[10]:
            newfood.added_sugars = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[11]:
            newfood.protein = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[12]:
            newfood.calcium = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[13]:
            newfood.iron = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[14]:
            newfood.potassium = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[15]:
            newfood.vitamin_a = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[16]:
            newfood.vitamin_b = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[17]:
            newfood.vitamin_c = forin_ordforertuple[0][curtple][1]
        if forin_ordforertuple[0][curtple][0] == fd_attr_lst[18]:
            newfood.vitamin_d = forin_ordforertuple[0][curtple][1]
        curtple += 1
    print('Serving: ')
    print(newfood.serving)
    print('Calories: ')
    print(newfood.calories)
    print('Total Fat: ')
    print(newfood.total_fat)
    print('Saturated Fat: ')
    print(newfood.saturated_fat)
    print('Trans Fat: ')
    print(newfood.trans_fat)
    print('Cholesterol: ')
    print(newfood.cholesterol)
    print('Sodium: ')
    print(newfood.sodium)
    print('Total Carb: ')
    print(newfood.total_carb)
    print('Fiber: ')
    print(newfood.fiber)
    print('Total Sugars: ')
    print(newfood.total_sugars)
    print('Added Sugars: ')
    print(newfood.added_sugars)
    print('Protein: ')
    print(newfood.protein)
    print('Calcium: ')
    print(newfood.calcium)
    print('Iron: ')
    print(newfood.iron)
    print('Potassium: ')
    print(newfood.potassium)
    print('Vitamin A: ')
    print(newfood.vitamin_a)
    print('Vitamin B: ')
    print(newfood.vitamin_b)
    print('Vitamin C: ')
    print(newfood.vitamin_c)
    print('Vitamin D: ')
    print(newfood.vitamin_d)
    food = newfood
    print(food)
    return food


def addfoodtotxt(capitalfood, txtfile):
    print('trying')
    try:
        bo = open(txtfile, 'r')
        bo.close()
    except FileNotFoundError:
        bm = open(txtfile, 'w')
        bm.close()
    with open(txtfile, 'a') as a:
        a.write(capitalfood.food_name + ' ' + capitalfood.food_datetime + ' ' + str(capitalfood.serving) + ' ' +
                str(capitalfood.calories) + ' ' + str(capitalfood.total_fat) + ' ' + str(capitalfood.saturated_fat) +
                ' ' + str(capitalfood.trans_fat) + ' ' + str(capitalfood.cholesterol) + ' ' + str(capitalfood.sodium) +
                ' ' + str(capitalfood.total_carb) + ' ' + str(capitalfood.fiber) + ' ' + str(capitalfood.total_sugars)
                + ' ' + str(capitalfood.added_sugars) + ' ' + str(capitalfood.protein) + ' ' + str(capitalfood.calcium)
                + ' ' + str(capitalfood.iron) + ' ' + str(capitalfood.potassium) + ' ' + str(capitalfood.vitamin_a) +
                ' ' + str(capitalfood.vitamin_b) + ' ' + str(capitalfood.vitamin_c) + ' ' + str(capitalfood.vitamin_d)
                + '\n')
    a.close()


def processstring(lcrd):
    toret = []
    fd = 0
    while fd < len(lcrd):
        posit = 0
        while posit < 21:
            if lcrd[fd].find(' ') != -1:
                crp = lcrd[fd][0:(lcrd[fd].find(' '))]
                if lcrd[fd].find(' ') + 1 < len(lcrd[fd]):
                    lcrd[fd] = lcrd[fd][lcrd[fd].find(' ') + 1:len(lcrd[fd])]
                else:
                    lcrd[fd].strip()
                    crp = lcrd[fd]
            else:
                lcrd[fd].strip()
                crp = lcrd[fd]
            try:
                crp = int(crp)
            except ValueError or TypeError:
                try:
                    crp = float(crp)
                except ValueError or TypeError:
                    pass
            finally:
                toret.append(crp)
            posit += 1
        fd += 1
    print('toret')
    print(toret)
    return toret


def multiplybyamt(foodtbm, numservings):
    foodtbm.serving *= numservings
    foodtbm.calories *= numservings
    foodtbm.total_fat *= numservings
    foodtbm.saturated_fat *= numservings
    foodtbm.trans_fat *= numservings
    foodtbm.cholesterol *= numservings
    foodtbm.sodium *= numservings
    foodtbm.total_carb *= numservings
    foodtbm.fiber *= numservings
    foodtbm.total_sugars *= numservings
    foodtbm.added_sugars *= numservings
    foodtbm.protein *= numservings
    foodtbm.calcium *= numservings
    foodtbm.iron *= numservings
    foodtbm.potassium *= numservings
    foodtbm.vitamin_a *= numservings
    foodtbm.vitamin_b *= numservings
    foodtbm.vitamin_c *= numservings
    foodtbm.vitamin_d *= numservings
    return foodtbm


def setvaluesforfood(lsf):
    lg = []
    k = 0
    while k < len(lsf):
        nfd = Food('', '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        nfd.food_name = lsf[k][0]
        nfd.food_datetime = lsf[k][1]
        nfd.serving = lsf[k][2]
        nfd.calories = lsf[k][3]
        nfd.total_fat = lsf[k][4]
        nfd.saturated_fat = lsf[k][5]
        nfd.trans_fat = lsf[k][6]
        nfd.cholesterol = lsf[k][7]
        nfd.sodium = lsf[k][8]
        nfd.total_carb = lsf[k][9]
        nfd.fiber = lsf[k][10]
        nfd.total_sugars = lsf[k][11]
        nfd.added_sugars = lsf[k][12]
        nfd.protein = lsf[k][13]
        nfd.calcium = lsf[k][14]
        nfd.iron = lsf[k][15]
        nfd.potassium = lsf[k][16]
        nfd.vitamin_a = lsf[k][17]
        nfd.vitamin_b = lsf[k][18]
        nfd.vitamin_c = lsf[k][19]
        nfd.vitamin_d = lsf[k][20]
        lg.append(nfd)
        k += 1
    return lg


def getlst(lff):
    nwlst = []
    o = 0
    while o < len(lff) / 21:
        oo = o * 21
        nlst = []
        while oo < (o + 1) * 21:
            nlst.append(lff[oo])
            oo += 1
        nwlst.append(nlst)
        o += 1
    return nwlst


def getter(lcrntrd):
    lstforfood = processstring(lcrntrd)
    lstforfoodtwo = getlst(lstforfood)
    lsttortrn = setvaluesforfood(lstforfoodtwo)
    return lsttortrn


def compare(c, it):
    crcount = 0
    if c.food_name == it.food_name:
        crcount += 1
    if c.food_datetime == it.food_datetime:
        crcount += 1
    if c.serving == it.serving:
        crcount += 1
    if c.calories == it.calories:
        crcount += 1
    if c.total_fat == it.total_fat:
        crcount += 1
    if c.saturated_fat == it.saturated_fat:
        crcount += 1
    if c.trans_fat == it.trans_fat:
        crcount += 1
    if c.cholesterol == it.cholesterol:
        crcount += 1
    if c.sodium == it.sodium:
        crcount += 1
    if c.total_carb == it.total_carb:
        crcount += 1
    if c.fiber == it.fiber:
        crcount += 1
    if c.total_sugars == it.total_sugars:
        crcount += 1
    if c.added_sugars == it.added_sugars:
        crcount += 1
    if c.protein == it.protein:
        crcount += 1
    if c.calcium == it.calcium:
        crcount += 1
    if c.iron == it.iron:
        crcount += 1
    if c.potassium == it.potassium:
        crcount += 1
    if c.vitamin_a == it.vitamin_a:
        crcount += 1
    if c.vitamin_b == it.vitamin_b:
        crcount += 1
    if c.vitamin_c == it.vitamin_c:
        crcount += 1
    if c.vitamin_d == it.vitamin_d:
        crcount += 1
    if crcount >= 19:
        return True
    else:
        return False


def addfoodst(d, du, ub, bu, ud, pu):
    dup = 0
    # ud = food list of est_ref food attribute lists
    while dup < len(du):
        if dup < len(bu):
            if bu[dup] == 'e':
                print("TRUE ERROR")
                # change up[dup][0] to up[dup].serving etc
                du[dup].food_name = ud[dup].food_name
                du[dup].food_datetime = ud[dup].food_datetime
                du[dup].serving = ud[dup].serving
                du[dup].calories = ud[dup].calories
                du[dup].total_fat = ud[dup].total_fat
                du[dup].saturated_fat = ud[dup].saturated_fat
                du[dup].trans_fat = ud[dup].trans_fat
                du[dup].cholesterol = ud[dup].cholesterol
                du[dup].sodium = ud[dup].sodium
                du[dup].total_carb = ud[dup].total_carb
                du[dup].fiber = ud[dup].fiber
                du[dup].total_sugars = ud[dup].total_sugars
                du[dup].added_sugars = ud[dup].added_sugars
                du[dup].protein = ud[dup].protein
                du[dup].calcium = ud[dup].calcium
                du[dup].iron = ud[dup].iron
                du[dup].potassium = ud[dup].potassium
                du[dup].vitamin_a = ud[dup].vitamin_a
                du[dup].vitamin_b = ud[dup].vitamin_b
                du[dup].vitamin_c = ud[dup].vitamin_c
                du[dup].vitamin_d = ud[dup].vitamin_d
                print('error')
                print(d.totalsfoodlist)
            elif bu[dup] == 'r':
                du[dup].food_name = pu[dup].food_name
                du[dup].food_datetime = pu[dup].food_datetime
                du[dup].serving = pu[dup].serving
                du[dup].calories = pu[dup].calories
                du[dup].total_fat = pu[dup].total_fat
                du[dup].saturated_fat = pu[dup].saturated_fat
                du[dup].trans_fat = pu[dup].trans_fat
                du[dup].cholesterol = pu[dup].cholesterol
                du[dup].sodium = pu[dup].sodium
                du[dup].total_carb = pu[dup].total_carb
                du[dup].fiber = pu[dup].fiber
                du[dup].total_sugars = pu[dup].total_sugars
                du[dup].added_sugars = pu[dup].added_sugars
                du[dup].protein = pu[dup].protein
                du[dup].calcium = pu[dup].calcium
                du[dup].iron = pu[dup].iron
                du[dup].potassium = pu[dup].potassium
                du[dup].vitamin_a = pu[dup].vitamin_a
                du[dup].vitamin_b = pu[dup].vitamin_b
                du[dup].vitamin_c = pu[dup].vitamin_c
                du[dup].vitamin_d = pu[dup].vitamin_d
            else:
                pass
        if du[dup].serving != -2000:
            print(str(d.totalsfoodlist[0]) + " += (" + str(du[dup].serving) + " * " + str(ub[dup]) + ")")
            d.totalsfoodlist[0] += (du[dup].serving * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[0].append(du[dup])
        if du[dup].calories != -2000:
            d.totalsfoodlist[1] += (du[dup].calories * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[1].append(du[dup])
        if du[dup].total_fat != -2000:
            d.totalsfoodlist[2] += (du[dup].total_fat * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[2].append(du[dup])
        if du[dup].saturated_fat != -2000:
            d.totalsfoodlist[3] += (du[dup].saturated_fat * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[3].append(du[dup])
        if du[dup].trans_fat != -2000:
            d.totalsfoodlist[4] += (du[dup].trans_fat * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[4].append(du[dup])
        if du[dup].cholesterol != -2000:
            d.totalsfoodlist[5] += (du[dup].cholesterol * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[5].append(du[dup])
        if du[dup].sodium != -2000:
            d.totalsfoodlist[6] += (du[dup].sodium * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[6].append(du[dup])
        if du[dup].total_carb != -2000:
            d.totalsfoodlist[7] += (du[dup].total_carb * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[7].append(du[dup])
        if du[dup].fiber != -2000:
            d.totalsfoodlist[8] += (du[dup].fiber * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[8].append(du[dup])
        if du[dup].total_sugars != -2000:
            d.totalsfoodlist[9] += (du[dup].total_sugars * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[9].append(du[dup])
        if du[dup].added_sugars != -2000:
            d.totalsfoodlist[10] += (du[dup].added_sugars * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[10].append(du[dup])
        if du[dup].protein != -2000:
            d.totalsfoodlist[11] += (du[dup].protein * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[11].append(du[dup])
        if du[dup].calcium != -2000:
            d.totalsfoodlist[12] += (du[dup].calcium * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[12].append(du[dup])
        if du[dup].iron != -2000:
            d.totalsfoodlist[13] += (du[dup].iron * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[13].append(du[dup])
        if du[dup].potassium != -2000:
            d.totalsfoodlist[14] += (du[dup].potassium * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[14].append(du[dup])
        if du[dup].vitamin_a != -2000:
            d.totalsfoodlist[15] += (du[dup].vitamin_a * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[15].append(du[dup])
        if du[dup].vitamin_b != -2000:
            d.totalsfoodlist[16] += (du[dup].vitamin_b * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[16].append(du[dup])
        if du[dup].vitamin_c != -2000:
            d.totalsfoodlist[17] += (du[dup].vitamin_c * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[17].append(du[dup])
        if du[dup].vitamin_d != -2000:
            d.totalsfoodlist[18] += (du[dup].vitamin_d * ub[dup])
        else:
            d.rejectsfoodlist_basedonfoodattributeindex[18].append(du[dup])
        dup += 1
        print(d.totalsfoodlist)
    print('HERE!')
    print(d.totalsfoodlist)


def changelstforcheckcustomtoafdlist(bylinevals):
    xrs = 0
    fdlstfrreturn = []
    while xrs < len(bylinevals):
        newfood = Food('', '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        xrn = 0
        while xrn < len(bylinevals[xrs]):
            if xrn >= 2:
                setattr(newfood, foodattrnamelst[xrn - 2], float(bylinevals[xrs][xrn][2]))
            elif xrn == 1:
                newfood.food_datetime = str(bylinevals[xrs][xrn][2])
            else:
                newfood.food_name = str(bylinevals[xrs][xrn][2])
            xrn += 1
        fdlstfrreturn.append(newfood)
        xrs += 1
    return fdlstfrreturn


def gettercustom(lstofcustomstrs):
    bylinevals = []
    gxx = 0
    while gxx < len(lstofcustomstrs):
        lstforcheck = []
        m = 0
        numrs = -1
        while m < len(lstofcustomstrs[gxx]) - 2:
            h = lstofcustomstrs[gxx][m:m + 1]
            g = lstofcustomstrs[gxx][m + 1:m + 2]
            if h == 'r' and g == ',':
                numrs += 1
                numhg = lstofcustomstrs[gxx][
                        m + 2:lstofcustomstrs[gxx][m + 2:len(lstofcustomstrs[gxx])].find(' ') + m + 2]
                lstforcheck.append([numrs, m, numhg])
            m += 1
        bylinevals.append(lstforcheck)
        gxx += 1
    bylinefds = changelstforcheckcustomtoafdlist(bylinevals)
    return bylinefds


def returncorrectusedfoods(dy, used):
    dup = 0
    # ud = food list of est_ref food attribute lists
    correctusedfoods = []
    txkt = defineifmissingestref(dy)
    typerc = readtypes(dy)
    txcustomt = defineifmissingindividuals(dy)
    with open(txcustomt, 'r') as klo:
        klor = klo.readlines()
        klo.close()
    customfds = gettercustom(klor)
    with open(txkt, 'r') as a:
        g = a.readlines()
        a.close()
    fds = getter(g)
    print("fds: ")
    print(fds)
    # Originally vvv len(used) instead len(fds)
    while dup < len(used):
        print("typerc[dup]")
        print(typerc[dup])
        if typerc[dup] == 'e':
            correctusedfoods.append(fds[dup])
            print("fds[dup]")
            print(fds[dup])
        elif typerc[dup] == 'r':
            correctusedfoods.append(customfds[dup])
            print("customfds[dup]")
            print(customfds[dup])
        else:
            correctusedfoods.append(used[dup])
            print("used[dup]")
            print(used[dup])
        dup += 1
    return correctusedfoods


def b_returncorrectusedfoods(used):
    dup = 0
    # ud = food list of est_ref food attribute lists
    correctusedfoods = []
    txkt = b_defineifmissingestref()
    typerc = b_readtypes()
    txcustomt = b_defineifmissingindividuals()
    with open(txcustomt, 'r') as klo:
        klor = klo.readlines()
        klo.close()
    customfds = gettercustom(klor)
    with open(txkt, 'r') as a:
        g = a.readlines()
        a.close()
    fds = getter(g)
    # Originally vvv len(used) instead len(fds)
    while dup < len(used):
        if typerc[dup] == 'e':
            correctusedfoods.append(fds[dup])
        elif typerc[dup] == 'r':
            correctusedfoods.append(customfds[dup])
        else:
            correctusedfoods.append(used[dup])
        dup += 1
    return correctusedfoods


def writetolinemix(cday, attribute, line, towrite):
    txtfile = defineifmissingmisessiontwo(cday, attribute)
    with open(txtfile, 'r') as rd:
        displacementlist = rd.readlines()
        rd.close()
    displacementlist[line - 1] = towrite
    with open(txtfile, 'w') as wr:
        wr.write('')
        wr.close()
    with open(txtfile, 'a+') as ap:
        for ln in displacementlist:
            ap.write(ln)
        ap.close()
    file_lnlst = (txtfile, displacementlist)
    return file_lnlst


def writetoline(cday, line, towrite):
    txtfile = defineifmissingtype(cday)
    with open(txtfile, 'r') as rd:
        displacementlist = rd.readlines()
        rd.close()
    displacementlist[line - 1] = towrite
    with open(txtfile, 'w') as wr:
        wr.write('')
        wr.close()
    with open(txtfile, 'a+') as ap:
        for ln in displacementlist:
            ap.write(ln)
        ap.close()
    file_lnlst = (txtfile, displacementlist)
    return file_lnlst


def b_writetoline(line, towrite):
    txtfile = b_defineifmissingtype()
    with open(txtfile, 'r') as rd:
        displacementlist = rd.readlines()
        rd.close()
    displacementlist[line - 1] = towrite
    with open(txtfile, 'w') as wr:
        wr.write('')
        wr.close()
    with open(txtfile, 'a+') as ap:
        for ln in displacementlist:
            ap.write(ln)
        ap.close()
    file_lnlst = (txtfile, displacementlist)
    return file_lnlst


def writetolineindividuals(cday, line, towrite):
    txtfile = defineifmissingindividuals(cday)
    with open(txtfile, 'r') as rd:
        displacementlist = rd.readlines()
        rd.close()
    displacementlist[line - 1] = towrite
    with open(txtfile, 'w') as wr:
        wr.write('')
        wr.close()
    with open(txtfile, 'a+') as ap:
        for ln in displacementlist:
            ap.write(ln)
        ap.close()
    file_lnlst = (txtfile, displacementlist)
    return file_lnlst


def b_writetolineindividuals(cday, line, towrite):
    txtfile = b_defineifmissingindividuals()
    with open(txtfile, 'r') as rd:
        displacementlist = rd.readlines()
        rd.close()
    displacementlist[line - 1] = towrite
    with open(txtfile, 'w') as wr:
        wr.write('')
        wr.close()
    with open(txtfile, 'a+') as ap:
        for ln in displacementlist:
            ap.write(ln)
        ap.close()
    file_lnlst = (txtfile, displacementlist)
    return file_lnlst


def addfoods(curday, fooditem, amtofsrvings):
    typecurtxt = defineifmissingtype(curday=curday)
    with open(typecurtxt, 'a+') as tct:
        tct.write('t,' + str(amtofsrvings) + '\n')
        tct.close()
    truecurtxt = defineifmissing(curday=curday)
    if fooditem.serving != -2000:
        curday.totalsfoodlist[0] += (fooditem.serving * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[0].append(fooditem)
    if fooditem.calories != -2000:
        curday.totalsfoodlist[1] += (fooditem.calories * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[1].append(fooditem)
    if fooditem.total_fat != -2000:
        curday.totalsfoodlist[2] += (fooditem.total_fat * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[2].append(fooditem)
    if fooditem.saturated_fat != -2000:
        curday.totalsfoodlist[3] += (fooditem.saturated_fat * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[3].append(fooditem)
    if fooditem.trans_fat != -2000:
        curday.totalsfoodlist[4] += (fooditem.trans_fat * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[4].append(fooditem)
    if fooditem.cholesterol != -2000:
        curday.totalsfoodlist[5] += (fooditem.cholesterol * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[5].append(fooditem)
    if fooditem.sodium != -2000:
        curday.totalsfoodlist[6] += (fooditem.sodium * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[6].append(fooditem)
    if fooditem.total_carb != -2000:
        curday.totalsfoodlist[7] += (fooditem.total_carb * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[7].append(fooditem)
    if fooditem.fiber != -2000:
        curday.totalsfoodlist[8] += (fooditem.fiber * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[8].append(fooditem)
    if fooditem.total_sugars != -2000:
        curday.totalsfoodlist[9] += (fooditem.total_sugars * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[9].append(fooditem)
    if fooditem.added_sugars != -2000:
        curday.totalsfoodlist[10] += (fooditem.added_sugars * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[10].append(fooditem)
    if fooditem.protein != -2000:
        curday.totalsfoodlist[11] += (fooditem.protein * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[11].append(fooditem)
    if fooditem.calcium != -2000:
        curday.totalsfoodlist[12] += (fooditem.calcium * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[12].append(fooditem)
    if fooditem.iron != -2000:
        curday.totalsfoodlist[13] += (fooditem.iron * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[13].append(fooditem)
    if fooditem.potassium != -2000:
        curday.totalsfoodlist[14] += (fooditem.potassium * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[14].append(fooditem)
    if fooditem.vitamin_a != -2000:
        curday.totalsfoodlist[15] += (fooditem.vitamin_a * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[15].append(fooditem)
    if fooditem.vitamin_b != -2000:
        curday.totalsfoodlist[16] += (fooditem.vitamin_b * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[16].append(fooditem)
    if fooditem.vitamin_c != -2000:
        curday.totalsfoodlist[17] += (fooditem.vitamin_c * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[17].append(fooditem)
    if fooditem.vitamin_d != -2000:
        curday.totalsfoodlist[18] += (fooditem.vitamin_d * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[18].append(fooditem)
    addfoodtotxt(fooditem, truecurtxt)
    return curday


def b_addfoods(curday, fooditem, amtofsrvings):
    typecurtxt = b_defineifmissingtype()
    with open(typecurtxt, 'a+') as tct:
        tct.write('t,' + str(amtofsrvings) + '\n')
        tct.close()
    truecurtxt = b_defineifmissing()
    if fooditem.serving != -2000:
        curday.totalsfoodlist[0] += (fooditem.serving * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[0].append(fooditem)
    if fooditem.calories != -2000:
        curday.totalsfoodlist[1] += (fooditem.calories * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[1].append(fooditem)
    if fooditem.total_fat != -2000:
        curday.totalsfoodlist[2] += (fooditem.total_fat * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[2].append(fooditem)
    if fooditem.saturated_fat != -2000:
        curday.totalsfoodlist[3] += (fooditem.saturated_fat * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[3].append(fooditem)
    if fooditem.trans_fat != -2000:
        curday.totalsfoodlist[4] += (fooditem.trans_fat * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[4].append(fooditem)
    if fooditem.cholesterol != -2000:
        curday.totalsfoodlist[5] += (fooditem.cholesterol * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[5].append(fooditem)
    if fooditem.sodium != -2000:
        curday.totalsfoodlist[6] += (fooditem.sodium * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[6].append(fooditem)
    if fooditem.total_carb != -2000:
        curday.totalsfoodlist[7] += (fooditem.total_carb * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[7].append(fooditem)
    if fooditem.fiber != -2000:
        curday.totalsfoodlist[8] += (fooditem.fiber * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[8].append(fooditem)
    if fooditem.total_sugars != -2000:
        curday.totalsfoodlist[9] += (fooditem.total_sugars * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[9].append(fooditem)
    if fooditem.added_sugars != -2000:
        curday.totalsfoodlist[10] += (fooditem.added_sugars * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[10].append(fooditem)
    if fooditem.protein != -2000:
        curday.totalsfoodlist[11] += (fooditem.protein * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[11].append(fooditem)
    if fooditem.calcium != -2000:
        curday.totalsfoodlist[12] += (fooditem.calcium * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[12].append(fooditem)
    if fooditem.iron != -2000:
        curday.totalsfoodlist[13] += (fooditem.iron * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[13].append(fooditem)
    if fooditem.potassium != -2000:
        curday.totalsfoodlist[14] += (fooditem.potassium * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[14].append(fooditem)
    if fooditem.vitamin_a != -2000:
        curday.totalsfoodlist[15] += (fooditem.vitamin_a * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[15].append(fooditem)
    if fooditem.vitamin_b != -2000:
        curday.totalsfoodlist[16] += (fooditem.vitamin_b * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[16].append(fooditem)
    if fooditem.vitamin_c != -2000:
        curday.totalsfoodlist[17] += (fooditem.vitamin_c * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[17].append(fooditem)
    if fooditem.vitamin_d != -2000:
        curday.totalsfoodlist[18] += (fooditem.vitamin_d * amtofsrvings)
    else:
        curday.rejectsfoodlist_basedonfoodattributeindex[18].append(fooditem)
    addfoodtotxt(fooditem, truecurtxt)
    return curday


def numfoodsenteredsofar(cwrdaynl):
    rcurtxtnl = defineifmissing(cwrdaynl)
    with open(rcurtxtnl, 'r') as rfnl:
        rlcurenterednl = rfnl.readlines()
        cwrdaynl.usedfoodsincount = getter(rlcurenterednl)
        print(len(cwrdaynl.usedfoodsincount))
    return len(cwrdaynl.usedfoodsincount)


def b_numfoodsenteredsofar(cwrdaynl):
    rcurtxtnl = b_defineifmissing()
    with open(rcurtxtnl, 'r') as rfnl:
        rlcurenterednl = rfnl.readlines()
        cwrdaynl.usedfoodsincount = getter(rlcurenterednl)
        print(len(cwrdaynl.usedfoodsincount))
    return len(cwrdaynl.usedfoodsincount)


def row_exists_client(cursor, username, date):
    cursor.execute("SELECT username FROM dietfriend_client_food_data WHERE username = \'" + username +
                   "\' AND date = \'" + date + "\'")
    return cursor.fetchone() is not None


def row_exists_business(cursor, username):
    cursor.execute("SELECT username FROM dietfriend_business_food_data WHERE username = \'" + username + "\'")
    return cursor.fetchone() is not None


def followers_exist(cursor, b_name):
    cursor.execute("SELECT followers FROM business_followers WHERE business_name = \'" + b_name + "\'")
    return cursor.fetchone() is not None


def followers_exist_c(cursor, username):
    cursor.execute("SELECT followings FROM client_followings WHERE username = \'" + username + "\'")
    return cursor.fetchone() is not None


def row_exists_theme(cursor, username):
    cursor.execute("SELECT username FROM client_settings WHERE username = \'" + username +
                   "\'")
    return cursor.fetchone() is not None


def row_exists_primary_p(cursor, username):
    cursor.execute("SELECT username FROM client_settings WHERE username = \'" + username +
                   "\'")
    return cursor.fetchone() is not None


def row_exists_bspecialinfo(cursor, username):
    cursor.execute("SELECT business_name FROM business_special_info WHERE business_name = \'" + username +
                   "\'")
    return cursor.fetchone() is not None


def row_exists_moreinfo(cursor, username, date):
    cursor.execute(
        "SELECT username FROM client_moreinfo_value_storage WHERE username = \'" + username + "\' AND datetime = \'" + date + "\'")
    return cursor.fetchone() is not None


def check_exists_description(cursor, username):
    if row_exists_bspecialinfo(cursor, username):
        cursor.execute("SELECT description FROM business_special_info WHERE business_name = \'" + username +
                       "\'")
        return str(cursor.fetchone()).find('None') == -1
    else:
        str_to_execute = \
            "INSERT INTO business_special_info(business_name) VALUES(\'" + username + "\')"
        cursor.execute(str_to_execute)
        cursor.execute("COMMIT")
        return False


def check_exists_icons(cursor, username):
    if row_exists_bspecialinfo(cursor, username):
        cursor.execute("SELECT icon_names FROM business_special_info WHERE business_name = \'" + username +
                       "\'")
        return str(cursor.fetchone()).find('None') == -1
    else:
        str_to_execute = \
            "INSERT INTO business_special_info(business_name) VALUES(\'" + username + "\')"
        cursor.execute(str_to_execute)
        cursor.execute("COMMIT")
        return False


def check_exists_loc(cursor, username):
    if row_exists_bspecialinfo(cursor, username):
        cursor.execute("SELECT lat FROM business_special_info WHERE business_name = \'" + username +
                       "\'")
        return str(cursor.fetchone()).find('None') == -1
    else:
        str_to_execute = \
            "INSERT INTO business_special_info(business_name) VALUES(\'" + username + "\')"
        cursor.execute(str_to_execute)
        cursor.execute("COMMIT")
        return False


def parameter_based_check_db_row(cursor, table, condition_1_name, cond_1, condition_2_name=None, cond_2=None):
    if condition_2_name is None and cond_2 is None:
        cursor.execute(
            "SELECT "+condition_1_name+" FROM "+table+" WHERE "+condition_1_name+" = \'" + cond_1 + "\'")
        return cursor.fetchone() is not None
    else:
        cursor.execute(
            "SELECT " + condition_1_name + " FROM " + table + " WHERE " + condition_1_name + " = \'" + cond_1 + "\' AND " + condition_2_name + " = \'" + cond_2 + "\'")
        return cursor.fetchone() is not None


def clearmydietfriend_pictures():
    global cleared
    internalpath = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
    if not os.path.exists(internalpath):
        os.makedirs(internalpath)
    else:
        uop = os.getcwd()
        os.chdir(internalpath)
        q = os.listdir(internalpath)
        qq = 0
        while qq < len(q):
            os.remove(q[qq])
            qq += 1
        os.chdir(uop)
        cleared = True


def defineifmissing(curday):
    global user
    global universal_list
    curdayupdtd = str(curday.date_time)
    curdayfupdtd = curdayupdtd[0:10]
    curtxt = curdayfupdtd + user + '.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        clearmydietfriend_pictures()
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, curdayfupdtd):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + curdayfupdtd + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    # query = sql.SQL("SELECT dietfriend_client_food_data.fulltextfile_t FROM dietfriend_client_food_data WHERE (username = \'"+user+"\') AND (date = \'"+curdayfupdtd+"\')")
    # print("Query: ")
    # print(query)
    # cur.execute(query)
    # cur.execute("COMMIT")
    return curtxt


def defineifmissingdt(dt):
    global user
    global universal_list
    curtxt = dt + user + '.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, dt):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + dt + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


def b_defineifmissing():
    global user
    curtxt = user + '.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_business(cur, user):
        query = sql.SQL(
            "INSERT INTO dietfriend_business_food_data (username) VALUES (\'" + user + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


def defineifmissingtype(curday):
    curdayupdtd = str(curday.date_time)
    curdayfupdtd = curdayupdtd[0:10]
    curtxt = curdayfupdtd + user + '_type.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, curdayfupdtd):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + curdayfupdtd + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    # query = sql.SQL(
    #     "SELECT dietfriend_client_food_data.typefile_tx_ex_rx_ FROM dietfriend_client_food_data WHERE (username = \'" + user + "\') AND (date = \'" + curdayfupdtd + "\')")
    # print("Query: ")
    # print(query)
    # cur.execute(query)
    # cur.execute("COMMIT")
    return curtxt


def defineifmissingtypedt(dt):
    curtxt = dt + user + '_type.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, dt):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + dt + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


def b_defineifmissingtype():
    global user
    curtxt = user + '_type.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_business(cur, user):
        query = sql.SQL(
            "INSERT INTO dietfriend_business_food_data (username) VALUES (\'" + user + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


# def fixfilestring(c):
#     h = 0
#     while h < len(c):
#         if c[h] == '.' or c[h] == ':' or c[h] == ' ':
#             c = c[0:h] + '-' + c[h+1:len(c)]
#             h -= 1
#         h += 1
#     return c


def b_defineifmissingmispecsession():
    curdayupdtd = str(running_id)
    # curdayfupdtd = fixfilestring(curdayupdtd)
    print("Session_ID9")
    print(curdayupdtd)
    curtxt = curdayupdtd + user + '_bmispec.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingmisession(stringtoconcatenate):
    curdayupdtd = str(running_id)
    # curdayfupdtd = fixfilestring(curdayupdtd)
    print("Session_ID1")
    print(curdayupdtd)
    curtxt = curdayupdtd + user + '_mi' + stringtoconcatenate + '.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingpic():
    curdayupdtd = str(running_id)
    curtxt = curdayupdtd + user + '_profile_pic.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingcod():
    curdayupdtd = str(running_id)
    curtxt = curdayupdtd + user + '_cod.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingclientfavorites():
    curdayupdtd = str(running_id)
    curtxt = curdayupdtd + user + '_client_favorites.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingmisessionpopup():
    curdayupdtd = str(running_id)
    curtxt = curdayupdtd + user + '_mip.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingmisessiontwo(cd, attribute):
    curtxt = str(cd.date_time)[0:10] + user + '_im_two' + attribute + '.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingmisessiontwodt(dt, attribute):
    curtxt = dt + user + '_im_two' + attribute + '.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingmisessiontype():
    curdayupdtd = str(running_id)
    # curdayfupdtd = fixfilestring(curdayupdtd)
    print("Session_ID2")
    print(curdayupdtd)
    curtxt = curdayupdtd + user + '_mitype.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingflsession():
    curdayupdtd = str(running_id)
    # curdayfupdtd = fixfilestring(curdayupdtd)
    print("Session_ID3")
    print(curdayupdtd)
    curtxt = curdayupdtd + user + '_fl.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingbusinesstofollowsession():
    curdayupdtd = str(running_id)
    # curdayfupdtd = fixfilestring(curdayupdtd)
    print("Session_ID3")
    print(curdayupdtd)
    curtxt = curdayupdtd + user + '_fllw.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def b_defineifmissingflsession():
    curdayupdtd = str(running_id)
    # curdayfupdtd = fixfilestring(curdayupdtd)
    print("Session_ID5")
    print(curdayupdtd)
    curtxt = curdayupdtd + user + '_bfl.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissinghint():
    global user
    curtxt = user + '_hint.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingtocheckforpopupadd():
    global user
    curtxt = user + '_popup_desig.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissingestref(curday):
    curdayupdtd = str(curday.date_time)
    curdayfupdtd = curdayupdtd[0:10]
    curtxt = curdayfupdtd + user + '_est_ref.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, curdayfupdtd):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + curdayfupdtd + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    # query = sql.SQL(
    #     "SELECT dietfriend_client_food_data.esttextfile_e FROM dietfriend_client_food_data WHERE (username = \'" + user + "\') AND (date = \'" + curdayfupdtd + "\')")
    # print("Query: ")
    # print(query)
    # cur.execute(query)
    # cur.execute("COMMIT")
    return curtxt


def defineifmissingestrefdt(dt):
    curtxt = dt + user + '_est_ref.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, dt):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + dt + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


def b_defineifmissingestref():
    global user
    curtxt = user + '_est_ref.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_business(cur, user):
        query = sql.SQL(
            "INSERT INTO dietfriend_business_food_data (username) VALUES (\'" + user + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


def defineifmissingindividuals(curday):
    curdayupdtd = str(curday.date_time)
    curdayfupdtd = curdayupdtd[0:10]
    curtxt = curdayfupdtd + user + '_individuals.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, curdayfupdtd):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + curdayfupdtd + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    # query = sql.SQL(
    #     "SELECT dietfriend_client_food_data.individualstextfile_r FROM dietfriend_client_food_data WHERE (username = \'" + user + "\') AND (date = \'" + curdayfupdtd + "\')")
    # print("Query: ")
    # print(query)
    # cur.execute(query)
    # cur.execute("COMMIT")
    return curtxt


def defineifmissingindividualsdt(dt):
    curtxt = dt + user + '_individuals.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_client(cur, user, dt):
        query = sql.SQL(
            "INSERT INTO dietfriend_client_food_data (username, date) VALUES (\'" + user + "\', \'" + dt + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


def b_defineifmissingindividuals():
    global user
    curtxt = user + '_individuals.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    global con
    cur = con.cursor()
    if not row_exists_business(cur, user):
        query = sql.SQL(
            "INSERT INTO dietfriend_business_food_data (username) VALUES (\'" + user + "\')")
        print("Query: ")
        print(query)
        cur.execute(query)
        cur.execute("COMMIT")
    return curtxt


def defineifmissing_prev_insecure_settings():
    curtxt = "prev_settings.txt"
    prev_user = ''
    try:
        with open(curtxt, 'r') as fo:
            prev_user = fo.readlines()
            print("prev_user in defineifmissing_prev_insecure_settings: ")
            print(prev_user)
            print("END prev_user")
            fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    # global con
    # cur = con.cursor()
    # if row_exists_theme(cur, prev_user):
    #     query = sql.SQL(
    #         "SELECT bg_theme FROM client_settings WHERE (username = \'" + prev_user + "\')")
    #     print("Query: ")
    #     print(query)
    #     cur.execute(query)
    try:
        return [curtxt, prev_user[0]]
    except IndexError:
        return [curtxt, '']


def readnumservings(cdy):
    txtt = defineifmissingtype(cdy)
    with open(txtt, 'r') as tgtg:
        h = tgtg.readlines()
        tgtg.close()
    print(h)
    n = 0
    while n < len(h):
        h[n] = h[n][h[n].find(',') + 1:h[n].find('\n')]
        n += 1
    print('readnumservings: ')
    print(h)
    nl = []
    for p in h:
        nl.append(float(p))
    print(nl)
    return nl


def b_readnumservings():
    txtt = b_defineifmissingtype()
    with open(txtt, 'r') as tgtg:
        h = tgtg.readlines()
        tgtg.close()
    print(h)
    n = 0
    while n < len(h):
        h[n] = h[n][h[n].find(',') + 1:h[n].find('\n')]
        n += 1
    print('readnumservings: ')
    print(h)
    nl = []
    for p in h:
        nl.append(float(p))
    print(nl)
    return nl


def readtypes(cdy):
    txtt = defineifmissingtype(cdy)
    with open(txtt, 'r') as tgtg:
        h = tgtg.readlines()
        tgtg.close()
    print(h)
    n = 0
    while n < len(h):
        h[n] = h[n][0:h[n].find(',')]
        n += 1
    print("h:")
    print(h)
    return h


def b_readtypes():
    txtt = b_defineifmissingtype()
    with open(txtt, 'r') as tgtg:
        h = tgtg.readlines()
        tgtg.close()
    print(h)
    n = 0
    while n < len(h):
        h[n] = h[n][0:h[n].find(',')]
        n += 1
    print("h:")
    print(h)
    return h


def getrandomid():
    inter = int((random() * 300 * random() * 800 * ((random() * 40) % 72) * random() * 16 - random() * 190 /
                 pow(random() * 2, 4)) * random() * 100000)
    return inter


def deletefoodfromalltextandreducetotal(cdy, q):
    """[readtxt, esttxt, typetxt, indtxt]"""
    dellst = [defineifmissing(cdy), defineifmissingestref(cdy), defineifmissingtype(cdy),
              defineifmissingindividuals(cdy)]
    for p in dellst:
        with open(p, 'r') as curp:
            curlst = curp.readlines()
            curfds = getter(curlst)
            curp.close()
        try:
            curlst.pop(q)
        except IndexError:
            pass
        st = ''
        for e in curlst:
            st += e
        with open(p, 'w') as nexp:
            nexp.write(st)
            nexp.close()
    try:
        cdy.totalsfoodlist[0] -= float(curfds[q].serving)
        cdy.totalsfoodlist[1] -= float(curfds[q].calories)
        cdy.totalsfoodlist[2] -= float(curfds[q].total_fat)
        cdy.totalsfoodlist[3] -= float(curfds[q].saturated_fat)
        cdy.totalsfoodlist[4] -= float(curfds[q].trans_fat)
        cdy.totalsfoodlist[5] -= float(curfds[q].cholesterol)
        cdy.totalsfoodlist[6] -= float(curfds[q].sodium)
        cdy.totalsfoodlist[7] -= float(curfds[q].total_carb)
        cdy.totalsfoodlist[8] -= float(curfds[q].fiber)
        cdy.totalsfoodlist[9] -= float(curfds[q].total_sugars)
        cdy.totalsfoodlist[10] -= float(curfds[q].added_sugars)
        cdy.totalsfoodlist[11] -= float(curfds[q].protein)
        cdy.totalsfoodlist[12] -= float(curfds[q].calcium)
        cdy.totalsfoodlist[13] -= float(curfds[q].iron)
        cdy.totalsfoodlist[14] -= float(curfds[q].potassium)
        cdy.totalsfoodlist[15] -= float(curfds[q].vitamin_a)
        cdy.totalsfoodlist[16] -= float(curfds[q].vitamin_b)
        cdy.totalsfoodlist[17] -= float(curfds[q].vitamin_c)
        cdy.totalsfoodlist[18] -= float(curfds[q].vitamin_d)
        curfds.pop(q)
    except IndexError:
        pass
    return curfds


def b_deletefoodfromalltextandreducetotal(cdy, q):
    """[readtxt, esttxt, typetxt, indtxt]"""
    dellst = [b_defineifmissing(), b_defineifmissingestref(), b_defineifmissingtype(),
              b_defineifmissingindividuals()]
    for p in dellst:
        with open(p, 'r') as curp:
            curlst = curp.readlines()
            curfds = getter(curlst)
            curp.close()
        try:
            curlst.pop(q)
        except IndexError:
            pass
        st = ''
        for e in curlst:
            st += e
        with open(p, 'w') as nexp:
            nexp.write(st)
            nexp.close()
    try:
        cdy.totalsfoodlist[0] -= float(curfds[q].serving)
        cdy.totalsfoodlist[1] -= float(curfds[q].calories)
        cdy.totalsfoodlist[2] -= float(curfds[q].total_fat)
        cdy.totalsfoodlist[3] -= float(curfds[q].saturated_fat)
        cdy.totalsfoodlist[4] -= float(curfds[q].trans_fat)
        cdy.totalsfoodlist[5] -= float(curfds[q].cholesterol)
        cdy.totalsfoodlist[6] -= float(curfds[q].sodium)
        cdy.totalsfoodlist[7] -= float(curfds[q].total_carb)
        cdy.totalsfoodlist[8] -= float(curfds[q].fiber)
        cdy.totalsfoodlist[9] -= float(curfds[q].total_sugars)
        cdy.totalsfoodlist[10] -= float(curfds[q].added_sugars)
        cdy.totalsfoodlist[11] -= float(curfds[q].protein)
        cdy.totalsfoodlist[12] -= float(curfds[q].calcium)
        cdy.totalsfoodlist[13] -= float(curfds[q].iron)
        cdy.totalsfoodlist[14] -= float(curfds[q].potassium)
        cdy.totalsfoodlist[15] -= float(curfds[q].vitamin_a)
        cdy.totalsfoodlist[16] -= float(curfds[q].vitamin_b)
        cdy.totalsfoodlist[17] -= float(curfds[q].vitamin_c)
        cdy.totalsfoodlist[18] -= float(curfds[q].vitamin_d)
        curfds.pop(q)
    except IndexError:
        pass
    return curfds


def checkforduplicatesandremove(cdy):
    tempufic = cdy.usedfoodsincount
    i = 0
    while i < len(tempufic) - 1:
        q = i + 1
        while q < len(tempufic):
            if compare(tempufic[q], tempufic[i]):
                tempufic.pop(q)
                deletefoodfromalltextandreducetotal(cdy, q)
                q -= 1
            q += 1
        i += 1
    return tempufic


def b_checkforduplicatesandremove(cdy):
    tempufic = cdy.usedfoodsincount
    i = 0
    while i < len(tempufic) - 1:
        q = i + 1
        while q < len(tempufic):
            if compare(tempufic[q], tempufic[i]):
                tempufic.pop(q)
                b_deletefoodfromalltextandreducetotal(cdy, q)
                q -= 1
            q += 1
        i += 1
    return tempufic


def getest_ref_food_list_of_food_attributes(cdy):
    ctexxt = defineifmissingestref(cdy)
    with open(ctexxt, 'r') as fh:
        glcurentered = fh.readlines()
        est_ref_food_list_ofa = getter(glcurentered)
        fh.close()
    return est_ref_food_list_ofa


def b_getest_ref_food_list_of_food_attributes():
    ctexxt = b_defineifmissingestref()
    with open(ctexxt, 'r') as fh:
        glcurentered = fh.readlines()
        est_ref_food_list_ofa = getter(glcurentered)
        fh.close()
    return est_ref_food_list_ofa


def standardize(cd):
    d = defineifmissingindividuals(cd)
    with open(d, 'a') as ap:
        ap.write('n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 \n')
        ap.close()


def b_standardize():
    d = b_defineifmissingindividuals()
    with open(d, 'a') as ap:
        ap.write('n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 n,0 \n')
        ap.close()


def fixalldata(data):
    datastring = str(data)
    datalist = []
    w = 0
    while datastring.find("\', \'") != -1 or datastring.find("\')") != -1:
        if datastring.find("\', \'") != -1 and w == 0:
            datalist.append(datastring[3:datastring.find("\', \'")])
            datastring = datastring[datastring.find("\', \'") + 4:len(datastring)]
        elif datastring.find("\', \'") != -1:
            datalist.append(datastring[0:datastring.find("\', \'")])
            datastring = datastring[datastring.find("\', \'") + 4:len(datastring)]
        else:
            datalist.append(datastring[0:datastring.find("\')")])
            datastring = ""
        w += 1
    w = 0
    while w < len(datalist):
        datalist[w] = datalist[w].replace('\\n', '\n')
        # while datalist[w].find('\\n') != -1:
        #     datalist[w] = datalist[w][0:datalist[w].find('\\n')+1] + \
        #                   datalist[w][datalist[w].find('\\n')+2:len(datalist[w])]
        w += 1
    # print("datalist: ")
    # print(datalist)
    # print(datalist[0])
    # print(datalist[1])
    # print(datalist[2])
    # print(datalist[3])
    return datalist


def fixalldatabasedonnewline(data):
    pdata = str(data).replace('[', '').replace('(', '').replace(')', '').replace(']', '').replace(',', '').replace('\'',
                                                                                                                   '').replace(
        '\\n', ',').replace('\n', ',')
    v = []
    while len(pdata) > 1:
        toappend = pdata[0:pdata.find(',')]
        print(toappend)
        v.append(toappend)
        pdata = pdata[pdata.find(',') + 1:len(pdata)]
    print(v)
    return v


def remakeifmade(cr, cuser, d):
    global con
    cursorp = con.cursor()
    if row_exists_client(cursor=cursorp, username=cuser, date=d):
        query = sql.SQL(
            "SELECT dietfriend_client_food_data.fulltextfile_t, dietfriend_client_food_data.esttextfile_e, dietfriend_client_food_data.individualstextfile_r, dietfriend_client_food_data.typefile_tx_ex_rx_ FROM dietfriend_client_food_data WHERE (username = \'" + cuser + "\') AND (date = \'" + d + "\')")
        print("Query: ")
        print(query)
        cursorp.execute(query)
        alldata = cursorp.fetchall()
        print("All Data:")
        print(alldata)
        alldatafixed = fixalldata(alldata)
        try:
            ftf_t = alldatafixed[0]
            etf_e = alldatafixed[1]
            itf_r = alldatafixed[2]
            tf_txexrx = alldatafixed[3]
            print("ftf_t")
            print(ftf_t)
            print("etf_e")
            print(etf_e)
            print("itf_r")
            print(itf_r)
            print("tf_txexrx")
            print(tf_txexrx)
            full = defineifmissing(cr)
            est = defineifmissingestref(cr)
            ind = defineifmissingindividuals(cr)
            typ = defineifmissingtype(cr)
            with open(full, 'w') as full_open:
                full_open.write(ftf_t)
                full_open.close()
            with open(est, 'w') as est_open:
                est_open.write(etf_e)
                est_open.close()
            with open(ind, 'w') as ind_open:
                ind_open.write(itf_r)
                ind_open.close()
            with open(typ, 'w') as typ_open:
                typ_open.write(tf_txexrx)
                typ_open.close()
        except IndexError:
            testercursor = con.cursor()
            str_to_execute = "DELETE FROM dietfriend_client_food_data WHERE username = \'"+cuser+"\' AND date = \'" + str(
                datetime.datetime.now())[0:10] + "\'"
            testercursor.execute(str_to_execute)
            testercursor.execute("COMMIT")


def b_remakeifmade():
    global user
    global con
    cursorp = con.cursor()
    if row_exists_business(cursor=cursorp, username=user):
        query = sql.SQL(
            "SELECT dietfriend_business_food_data.fulltextfile_t, dietfriend_business_food_data.esttextfile_e, dietfriend_business_food_data.individualstextfile_r, dietfriend_business_food_data.typefile_tx_ex_rx_ FROM dietfriend_business_food_data WHERE (username = \'" + user + "\')")
        print("Query: ")
        print(query)
        cursorp.execute(query)
        alldata = cursorp.fetchall()
        print("All Data:")
        print(alldata)
        alldatafixed = fixalldata(alldata)
        ftf_t = alldatafixed[0]
        etf_e = alldatafixed[1]
        itf_r = alldatafixed[2]
        tf_txexrx = alldatafixed[3]
        print("ftf_t")
        print(ftf_t)
        print("etf_e")
        print(etf_e)
        print("itf_r")
        print(itf_r)
        print("tf_txexrx")
        print(tf_txexrx)
        full = b_defineifmissing()
        est = b_defineifmissingestref()
        ind = b_defineifmissingindividuals()
        typ = b_defineifmissingtype()
        with open(full, 'w') as full_open:
            full_open.write(ftf_t)
            full_open.close()
        with open(est, 'w') as est_open:
            est_open.write(etf_e)
            est_open.close()
        with open(ind, 'w') as ind_open:
            ind_open.write(itf_r)
            ind_open.close()
        with open(typ, 'w') as typ_open:
            typ_open.write(tf_txexrx)
            typ_open.close()


def negativechecker(cd, cdusedfoodsincount):
    r = defineifmissing(cd)
    print("START Negative")
    with open(r, 'r') as y:
        y_lines = y.readlines()
        y.close()
    d = 0
    print("len(y_lines):")
    print(len(y_lines))
    print(cd.usedfoodsincount)
    while d < len(y_lines):
        e = 0
        while e < len(y_lines[d])-5:
            s = y_lines[d]
            if s[e:e+1] == '-' and s[e+1:e+2] == '0':
                f = s[e:len(s)].find(' ')
                if f == -1:
                    f = s[e:len(s)].find('\n')
                p = stringcount(s, ' ', e)
                print("s, f, p")
                print(str(s) + ", " + str(f) + ", " + str(p))
                cd.totalsfoodlist[p-2] -= float(s[e:e+f])
                print(cd.totalsfoodlist[p-2])
                print("-=")
                print(s[e:e+f])
                print("setattr")
                print(cd.usedfoodsincount[d])
                print(foodattrnamelst[p-2])
                setattr(cd.usedfoodsincount[d], foodattrnamelst[p-2], 0.0)
                s = s[0:e] + "0.0" + s[f+e:len(s)]
                y_lines[d] = s
            elif s[e:e+1] == '-' and (((stringcount(s, ' ', e) == 20 and s[e:len(s)][0:s[e:len(s)].find('\n')] != "-2000") or (stringcount(s, ' ', e) < 20 and s[e:len(s)][0:s[e:len(s)].find(' ')] != "-2000")) or s[e:e+5] != "-2000"):
                f = s[e:len(s)].find(' ')
                if f == -1:
                    f = s[e:len(s)].find('\n')
                p = stringcount(s, ' ', e)
                print("s, f, p")
                print(str(s) + ", " + str(f) + ", " + str(p))
                cd.totalsfoodlist[p - 2] -= float(s[e:e+f])
                print(cd.totalsfoodlist[p - 2])
                print("-=")
                print(s[e:e + f])
                print("setattr")
                print(cd.usedfoodsincount[d])
                print(foodattrnamelst[p - 2])
                setattr(cd.usedfoodsincount[d], foodattrnamelst[p - 2], -2000)
                s = s[0:e] + "-2000" + s[f + e:len(s)]
                y_lines[d] = s
            else:
                pass
            e += 1

        # try:
        #     while y_lines[d].find('-') != -1 and (y_lines[d][y_lines[d].find('-')+1:y_lines[d].find('-')+5] != "2000"):
        #         negstart = y_lines[d].find('-')
        #         spacesover = y_lines[d][negstart:len(y_lines[d])].find(' ')
        #         curvalue = y_lines[d][negstart:negstart+spacesover]
        #         print("f")
        #         if (0.0 + float(curvalue)) != 0.0:
        #             p = stringcount(y_lines[d], ' ', negstart)
        #             cd.totalsfoodlist[p-2] -= float(curvalue)
        #             setattr(cd.usedfoodsincount[d], foodattrnamelst[p-2], -2000)
        #             y_lines[d] = y_lines[d][0:negstart] + "-2000" + y_lines[d][negstart+spacesover:len(y_lines[d])]
        #             print("g")
        #         else:
        #             p = stringcount(y_lines[d], ' ', negstart)
        #             cd.totalsfoodlist[p - 2] -= float(curvalue)
        #             setattr(cd.usedfoodsincount[d], foodattrnamelst[p - 2], -2000)
        #             y_lines[d] = y_lines[d][0:negstart] + "0.0" + y_lines[d][negstart + spacesover:len(y_lines[d])]
        #             print("gg")
        # except IndexError:
        #     pass
        d += 1
    print("Done negative")
    fixednegstr = ""
    for h in y_lines:
        fixednegstr += str(h)
    with open(r, 'w') as y_write:
        y_write.write(fixednegstr)
        y_write.close()


def stringcount(strg, substrg, findex):
    print("stringcount")
    count = 0
    t = 0
    while t < findex:
        if strg[t:t+1] == substrg:
            count += 1
        print(count)
        t += 1
    return count


def rev(lst):
    i = len(lst) - 1
    nlst = []
    while i >= 0:
        nlst.append(lst[i])
        i -= 1
    print(nlst)
    return nlst


# For before getimgs() in selectimg() = function(): select image and upload to internal path


def get_date_taken(path):
    im = PIL.Image.open(path)
    exif = im.getexif()
    creation_time = exif.get(36867)
    return creation_time


# import json
# from PIL import Image, ExifTags
# from datetime import datetime
#
# def main(filename):
#     image_exif = PIL.Image.open(filename)._getexif()
#     if image_exif:
#         # Make a map with tag names
#         exif = { ExifTags.TAGS[k]: v for k, v in image_exif.items() if k in ExifTags.TAGS and type(v) is not bytes }
#         print(json.dumps(exif, indent=4))
#         # Grab the date
#         date_obj = datetime.strptime(exif['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
#         print(date_obj)
#     else:
#         print('Unable to get date from exif for %s' % filename)


def getimgs():
    global universal_list
    global user
    internalpath = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
    if not os.path.exists(internalpath):
        os.makedirs(internalpath)
    img_list = listdir(internalpath)
    print(img_list)
    # img_list = rev(img_list)
    # # return [[imglist], [nmlst], [numservlst]]
    universal_list[0] = img_list
    print("Universal List:")
    print(universal_list[0])
    return universal_list


def getbimgs():
    global universal_list
    binternalpath = os.path.join(os.path.dirname(__file__), "B_DietFriend_Pictures")
    img_list = listdir(binternalpath)
    print(img_list)
    # img_list = rev(img_list)
    # # return [[imglist], [nmlst], [numservlst]]
    universal_list[0] = img_list
    print("Universal List:")
    print(universal_list[0])
    return universal_list


def doprocess(getimgslst):
    global loggedin
    loggedin = True
    global user
    global con
    global cleared
    cleared = False
    # testcursor = con.cursor()
    # ####
    # str_to_execute = "DELETE FROM dietfriend_client_food_data WHERE username = \'test18\' AND date = \'2022-08-04\'"
    # # str_to_execute = "DELETE FROM client_settings WHERE username = \'test18\'"
    # # str_to_execute = "DELETE FROM dietfriend_usernames_and_passwords_business WHERE username = \'btest1\'"
    # print(str_to_execute)
    # str_to_execute = "DELETE FROM client_moreinfo_value_storage WHERE username = \'test18\' AND datetime = \'2022-08-03\'"
    # testcursor.execute(str_to_execute)
    # testcursor.execute("COMMIT")

    # """BUSINESS"""
    # testcursor = con.cursor()
    # ####
    # str_to_execute = "DELETE FROM dietfriend_business_food_data WHERE username = \'btest50\'"
    # # str_to_execute = "DELETE FROM client_settings WHERE username = \'test18\'"
    # # str_to_execute = "DELETE FROM dietfriend_usernames_and_passwords_business WHERE username = \'btest1\'"
    # print(str_to_execute)
    # testcursor.execute(str_to_execute)
    # testcursor.execute("COMMIT")
    # """#"""

    internalpath = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
    """Convert IMGs from list to String"""
    datetimefortoday = datetime.datetime.now()
    crday = Day(datetimefortoday, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [],
                [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], getrandomid())
    datestring = crday.date_time
    datestring = str(datestring)
    datestring = datestring[0:10]
    remakeifmade(crday, user, datestring)
    crtxt = defineifmissing(crday)
    if cleared:
        getimgslst = [[], [], []]
    numserv = readnumservings(crday)
    typer = readtypes(crday)
    fromtyper = defineifmissingtype(crday)
    est_ref_food_list_of_food_attributes = getest_ref_food_list_of_food_attributes(crday)
    customtxt = defineifmissingindividuals(crday)
    with open(customtxt, 'r') as cust:
        custstr = cust.readlines()
        cust.close()
    custindividuals = gettercustom(custstr)
    with open(crtxt, 'r') as f:
        lcurentered = f.readlines()
        crday.usedfoodsincount = getter(lcurentered)
        print('getter: ')
        print(crday.usedfoodsincount)
        print(len(crday.usedfoodsincount))
        # Add rejects list
        print('numserv: ')
        print(numserv)
        addfoodst(crday, crday.usedfoodsincount, numserv, typer, est_ref_food_list_of_food_attributes, custindividuals)
        f.close()
    imga = numfoodsenteredsofar(crday)
    if len(getimgslst[0]) < imga:
        imga = 0
    print('cr: ')
    print(crday.usedfoodsincount)
    print(imga)
    print(getimgslst[0])
    while imga < len(getimgslst[0]):
        print("Current IMG:")
        print(getimgslst[0][imga])
        print(getimgslst[1][imga])
        print(getimgslst[2][imga])
        numserv.append(getimgslst[2][imga])
        count = 0
        for filename in glob.glob(internalpath):
            if count == 0:
                PIL.Image.open(filename + '\\' + getimgslst[0][imga])
                print("glob.glob IMG:")
                print(filename + '\\' + getimgslst[0][imga])
                count += 1
        imgb = './DietFriend_Pictures'+user+'/' + getimgslst[0][imga]
        f = internalpath + '\\' + getimgslst[0][imga]
        text = recognize_text(imgb, f)
        ftext = onlystrings(text)
        ltext = onlystringstostring(ftext)
        fatext = autocorrect(for_ltext=ltext)
        # Increase accuracy of ltext using autocorrect functions
        fdfrappnd = decrypt(fatext)
        print(fatext)
        standardize(crday)
        crday.usedfoodsincount.append(fdfrappnd)
        crday.usedfoodsincount[imga].food_name = getimgslst[1][imga]
        addfoods(crday, crday.usedfoodsincount[len(crday.usedfoodsincount) - 1], numserv[imga])
        print(crday.usedfoodsincount)
        imga += 1
    i = 0
    while i < len(crday.totalsfoodlist):
        print(i)
        print(crday.totalsfoodlist[i])
        i += 1
    crday.usedfoodsincount = checkforduplicatesandremove(crday)
    print("crday.usedfoodsincount")
    print(crday.usedfoodsincount)
    crday.usedfoodsincount = returncorrectusedfoods(crday, crday.usedfoodsincount)
    print("crday.usedfoodsincount")
    print(crday.usedfoodsincount)
    negativechecker(crday, crday.usedfoodsincount)
    esttext = defineifmissingestref(crday)
    with open(crtxt, 'r') as readfull:
        fulltextlist = readfull.readlines()
        print("fulltextlist: ")
        readfull.close()
    print(fulltextlist)
    with open(esttext, 'r') as readest:
        esttextlist = readest.readlines()
        print("esttextlist: ")
        readest.close()
    print(esttextlist)
    with open(customtxt, 'r') as readindividuals:
        custtextlist = readindividuals.readlines()
        print("individualstextlist: ")
        readindividuals.close()
    print(custtextlist)
    with open(fromtyper, 'r') as readtype:
        typelist = readtype.readlines()
        print("typelist: ")
        readtype.close()
    print(typelist)
    fulltext = ""
    u = 0
    while u < len(fulltextlist):
        fulltext += fulltextlist[u]
        u += 1
    esttxt = ""
    u = 0
    while u < len(esttextlist):
        esttxt += esttextlist[u]
        u += 1
    individualstext = ""
    u = 0
    while u < len(custtextlist):
        individualstext += custtextlist[u]
        u += 1
    typetext = ""
    u = 0
    while u < len(typelist):
        typetext += typelist[u]
        u += 1
    cur = con.cursor()
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET fulltextfile_t = \'" + fulltext + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET esttextfile_e = \'" + esttxt + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET individualstextfile_r = \'" + individualstext + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET typefile_tx_ex_rx_ = \'" + typetext + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    return crday


def dobprocess(getimgslst):
    global loggedin
    loggedin = False
    global user
    global con

    # BUSINESS: DOES NOT WORK DUE TO onloadbfdl()
    # testcursor = con.cursor()
    # ####
    # str_to_execute = "DELETE FROM dietfriend_business_food_data WHERE username = \'btest50\'"
    # # str_to_execute = "DELETE FROM client_settings WHERE username = \'test18\'"
    # # str_to_execute = "DELETE FROM dietfriend_usernames_and_passwords_business WHERE username = \'btest1\'"
    # print(str_to_execute)
    # testcursor.execute(str_to_execute)
    # testcursor.execute("COMMIT")

    binternalpath = os.path.join(os.path.dirname(__file__), "B_DietFriend_Pictures")
    """Convert IMGs from list to String"""
    crday = Day(user, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [],
                [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], getrandomid())
    b_remakeifmade()
    crtxt = b_defineifmissing()
    numserv = b_readnumservings()
    typer = b_readtypes()
    fromtyper = b_defineifmissingtype()
    est_ref_food_list_of_food_attributes = b_getest_ref_food_list_of_food_attributes()
    customtxt = b_defineifmissingindividuals()
    with open(customtxt, 'r') as cust:
        custstr = cust.readlines()
        cust.close()
    custindividuals = gettercustom(custstr)
    with open(crtxt, 'r') as f:
        lcurentered = f.readlines()
        crday.usedfoodsincount = getter(lcurentered)
        print('getter: ')
        print(crday.usedfoodsincount)
        print(len(crday.usedfoodsincount))
        # Add rejects list
        print('numserv: ')
        print(numserv)
        addfoodst(crday, crday.usedfoodsincount, numserv, typer, est_ref_food_list_of_food_attributes, custindividuals)
        f.close()
    imga = b_numfoodsenteredsofar(crday)
    print('cr: ')
    print(crday.usedfoodsincount)
    print(imga)
    print(getimgslst[0])
    while imga < len(getimgslst[0]):
        print("Current IMG:")
        print(getimgslst[0][imga])
        print(getimgslst[1][imga])
        print(getimgslst[2][imga])
        numserv.append(getimgslst[2][imga])
        count = 0
        for filename in glob.glob(binternalpath):
            if count == 0:
                PIL.Image.open(filename + '\\' + getimgslst[0][imga])
                print("glob.glob IMG:")
                print(filename + '\\' + getimgslst[0][imga])
                count += 1
        imgb = './B_DietFriend_Pictures/' + getimgslst[0][imga]
        text = recognize_text(imgb)
        ftext = onlystrings(text)
        ltext = onlystringstostring(ftext)
        fatext = autocorrect(for_ltext=ltext)
        # Increase accuracy of ltext using autocorrect functions
        fdfrappnd = decrypt(fatext)
        print(fatext)
        b_standardize()
        crday.usedfoodsincount.append(fdfrappnd)
        crday.usedfoodsincount[imga].food_name = getimgslst[1][imga]
        b_addfoods(crday, crday.usedfoodsincount[len(crday.usedfoodsincount) - 1], numserv[imga])
        print(crday.usedfoodsincount)
        imga += 1
    i = 0
    while i < len(crday.totalsfoodlist):
        print(i)
        print(crday.totalsfoodlist[i])
        i += 1
    crday.usedfoodsincount = b_checkforduplicatesandremove(crday)
    crday.usedfoodsincount = b_returncorrectusedfoods(crday.usedfoodsincount)
    esttext = b_defineifmissingestref()
    with open(crtxt, 'r') as readfull:
        fulltextlist = readfull.readlines()
        print("fulltextlist: ")
        readfull.close()
    print(fulltextlist)
    with open(esttext, 'r') as readest:
        esttextlist = readest.readlines()
        print("esttextlist: ")
        readest.close()
    print(esttextlist)
    with open(customtxt, 'r') as readindividuals:
        custtextlist = readindividuals.readlines()
        print("individualstextlist: ")
        readindividuals.close()
    print(custtextlist)
    with open(fromtyper, 'r') as readtype:
        typelist = readtype.readlines()
        print("typelist: ")
        readtype.close()
    print(typelist)
    fulltext = ""
    u = 0
    while u < len(fulltextlist):
        fulltext += fulltextlist[u]
        u += 1
    esttxt = ""
    u = 0
    while u < len(esttextlist):
        esttxt += esttextlist[u]
        u += 1
    individualstext = ""
    u = 0
    while u < len(custtextlist):
        individualstext += custtextlist[u]
        u += 1
    typetext = ""
    u = 0
    while u < len(typelist):
        typetext += typelist[u]
        u += 1
    cur = con.cursor()
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET fulltextfile_t = \'" + fulltext + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET esttextfile_e = \'" + esttxt + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET individualstextfile_r = \'" + individualstext + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET typefile_tx_ex_rx_ = \'" + typetext + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    cur.execute(str_to_execute)
    cur.execute("COMMIT")
    return crday


def business_doprocess():
    return dobprocess(getbimgs())


def searchindatabase(fud):
    with open('fooddatabase.txt', 'r') as r:
        rdlnes = r.readlines()
        db = getter(rdlnes)
        r.close()
    firstlayer = 0
    while firstlayer < len(db):
        if db[firstlayer].food_name == fud:
            tpfrrt = (firstlayer, db[firstlayer])
            return tpfrrt
        firstlayer += 1
    tpfrrt = (-1, '')
    return tpfrrt


def estimate(fuud):
    num = searchindatabase(fuud.food_name)
    nfuud = Food(fuud.food_name, fuud.food_datetime, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    if num[0] != -1:
        nfuud.serving = num[1].serving
        nfuud.calories = num[1].calories
        nfuud.total_fat = num[1].total_fat
        nfuud.saturated_fat = num[1].saturated_fat
        nfuud.trans_fat = num[1].trans_fat
        nfuud.cholesterol = num[1].cholesterol
        nfuud.sodium = num[1].sodium
        nfuud.total_carb = num[1].total_carb
        nfuud.fiber = num[1].fiber
        nfuud.total_sugars = num[1].total_sugars
        nfuud.added_sugars = num[1].added_sugars
        nfuud.protein = num[1].protein
        nfuud.calcium = num[1].calcium
        nfuud.iron = num[1].iron
        nfuud.potassium = num[1].potassium
        nfuud.vitamin_a = num[1].vitamin_a
        nfuud.vitamin_b = num[1].vitamin_b
        nfuud.vitamin_c = num[1].vitamin_c
        nfuud.vitamin_d = num[1].vitamin_d
    return nfuud


def datafromestimatednormal(nfile):
    with open(nfile, 'r') as nfle:
        nlst = nfle.readlines()
        nfle.close()
    nlstn = getter(nlst)
    fnum = 0
    while fnum < len(nlstn):
        nlstn[fnum] = estimate(nlstn[fnum])
        fnum += 1
    return nlstn


def estdec():
    crday = doprocess(getimgs())
    normalfile = defineifmissing(crday)
    txtfile = defineifmissingestref(crday)
    with open(txtfile, 'r') as rtxt:
        forlen = rtxt.readlines()
        rtxt.close()
    fdnlstn = datafromestimatednormal(normalfile)
    fd = len(forlen)
    while fd < len(fdnlstn):
        addfoodtotxt(fdnlstn[fd], txtfile=txtfile)
        fd += 1
    return [crday, fdnlstn, txtfile]


def b_estdec():
    crday = business_doprocess()
    normalfile = b_defineifmissing()
    txtfile = b_defineifmissingestref()
    with open(txtfile, 'r') as rtxt:
        forlen = rtxt.readlines()
        rtxt.close()
    fdnlstn = datafromestimatednormal(normalfile)
    fd = len(forlen)
    while fd < len(fdnlstn):
        addfoodtotxt(fdnlstn[fd], txtfile=txtfile)
        fd += 1
    return [crday, fdnlstn, txtfile]


def partialestref():
    v = estdec()
    crday = v[0]
    fdnlstn = v[1]
    print(fdnlstn)
    typefile = defineifmissingtype(crday)
    with open(typefile, 'r') as ic:
        tflst = ic.readlines()
        ic.close()
    let = []
    s = 0
    while s < len(tflst):
        let.append(tflst[s][0:tflst[s].find(',')])
        s += 1
    i = 0
    while i < len(tflst):
        editedtflst = 'e' + tflst[i][tflst[i].find(','):len(tflst[i])]
        writetoline(crday, i, editedtflst)
        i += 1
    curfds = []
    xx = 0
    while xx < len(crday.usedfoodsincount):
        nfdxx = Food(crday.usedfoodsincount[xx].food_name, crday.usedfoodsincount[xx].food_datetime,
                     crday.usedfoodsincount[xx].serving, crday.usedfoodsincount[xx].calories,
                     crday.usedfoodsincount[xx].total_fat, crday.usedfoodsincount[xx].saturated_fat,
                     crday.usedfoodsincount[xx].trans_fat, crday.usedfoodsincount[xx].cholesterol,
                     crday.usedfoodsincount[xx].sodium, crday.usedfoodsincount[xx].total_carb,
                     crday.usedfoodsincount[xx].fiber, crday.usedfoodsincount[xx].total_sugars,
                     crday.usedfoodsincount[xx].added_sugars, crday.usedfoodsincount[xx].protein,
                     crday.usedfoodsincount[xx].calcium, crday.usedfoodsincount[xx].iron,
                     crday.usedfoodsincount[xx].potassium, crday.usedfoodsincount[xx].vitamin_a,
                     crday.usedfoodsincount[xx].vitamin_b, crday.usedfoodsincount[xx].vitamin_c,
                     crday.usedfoodsincount[xx].vitamin_d)
        curfds.append(nfdxx)
        xx += 1
    y = 0
    while y < len(crday.totalsfoodlist):
        crday.totalsfoodlist[y] = 0
        y += 1
    yy = 0
    while yy < len(crday.usedfoodsincount):
        crday.usedfoodsincount[yy].serving = fdnlstn[yy].serving
        crday.usedfoodsincount[yy].calories = fdnlstn[yy].calories
        crday.usedfoodsincount[yy].total_fat = fdnlstn[yy].total_fat
        crday.usedfoodsincount[yy].saturated_fat = fdnlstn[yy].saturated_fat
        print("crday.usedfoodsincount[yy].saturated_fat")
        print(crday.usedfoodsincount[yy].saturated_fat)
        print(fdnlstn[yy].saturated_fat)
        crday.usedfoodsincount[yy].trans_fat = fdnlstn[yy].trans_fat
        crday.usedfoodsincount[yy].cholesterol = fdnlstn[yy].cholesterol
        crday.usedfoodsincount[yy].sodium = fdnlstn[yy].sodium
        crday.usedfoodsincount[yy].total_carb = fdnlstn[yy].total_carb
        crday.usedfoodsincount[yy].fiber = fdnlstn[yy].fiber
        crday.usedfoodsincount[yy].total_sugars = fdnlstn[yy].total_sugars
        crday.usedfoodsincount[yy].added_sugars = fdnlstn[yy].added_sugars
        crday.usedfoodsincount[yy].protein = fdnlstn[yy].protein
        crday.usedfoodsincount[yy].calcium = fdnlstn[yy].calcium
        crday.usedfoodsincount[yy].iron = fdnlstn[yy].iron
        crday.usedfoodsincount[yy].potassium = fdnlstn[yy].potassium
        crday.usedfoodsincount[yy].vitamin_a = fdnlstn[yy].vitamin_a
        crday.usedfoodsincount[yy].vitamin_b = fdnlstn[yy].vitamin_b
        crday.usedfoodsincount[yy].vitamin_c = fdnlstn[yy].vitamin_c
        crday.usedfoodsincount[yy].vitamin_d = fdnlstn[yy].vitamin_d
        yy += 1
    numserv = readnumservings(crday)
    typer = readtypes(crday)
    individualsfile = defineifmissingindividuals(crday)
    with open(individualsfile, 'r') as rin:
        indivstr = rin.readlines()
        rin.close()
    indkst = gettercustom(indivstr)
    addfoodst(crday, crday.usedfoodsincount, numserv, typer, fdnlstn, indkst)
    datestring = crday.date_time
    datestring = str(datestring)
    datestring = datestring[0:10]
    esttext = defineifmissingestref(crday)
    with open(esttext, 'r') as readest:
        esttextlist = readest.readlines()
        print("esttextlist: ")
        readest.close()
    esttxt = ""
    u = 0
    while u < len(esttextlist):
        esttxt += esttextlist[u]
        u += 1
    global con
    estcur = con.cursor()
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET esttextfile_e = \'" + esttxt + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    estcur.execute(str_to_execute)
    estcur.execute("COMMIT")
    y = 0
    while y < len(crday.totalsfoodlist):
        crday.totalsfoodlist[y] = 0
        y += 1
    y = 0
    while y < len(curfds):
        crday.usedfoodsincount[y] = curfds[y]
        y += 1
    with open(typefile, 'r') as ic:
        tflst = ic.readlines()
        ic.close()
    i = 0
    while i < len(tflst):
        editedtflst = let[i] + tflst[i][tflst[i].find(','):len(tflst[i])]
        writetoline(crday, i, editedtflst)
        i += 1
    print("READY FOR PARTIAL REVERSION")
    print("crday.usedfoodsincount:")
    print(crday.usedfoodsincount)
    addfoodst(crday, crday.usedfoodsincount, numserv, typer, fdnlstn, indkst)
    return crday


def b_partialestref():
    v = b_estdec()
    crday = v[0]
    fdnlstn = v[1]
    print(fdnlstn)
    typefile = b_defineifmissingtype()
    with open(typefile, 'r') as ic:
        tflst = ic.readlines()
        ic.close()
    let = []
    s = 0
    while s < len(tflst):
        let.append(tflst[s][0:tflst[s].find(',')])
        s += 1
    i = 0
    while i < len(tflst):
        editedtflst = 'e' + tflst[i][tflst[i].find(','):len(tflst[i])]
        b_writetoline(i, editedtflst)
        i += 1
    curfds = []
    xx = 0
    print("b_partialestref crday.usedfoodsincount")
    print(crday.usedfoodsincount)
    while xx < len(crday.usedfoodsincount):
        nfdxx = Food(crday.usedfoodsincount[xx].food_name, crday.usedfoodsincount[xx].food_datetime,
                     crday.usedfoodsincount[xx].serving, crday.usedfoodsincount[xx].calories,
                     crday.usedfoodsincount[xx].total_fat, crday.usedfoodsincount[xx].saturated_fat,
                     crday.usedfoodsincount[xx].trans_fat, crday.usedfoodsincount[xx].cholesterol,
                     crday.usedfoodsincount[xx].sodium, crday.usedfoodsincount[xx].total_carb,
                     crday.usedfoodsincount[xx].fiber, crday.usedfoodsincount[xx].total_sugars,
                     crday.usedfoodsincount[xx].added_sugars, crday.usedfoodsincount[xx].protein,
                     crday.usedfoodsincount[xx].calcium, crday.usedfoodsincount[xx].iron,
                     crday.usedfoodsincount[xx].potassium, crday.usedfoodsincount[xx].vitamin_a,
                     crday.usedfoodsincount[xx].vitamin_b, crday.usedfoodsincount[xx].vitamin_c,
                     crday.usedfoodsincount[xx].vitamin_d)
        curfds.append(nfdxx)
        xx += 1
    y = 0
    while y < len(crday.totalsfoodlist):
        crday.totalsfoodlist[y] = 0
        y += 1
    yy = 0
    while yy < len(crday.usedfoodsincount):
        crday.usedfoodsincount[yy].serving = fdnlstn[yy].serving
        crday.usedfoodsincount[yy].calories = fdnlstn[yy].calories
        crday.usedfoodsincount[yy].total_fat = fdnlstn[yy].total_fat
        crday.usedfoodsincount[yy].saturated_fat = fdnlstn[yy].saturated_fat
        print("crday.usedfoodsincount[yy].saturated_fat")
        print(crday.usedfoodsincount[yy].saturated_fat)
        print(fdnlstn[yy].saturated_fat)
        crday.usedfoodsincount[yy].trans_fat = fdnlstn[yy].trans_fat
        crday.usedfoodsincount[yy].cholesterol = fdnlstn[yy].cholesterol
        crday.usedfoodsincount[yy].sodium = fdnlstn[yy].sodium
        crday.usedfoodsincount[yy].total_carb = fdnlstn[yy].total_carb
        crday.usedfoodsincount[yy].fiber = fdnlstn[yy].fiber
        crday.usedfoodsincount[yy].total_sugars = fdnlstn[yy].total_sugars
        crday.usedfoodsincount[yy].added_sugars = fdnlstn[yy].added_sugars
        crday.usedfoodsincount[yy].protein = fdnlstn[yy].protein
        crday.usedfoodsincount[yy].calcium = fdnlstn[yy].calcium
        crday.usedfoodsincount[yy].iron = fdnlstn[yy].iron
        crday.usedfoodsincount[yy].potassium = fdnlstn[yy].potassium
        crday.usedfoodsincount[yy].vitamin_a = fdnlstn[yy].vitamin_a
        crday.usedfoodsincount[yy].vitamin_b = fdnlstn[yy].vitamin_b
        crday.usedfoodsincount[yy].vitamin_c = fdnlstn[yy].vitamin_c
        crday.usedfoodsincount[yy].vitamin_d = fdnlstn[yy].vitamin_d
        yy += 1
    numserv = b_readnumservings()
    typer = b_readtypes()
    individualsfile = b_defineifmissingindividuals()
    with open(individualsfile, 'r') as rin:
        indivstr = rin.readlines()
        rin.close()
    indkst = gettercustom(indivstr)
    addfoodst(crday, crday.usedfoodsincount, numserv, typer, fdnlstn, indkst)
    esttext = b_defineifmissingestref()
    with open(esttext, 'r') as readest:
        esttextlist = readest.readlines()
        print("esttextlist: ")
        readest.close()
    esttxt = ""
    u = 0
    while u < len(esttextlist):
        esttxt += esttextlist[u]
        u += 1
    global con
    estcur = con.cursor()
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET esttextfile_e = \'" + esttxt + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    estcur.execute(str_to_execute)
    estcur.execute("COMMIT")
    y = 0
    while y < len(crday.totalsfoodlist):
        crday.totalsfoodlist[y] = 0
        y += 1
    y = 0
    while y < len(curfds):
        crday.usedfoodsincount[y] = curfds[y]
        y += 1
    with open(typefile, 'r') as ic:
        tflst = ic.readlines()
        ic.close()
    i = 0
    while i < len(tflst):
        editedtflst = let[i] + tflst[i][tflst[i].find(','):len(tflst[i])]
        b_writetoline(i, editedtflst)
        i += 1
    print("READY FOR PARTIAL REVERSION")
    print("crday.usedfoodsincount:")
    print(crday.usedfoodsincount)
    addfoodst(crday, crday.usedfoodsincount, numserv, typer, fdnlstn, indkst)
    return crday


def doprocessestref():
    v = estdec()
    crday = v[0]
    fdnlstn = v[1]
    typefile = defineifmissingtype(crday)
    individualsfile = defineifmissingindividuals(crday)
    with open(individualsfile, 'r') as rin:
        indivstr = rin.readlines()
        rin.close()
    indkst = gettercustom(indivstr)
    with open(typefile, 'r') as ic:
        tflst = ic.readlines()
        ic.close()
    i = 0
    """ i then <---> i+1 now"""
    while i < len(tflst):
        editedtflst = 'e' + tflst[i][tflst[i].find(','):len(tflst[i])]
        writetoline(crday, i + 1, editedtflst)
        i += 1
    y = 0
    while y < len(crday.totalsfoodlist):
        crday.totalsfoodlist[y] = 0
        y += 1
    yy = 0
    while yy < len(crday.usedfoodsincount):
        crday.usedfoodsincount[yy].serving = fdnlstn[yy].serving
        crday.usedfoodsincount[yy].calories = fdnlstn[yy].calories
        crday.usedfoodsincount[yy].total_fat = fdnlstn[yy].total_fat
        crday.usedfoodsincount[yy].saturated_fat = fdnlstn[yy].saturated_fat
        print("crday.usedfoodsincount[yy].saturated_fat")
        print(crday.usedfoodsincount[yy].saturated_fat)
        print(fdnlstn[yy].saturated_fat)
        crday.usedfoodsincount[yy].trans_fat = fdnlstn[yy].trans_fat
        crday.usedfoodsincount[yy].cholesterol = fdnlstn[yy].cholesterol
        crday.usedfoodsincount[yy].sodium = fdnlstn[yy].sodium
        crday.usedfoodsincount[yy].total_carb = fdnlstn[yy].total_carb
        crday.usedfoodsincount[yy].fiber = fdnlstn[yy].fiber
        crday.usedfoodsincount[yy].total_sugars = fdnlstn[yy].total_sugars
        crday.usedfoodsincount[yy].added_sugars = fdnlstn[yy].added_sugars
        crday.usedfoodsincount[yy].protein = fdnlstn[yy].protein
        crday.usedfoodsincount[yy].calcium = fdnlstn[yy].calcium
        crday.usedfoodsincount[yy].iron = fdnlstn[yy].iron
        crday.usedfoodsincount[yy].potassium = fdnlstn[yy].potassium
        crday.usedfoodsincount[yy].vitamin_a = fdnlstn[yy].vitamin_a
        crday.usedfoodsincount[yy].vitamin_b = fdnlstn[yy].vitamin_b
        crday.usedfoodsincount[yy].vitamin_c = fdnlstn[yy].vitamin_c
        crday.usedfoodsincount[yy].vitamin_d = fdnlstn[yy].vitamin_d
        yy += 1
    numserv = readnumservings(crday)
    typer = readtypes(crday)
    addfoodst(crday, crday.usedfoodsincount, numserv, typer, fdnlstn, indkst)
    datestring = crday.date_time
    datestring = str(datestring)
    datestring = datestring[0:10]
    esttext = defineifmissingestref(crday)
    with open(esttext, 'r') as readest:
        esttextlist = readest.readlines()
        print("esttextlist: ")
        readest.close()
    esttxt = ""
    u = 0
    while u < len(esttextlist):
        esttxt += esttextlist[u]
        u += 1
    global con
    estcur = con.cursor()
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET esttextfile_e = \'" + esttxt + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    estcur.execute(str_to_execute)
    estcur.execute("COMMIT")

    typtext = defineifmissingtype(crday)
    with open(typtext, 'r') as readtyp:
        typtextlist = readtyp.readlines()
        print("typtextlist: ")
        readtyp.close()
    typtxt = ""
    u = 0
    while u < len(typtextlist):
        typtxt += typtextlist[u]
        u += 1
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    estcur.execute(str_to_execute)
    estcur.execute("COMMIT")
    estermoreinfos(crday)
    return crday


def b_doprocessestref():
    v = b_estdec()
    crday = v[0]
    fdnlstn = v[1]
    typefile = b_defineifmissingtype()
    individualsfile = b_defineifmissingindividuals()
    with open(individualsfile, 'r') as rin:
        indivstr = rin.readlines()
        rin.close()
    indkst = gettercustom(indivstr)
    with open(typefile, 'r') as ic:
        tflst = ic.readlines()
        ic.close()
    i = 0
    while i < len(tflst):
        editedtflst = 'e' + tflst[i][tflst[i].find(','):len(tflst[i])]
        b_writetoline(i + 1, editedtflst)
        i += 1
    y = 0
    while y < len(crday.totalsfoodlist):
        crday.totalsfoodlist[y] = 0
        y += 1
    yy = 0
    while yy < len(crday.usedfoodsincount):
        crday.usedfoodsincount[yy].serving = fdnlstn[yy].serving
        crday.usedfoodsincount[yy].calories = fdnlstn[yy].calories
        crday.usedfoodsincount[yy].total_fat = fdnlstn[yy].total_fat
        crday.usedfoodsincount[yy].saturated_fat = fdnlstn[yy].saturated_fat
        print("crday.usedfoodsincount[yy].saturated_fat")
        print(crday.usedfoodsincount[yy].saturated_fat)
        print(fdnlstn[yy].saturated_fat)
        crday.usedfoodsincount[yy].trans_fat = fdnlstn[yy].trans_fat
        crday.usedfoodsincount[yy].cholesterol = fdnlstn[yy].cholesterol
        crday.usedfoodsincount[yy].sodium = fdnlstn[yy].sodium
        crday.usedfoodsincount[yy].total_carb = fdnlstn[yy].total_carb
        crday.usedfoodsincount[yy].fiber = fdnlstn[yy].fiber
        crday.usedfoodsincount[yy].total_sugars = fdnlstn[yy].total_sugars
        crday.usedfoodsincount[yy].added_sugars = fdnlstn[yy].added_sugars
        crday.usedfoodsincount[yy].protein = fdnlstn[yy].protein
        crday.usedfoodsincount[yy].calcium = fdnlstn[yy].calcium
        crday.usedfoodsincount[yy].iron = fdnlstn[yy].iron
        crday.usedfoodsincount[yy].potassium = fdnlstn[yy].potassium
        crday.usedfoodsincount[yy].vitamin_a = fdnlstn[yy].vitamin_a
        crday.usedfoodsincount[yy].vitamin_b = fdnlstn[yy].vitamin_b
        crday.usedfoodsincount[yy].vitamin_c = fdnlstn[yy].vitamin_c
        crday.usedfoodsincount[yy].vitamin_d = fdnlstn[yy].vitamin_d
        yy += 1
    numserv = b_readnumservings()
    typer = b_readtypes()
    addfoodst(crday, crday.usedfoodsincount, numserv, typer, fdnlstn, indkst)
    esttext = b_defineifmissingestref()
    with open(esttext, 'r') as readest:
        esttextlist = readest.readlines()
        print("esttextlist: ")
        readest.close()
    esttxt = ""
    u = 0
    while u < len(esttextlist):
        esttxt += esttextlist[u]
        u += 1
    global con
    estcur = con.cursor()
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET esttextfile_e = \'" + esttxt + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    estcur.execute(str_to_execute)
    estcur.execute("COMMIT")

    typtext = b_defineifmissingtype()
    with open(typtext, 'r') as readtyp:
        typtextlist = readtyp.readlines()
        print("typtextlist: ")
        readtyp.close()
    typtxt = ""
    u = 0
    while u < len(typtextlist):
        typtxt += typtextlist[u]
        u += 1
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    estcur.execute(str_to_execute)
    estcur.execute("COMMIT")
    # b_estermoreinfos(crday)
    return crday


def estermoreinfos(cd):
    global con
    global user
    cr = con.cursor()
    i = 1
    while i < len(foodattrnamelst):
        sc = defineifmissingmisessiontwo(cd, foodattrnamelst[i])
        # ##############################################FIX: Needs SELECT from _im_two on database
        with open(sc, 'r') as fiil:
            f = fiil.readlines()
            fiil.close()
        print(f)
        if f == []:
            pass
        else:
            if f[len(f) - 1].find('i') == -1 and f[len(f) - 1].find('x') == -1:
                f.pop(len(f) - 1)
            strtowrt = ""
            u = 0
            while u < len(f):
                b = f[u].replace('x', 'i')
                b = b[0:b.find(',') + 1] + str(getattr(cd.usedfoodsincount[u], foodattrnamelst[i])) + "\n"
                strtowrt += b
                u += 1
            with open(sc, 'w') as fiil:
                fiil.write(strtowrt)
                fiil.close()
            if row_exists_moreinfo(cr, user, str(datetime.datetime.now())[0:10]):
                str_to_execute = "UPDATE client_moreinfo_value_storage SET _im_two" + str(
                    foodattrnamelst[i]) + " = \'" + strtowrt.replace('None', '').replace('\\n',
                                                                                         '\n') + "\' WHERE username = \'" + user + "\' AND datetime = \'" + str(
                    datetime.datetime.now())[0:10] + "\'"
                print(str_to_execute)
                cr.execute(str_to_execute)
                cr.execute("COMMIT")
            else:
                query = sql.SQL(
                    "INSERT INTO client_moreinfo_value_storage (username, datetime, _im_two" + str(
                        foodattrnamelst[i]) + ") VALUES (\'" + user + "\', \'" + str(
                        datetime.datetime.now())[0:10] + "\', \'" + strtowrt + "\')")
                print("Query: ")
                print(query)
                cr.execute(query)
                cr.execute("COMMIT")
        i += 1


def estermoreinfos_focused_basedonlst(cd, lstofchanged, f_ind):
    global con
    global user
    cr = con.cursor()
    i = 0
    while i < len(lstofchanged):
        print("lstofchanged")
        print(lstofchanged)
        sc = defineifmissingmisessiontwo(cd, lstofchanged[i][0])
        # ##############################################FIX: Needs SELECT from _im_two on database
        with open(sc, 'r') as fiil:
            f = fiil.readlines()
            fiil.close()
        print(f)
        if f == []:
            pass
        else:
            if f[len(f) - 1].find('i') == -1 and f[len(f) - 1].find('x') == -1:
                f.pop(len(f) - 1)
            strtowrt = ""
            u = 0
            while u < len(f):
                if u == f_ind:
                    b = f[u].replace('x', 'i')
                    b = b[0:b.find(',') + 1] + str(lstofchanged[i][1]) + "\n"
                else:
                    b = f[u]
                strtowrt += b
                u += 1
            with open(sc, 'w') as fiil:
                fiil.write(strtowrt)
                fiil.close()
            if row_exists_moreinfo(cr, user, str(datetime.datetime.now())[0:10]):
                str_to_execute = "UPDATE client_moreinfo_value_storage SET _im_two" + str(
                    lstofchanged[i][0]) + " = \'" + strtowrt.replace('None', '').replace('\\n',
                                                                                         '\n') + "\' WHERE username = \'" + user + "\' AND datetime = \'" + str(
                    datetime.datetime.now())[0:10] + "\'"
                print(str_to_execute)
                cr.execute(str_to_execute)
                cr.execute("COMMIT")
            else:
                query = sql.SQL(
                    "INSERT INTO client_moreinfo_value_storage (username, datetime, _im_two" + str(
                        lstofchanged[i][0]) + ") VALUES (\'" + user + "\', \'" + str(
                        datetime.datetime.now())[0:10] + "\', \'" + strtowrt + "\')")
                print("Query: ")
                print(query)
                cr.execute(query)
                cr.execute("COMMIT")
        i += 1


# def b_estermoreinfos(cd):
#     pass


def readlinexaftinit(txtfile, linenum):
    with open(txtfile, 'r') as ret:
        rlst = ret.readlines()
        ret.close()
    forfix = rlst[linenum]
    return forfix


def revert(intr, cd):
    datetxt = defineifmissing(cd)
    datetypetxt = defineifmissingtype(cd)
    with open(datetxt, 'r') as re:
        lstrdestref = re.readlines()
        f = getter(lstrdestref)
        re.close()
    cd.usedfoodsincount[intr].calories = f[intr].calories
    cd.usedfoodsincount[intr].total_fat = f[intr].total_fat
    cd.usedfoodsincount[intr].saturated_fat = f[intr].saturated_fat
    cd.usedfoodsincount[intr].trans_fat = f[intr].trans_fat
    cd.usedfoodsincount[intr].cholesterol = f[intr].cholesterol
    cd.usedfoodsincount[intr].sodium = f[intr].sodium
    cd.usedfoodsincount[intr].total_carb = f[intr].total_carb
    cd.usedfoodsincount[intr].fiber = f[intr].fiber
    cd.usedfoodsincount[intr].total_sugars = f[intr].total_sugars
    cd.usedfoodsincount[intr].added_sugars = f[intr].added_sugars
    cd.usedfoodsincount[intr].protein = f[intr].protein
    cd.usedfoodsincount[intr].calcium = f[intr].calcium
    cd.usedfoodsincount[intr].iron = f[intr].iron
    cd.usedfoodsincount[intr].potassium = f[intr].potassium
    cd.usedfoodsincount[intr].vitamin_a = f[intr].vitamin_a
    cd.usedfoodsincount[intr].vitamin_b = f[intr].vitamin_b
    cd.usedfoodsincount[intr].vitamin_c = f[intr].vitamin_c
    cd.usedfoodsincount[intr].vitamin_d = f[intr].vitamin_d
    print('Serving: ')
    print(cd.usedfoodsincount[intr].serving)
    print('Calories: ')
    print(cd.usedfoodsincount[intr].calories)
    print('Total Fat: ')
    print(cd.usedfoodsincount[intr].total_fat)
    print('Saturated Fat: ')
    print(cd.usedfoodsincount[intr].saturated_fat)
    print('Trans Fat: ')
    print(cd.usedfoodsincount[intr].trans_fat)
    print('Cholesterol: ')
    print(cd.usedfoodsincount[intr].cholesterol)
    print('Sodium: ')
    print(cd.usedfoodsincount[intr].sodium)
    print('Total Carb: ')
    print(cd.usedfoodsincount[intr].total_carb)
    print('Fiber: ')
    print(cd.usedfoodsincount[intr].fiber)
    print('Total Sugars: ')
    print(cd.usedfoodsincount[intr].total_sugars)
    print('Added Sugars: ')
    print(cd.usedfoodsincount[intr].added_sugars)
    print('Protein: ')
    print(cd.usedfoodsincount[intr].protein)
    print('Calcium: ')
    print(cd.usedfoodsincount[intr].calcium)
    print('Iron: ')
    print(cd.usedfoodsincount[intr].iron)
    print('Potassium: ')
    print(cd.usedfoodsincount[intr].potassium)
    print('Vitamin A: ')
    print(cd.usedfoodsincount[intr].vitamin_a)
    print('Vitamin B: ')
    print(cd.usedfoodsincount[intr].vitamin_b)
    print('Vitamin C: ')
    print(cd.usedfoodsincount[intr].vitamin_c)
    print('Vitamin D: ')
    print(cd.usedfoodsincount[intr].vitamin_d)
    p = readlinexaftinit(datetypetxt, intr)
    q = 't' + p[p.find(','):len(p)]
    writetoline(cd, intr + 1, q)
    return cd


def b_revert(intr, cd):
    datetxt = b_defineifmissing()
    datetypetxt = b_defineifmissingtype()
    with open(datetxt, 'r') as re:
        lstrdestref = re.readlines()
        f = getter(lstrdestref)
        re.close()
    cd.usedfoodsincount[intr].calories = f[intr].calories
    cd.usedfoodsincount[intr].total_fat = f[intr].total_fat
    cd.usedfoodsincount[intr].saturated_fat = f[intr].saturated_fat
    cd.usedfoodsincount[intr].trans_fat = f[intr].trans_fat
    cd.usedfoodsincount[intr].cholesterol = f[intr].cholesterol
    cd.usedfoodsincount[intr].sodium = f[intr].sodium
    cd.usedfoodsincount[intr].total_carb = f[intr].total_carb
    cd.usedfoodsincount[intr].fiber = f[intr].fiber
    cd.usedfoodsincount[intr].total_sugars = f[intr].total_sugars
    cd.usedfoodsincount[intr].added_sugars = f[intr].added_sugars
    cd.usedfoodsincount[intr].protein = f[intr].protein
    cd.usedfoodsincount[intr].calcium = f[intr].calcium
    cd.usedfoodsincount[intr].iron = f[intr].iron
    cd.usedfoodsincount[intr].potassium = f[intr].potassium
    cd.usedfoodsincount[intr].vitamin_a = f[intr].vitamin_a
    cd.usedfoodsincount[intr].vitamin_b = f[intr].vitamin_b
    cd.usedfoodsincount[intr].vitamin_c = f[intr].vitamin_c
    cd.usedfoodsincount[intr].vitamin_d = f[intr].vitamin_d
    print('Serving: ')
    print(cd.usedfoodsincount[intr].serving)
    print('Calories: ')
    print(cd.usedfoodsincount[intr].calories)
    print('Total Fat: ')
    print(cd.usedfoodsincount[intr].total_fat)
    print('Saturated Fat: ')
    print(cd.usedfoodsincount[intr].saturated_fat)
    print('Trans Fat: ')
    print(cd.usedfoodsincount[intr].trans_fat)
    print('Cholesterol: ')
    print(cd.usedfoodsincount[intr].cholesterol)
    print('Sodium: ')
    print(cd.usedfoodsincount[intr].sodium)
    print('Total Carb: ')
    print(cd.usedfoodsincount[intr].total_carb)
    print('Fiber: ')
    print(cd.usedfoodsincount[intr].fiber)
    print('Total Sugars: ')
    print(cd.usedfoodsincount[intr].total_sugars)
    print('Added Sugars: ')
    print(cd.usedfoodsincount[intr].added_sugars)
    print('Protein: ')
    print(cd.usedfoodsincount[intr].protein)
    print('Calcium: ')
    print(cd.usedfoodsincount[intr].calcium)
    print('Iron: ')
    print(cd.usedfoodsincount[intr].iron)
    print('Potassium: ')
    print(cd.usedfoodsincount[intr].potassium)
    print('Vitamin A: ')
    print(cd.usedfoodsincount[intr].vitamin_a)
    print('Vitamin B: ')
    print(cd.usedfoodsincount[intr].vitamin_b)
    print('Vitamin C: ')
    print(cd.usedfoodsincount[intr].vitamin_c)
    print('Vitamin D: ')
    print(cd.usedfoodsincount[intr].vitamin_d)
    p = readlinexaftinit(datetypetxt, intr)
    q = 't' + p[p.find(','):len(p)]
    b_writetoline(intr + 1, q)
    return cd


def doprocessrevertall():
    cday = doprocessestref()
    x = 0
    while x < len(cday.usedfoodsincount):
        revert(x, cday)
        x += 1
    iy = 0
    while iy < 19:
        cday.totalsfoodlist[iy] = 0
        iy += 1
    nserv = readnumservings(cday)
    typr = readtypes(cday)
    addfoodst(cday, cday.usedfoodsincount, nserv, typr, [], [])
    datestring = cday.date_time
    datestring = str(datestring)
    datestring = datestring[0:10]
    global con
    revertcur = con.cursor()
    typetext = defineifmissingtype(cday)
    with open(typetext, 'r') as readtyp:
        typtextlist = readtyp.readlines()
        print("typtextlist: ")
        readtyp.close()
    typtxt = ""
    u = 0
    while u < len(typtextlist):
        typtxt += typtextlist[u]
        u += 1
    str_to_execute = \
        "UPDATE dietfriend_client_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
        user + "\' AND date = \'" + datestring + "\'"
    print(str_to_execute)
    revertcur.execute(str_to_execute)
    revertcur.execute("COMMIT")
    estermoreinfos(cday)
    return cday


def b_doprocessrevertall():
    cday = b_doprocessestref()
    x = 0
    while x < len(cday.usedfoodsincount):
        b_revert(x, cday)
        x += 1
    iy = 0
    while iy < 19:
        cday.totalsfoodlist[iy] = 0
        iy += 1
    nserv = b_readnumservings()
    typr = b_readtypes()
    addfoodst(cday, cday.usedfoodsincount, nserv, typr, [], [])
    global con
    revertcur = con.cursor()
    typetext = b_defineifmissingtype()
    with open(typetext, 'r') as readtyp:
        typtextlist = readtyp.readlines()
        print("typtextlist: ")
        readtyp.close()
    typtxt = ""
    u = 0
    while u < len(typtextlist):
        typtxt += typtextlist[u]
        u += 1
    str_to_execute = \
        "UPDATE dietfriend_business_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
        user + "\'"
    print(str_to_execute)
    revertcur.execute(str_to_execute)
    revertcur.execute("COMMIT")
    # b_estermoreinfos(cday)
    return cday


def defineifmissinggraphsubject():
    curdayupdtd = str(running_id)
    # curdayfupdtd = fixfilestring(curdayupdtd)
    print("Session_ID4")
    print(curdayupdtd)
    curtxt = curdayupdtd + user + '_gs.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def findstartwords(strofwords):
    wordslst = []
    formanipulation = strofwords
    while formanipulation.find('_') > -1:
        f_pos = formanipulation.find('_')
        curword = formanipulation[0:f_pos]
        wordslst.append(curword)
        formanipulation = formanipulation[f_pos + 1: len(formanipulation)]
    return wordslst


def ranker(fdblstrp, typedthusfar):
    """CHECK: Need to add '__' to end of each food_name to ensure glitch where food_name ends in 'r,' does not interfere
     with 'r,hamburger' ?"""
    print("ranker(")
    print(fdblstrp)
    print(", " + typedthusfar + ")")
    fixedtyped = typedthusfar.replace(" ", "_")
    typedwords = findstartwords(fixedtyped)
    typedwords.append(findendword(fixedtyped))
    rank = 0
    wordn = 0
    countdivided = 0
    if fdblstrp[0].find(fixedtyped) > -1:
        rank += 14
        rank *= 2
    while wordn < len(typedwords):
        p = 0
        if fdblstrp[0].find(typedwords[wordn]) > -1:
            rank += 7
            rank *= 1.75
        while p < len(fdblstrp[1]):
            if fixedtyped != '' and fixedtyped != '_' and fixedtyped != ' ' and fdblstrp[1][p] != '' \
                    and fdblstrp[1][p] != '_' and fdblstrp[1][p] != ' ':
                if fdblstrp[1][p].find(typedwords[wordn]) > -1:
                    print(fdblstrp[1][p])
                    print(".find(" + fixedtyped + ")")
                    print(fdblstrp[1][p].find(fixedtyped) > -1)
                    rank += 1
                    rank *= 1.5
                if fdblstrp[1][p].find(typedwords[len(typedwords) - 1]) > -1:
                    print(fdblstrp[1][p])
                    print(".find(" + typedwords[len(typedwords) - 1] + ")")
                    print(fdblstrp[1][p].find(typedwords[len(typedwords) - 1]) > -1)
                    rank += 1
                if fdblstrp[1][len(fdblstrp[1]) - 1].find(typedwords[len(typedwords) - 1]) > -1:
                    print(fdblstrp[1][len(fdblstrp[1]) - 1])
                    print(".find(" + typedwords[len(typedwords) - 1] + ")")
                    print(fdblstrp[1][len(fdblstrp[1]) - 1].find(typedwords[len(typedwords) - 1]) > -1)
                    rank += 1
                if typedwords[wordn] == fdblstrp[1][p]:
                    print(typedwords[wordn] + " == " + fdblstrp[1][p])
                    print(typedwords[wordn] == fdblstrp[1][p])
                    rank += 7
                else:
                    if countdivided <= 4:
                        rank /= 8
                        countdivided += 1
            p += 1
        wordn += 1
    print(rank)
    print("^ rank ^")
    return rank


def opener():
    with open('fooddatabase.txt', 'r') as fdbtxt:
        fdblst = fdbtxt.readlines()
        fdbtxt.close()
    fdblst.pop(0)
    p = 0
    while p < len(fdblst):
        fdblst[p] = fdblst[p].strip()
        p += 1
    return fdblst


def findendword(strofwords):
    print("findendword")
    tempstrofwords = strofwords
    i = len(strofwords) - 1
    found_f = False
    incword = ''
    while i >= 0 and not found_f:
        if tempstrofwords[i:len(tempstrofwords)].find('_') == 0:
            incword = tempstrofwords[i + 1:len(tempstrofwords)]
            found_f = True
        i -= 1
    print(incword)
    print("^ incword ^")
    return incword


def autofind(curtyped):
    print("curtyped: ")
    print(curtyped)
    fdblstf = opener()
    i = 0
    while i < len(fdblstf):
        fdblstf[i] = fdblstf[i][0:fdblstf[i].find(' ')]
        i += 1
    fdblstr = []
    words = []
    rank = []
    k = 0
    while k < len(fdblstf):
        print("k: ")
        print(k)
        words.append(findstartwords(fdblstf[k]))
        print("words 0")
        print(words)
        words[k].append(findendword(fdblstf[k]))
        print("words 1")
        print(words)
        fdblstr.append([fdblstf[k], words[k]])
        print("fdblstr 2")
        print(fdblstr)
        rank.append(ranker(fdblstr[k], curtyped))
        print("rank 3")
        print(rank)
        fdblstr[k].append(rank[k])
        print("fdblstr 4")
        print(fdblstr)
        k += 1
    print("Full fdblstr")
    print(fdblstr)
    innery = 0
    while innery < len(fdblstr) - 1:
        y = innery + 1
        while y < len(fdblstr):
            if fdblstr[y][2] > fdblstr[innery][2]:
                temp = fdblstr.pop(y)
                fdblstr.insert(innery, temp)
            y += 1
        innery += 1
    print("Sorted fdblstr: ")
    print(fdblstr)
    return fdblstr


def getgraphsubject():
    txt = defineifmissinggraphsubject()
    with open(txt, 'r') as rr:
        try:
            fr = rr.readlines()[0].strip()
        except IndexError:
            fr = 'calories'
        rr.close()
    return fr


def getdate(cday, da):
    date_abase = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    print("B")
    print(cday.date_time)
    le = str(cday.date_time)
    print("C")
    curdate = le[0:le.find(' ') + 1]
    print("D")
    numyear = curdate[0:curdate.find('-')]
    print(numyear)
    numyear = int(numyear)
    print(numyear)
    numday = curdate[8:curdate.find(' ')]
    print(numday)
    numday = int(numday)
    print(numday)
    nummonthstr = curdate[5:len(curdate) - 1]
    print(nummonthstr)
    print("AAAA")
    nummonth = nummonthstr[0:nummonthstr.find('-')]
    print(nummonth)
    nummonth = int(nummonth)
    print(nummonth)
    numday -= da
    if numday < 1:
        nummonth -= 1
        if nummonth == 0:
            nummonth = 12
            numyear -= 1
        finalnumday = date_abase[nummonth - 1] + numday
        numday = finalnumday
    if nummonth < 10 and numday < 10:
        rdatestr = str(numyear) + '-0' + str(nummonth) + '-0' + str(numday)
    elif nummonth < 10:
        rdatestr = str(numyear) + '-0' + str(nummonth) + '-' + str(numday)
    elif numday < 10:
        rdatestr = str(numyear) + '-' + str(nummonth) + '-0' + str(numday)
    else:
        rdatestr = str(numyear) + '-' + str(nummonth) + '-' + str(numday)
    return rdatestr


def defineifmissing_yest_(dt):
    curtxt = dt + user + '.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def defineifmissing_type_date(dt):
    curtxt = dt + user + '_type.txt'
    try:
        fo = open(curtxt, 'r')
        fo.close()
    except FileNotFoundError:
        fm = open(curtxt, 'w')
        fm.close()
    return curtxt


def getxda(crday, daysago, feature):
    date = getdate(crday, daysago)
    flefryest = defineifmissing_yest_(date)
    # flefryestest = defineifmissing_yest_est(date)
    # flefryesttype = defineifmissing_yest_type(date)
    # flefryesttype = defineifmissing_yest_final(date)
    with open(flefryest, 'r') as fryest:
        lstfryst = fryest.readlines()
        fryest.close()
    final = 0
    print('1111')
    # Use finallstfryst insead of lstfryst
    if len(lstfryst) != 0:
        fdlstfryest = getter(lstfryst)
        srvings = getallservingamts(date)
        u = 0
        while u < len(fdlstfryest):
            if getattr(fdlstfryest[u], feature) != -2000:
                final += getattr(fdlstfryest[u], feature) * float(srvings[u])
            print('getattr')
            print(getattr(fdlstfryest[u], feature))
            u += 1
    print('final:')
    print(final)
    return final


def maxcal(crday, feature):
    daysago = 0
    maxd = 0
    while daysago < 7:
        print("HAHA")
        if maxd < getxda(crday, daysago, feature):
            maxd = getxda(crday, daysago, feature)
        daysago += 1
    return maxd


def isinfd(fdlp, fdp, dfd):
    s = 0
    while s < len(fdlp):
        if dfd[fdp].food_name == fdlp[s].strip():
            return True
        s += 1
    return False


def isinbfllw(fdlp, fdp, dfd):
    s = 0
    while s < len(fdlp):
        if dfd[fdp] == fdlp[s].strip():
            return True
        s += 1
    return False


def addfdnametotext(capitalfoodnm, txtfile):
    print('trying')
    try:
        bo = open(txtfile, 'r')
        bo.close()
    except FileNotFoundError:
        bm = open(txtfile, 'w')
        bm.close()
    with open(txtfile, 'a') as a:
        a.write(capitalfoodnm + '\n')
        a.close()


def removefromtext(lineind, file):
    with open(file, 'r') as fr:
        lstt = fr.readlines()
        fr.close()
    lstt.pop(lineind)
    strtowrt = ''
    for i in lstt:
        strtowrt += i
    with open(file, 'w') as fw:
        fw.write(strtowrt)
        fw.close()


# class ChildApp(GridLayout):
#    def __init__(self, **kwargs):
#        super(ChildApp, self).__init__()
# self.cols = 5
# self.add_widget(Label(text='something'))
# self.f_name = TextInput()
# self.add_widget(self.f_name)
#
# self.add_widget(Label(text='something else'))
# self.f_sodium = TextInput()
# self.add_widget(self.f_sodium)
#
# self.add_widget(Label(text='another thing'))
# self.f_potassium = TextInput()
# self.add_widget(self.f_potassium)


class SignInPage(Screen):
    def sign_in(self):
        cuser = self.ids.username.text
        global user
        user = cuser
        password = self.ids.password.text
        global con
        con = psycopg2.connect(
            database="dietfriendcab", user=cuser, password=password, host='127.0.0.1', port='5432'
        )
        cur = con.cursor()
        global theme
        global primary_p
        past = defineifmissing_prev_insecure_settings()
        if row_exists_theme(cur, user):
            query = sql.SQL(
                "SELECT bg_theme FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            theme_setting = str(cur.fetchall())
            print("theme_setting:")
            print(theme_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_theme_index = int(random() * 100) % 2
            if random_theme_index == 0:
                theme = "Light"
            else:
                theme = "Dark"
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        if row_exists_primary_p(cur, user):
            query = sql.SQL(
                "SELECT primary_p FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            primary_p_setting = str(cur.fetchall())
            print("primary_p_setting:")
            print(primary_p_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_primary_p_index = int(random() * 100) % 10
            if random_primary_p_index == 0:
                primary_p = "Teal"
            elif random_primary_p_index == 1:
                primary_p = "Red"
            elif random_primary_p_index == 2:
                primary_p = "Pink"
            elif random_primary_p_index == 3:
                primary_p = "Indigo"
            elif random_primary_p_index == 4:
                primary_p = "Blue"
            elif random_primary_p_index == 5:
                primary_p = "LightBlue"
            elif random_primary_p_index == 6:
                primary_p = "Lime"
            elif random_primary_p_index == 7:
                primary_p = "Yellow"
            elif random_primary_p_index == 8:
                primary_p = "Orange"
            else:
                primary_p = "Amber"
            query = sql.SQL(
                "INSERT INTO client_settings (username, primary_p, bg_theme) VALUES (\'" + user + "\', \'" + primary_p + "\', \'" + theme + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            cur.execute("COMMIT")
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        path = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
        if not os.path.exists(path):
            os.makedirs(path)
        # ADD USERNAME TO ALL 'path's
        global universal_list

        try:
            print("Signed in, constructing universal_list")
            with open(str(datetime.datetime.now())[0:10] + str(user) + ".txt", 'r') as p:
                lines = p.readlines()
                p.close()
            words = []
            for i in lines:
                words.append(i[0:i.find(' ')])
            with open(str(datetime.datetime.now())[0:10] + str(user) + "_type.txt", 'r') as p:
                lines_two = p.readlines()
                p.close()
            numservings = []
            h = 0
            while h < len(lines_two):
                numservings.append(float(lines_two[h][lines_two[h].find(',') + 1:lines_two[h].find('\n')]))
                h += 1
            universal_list = [[], words, numservings]
            print("universal_list: ")
            print(universal_list)
        except:
            universal_list = [[], [], []]
        # UNCOMMENT ABOVE + COMMENT LINE BELOW THIS FOR REAL PRODUCT

        # universal_list = [[], ['pop_tarts', 'turkey_sticks', 'calcium_powder', 'chili_magic', 'planters_peanuts',
        #                        'buddig_beef', 'buddig_beef', 'tuna_can', 'wolf_brand_chili_magic'],
        #                   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.50, 2.00, 1.60]]


class FindFood(Screen):
    def checkrequestloc(self):
        pass


class SetDescriptionPopUp(Popup):
    def sdloader(self):
        self.open()

    def savechangesstop(self):
        self.ids.savechangestext.text = ""

    def savechangesstart(self):
        global con
        global user
        crs = con.cursor()
        if row_exists_bspecialinfo(crs, user):
            str_to_execute = "UPDATE business_special_info SET description = \'" + self.ids.tisdpu.text + "\' WHERE business_name = \'" + \
                             user + "\'"
            print(str_to_execute)
            crs.execute(str_to_execute)
            crs.execute("COMMIT")
        else:
            query = sql.SQL(
                "INSERT INTO business_special_info(description, business_name) VALUES (\'" + self.ids.tisdpu.text + "\', \'" + user + "\')")
            print("Query: ")
            print(query)
            crs.execute(query)
            crs.execute("COMMIT")
        self.ids.savechangestext.text = "Saved"


def newlineadder(strtoaddto, size_hint_x):
    wordsperline = int(size_hint_x * 10)
    counter = 0
    u = 0
    while u < len(strtoaddto):
        if strtoaddto[u:u + 1] == " ":
            counter += 1
        if counter == wordsperline:
            strtoaddto = strtoaddto[0:u] + "\n" + strtoaddto[u + 1:len(strtoaddto)]
            counter = 0
        u += 1
    return strtoaddto


class BusinessDemandInfoPage(Screen):
    def checkforstuff(self):
        global user
        global con
        pcrs = con.cursor()
        if check_exists_icons(pcrs, user):
            query = sql.SQL(
                "SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
            print("Query: ")
            print(query)
            pcrs.execute(query)
            icons = fixalldatabasedonnewline(str(pcrs.fetchall()))
            print(icons)
            names = []
            for icon in icons:
                query = sql.SQL(
                    "SELECT icon_hint FROM icon_colors WHERE (icon_nme = \'" + icon + "\')")
                print("Query: ")
                print(query)
                pcrs.execute(query)
                ichint = str(pcrs.fetchall()).replace(' ', '_').replace('[', '').replace(']', '').replace('(',
                                                                                                          '').replace(
                    ')', '').replace(',', '').replace('\'', '')
                names.append(ichint)
            strtoset = ""
            for name in names:
                strtoset += name + ", "
            strtoset = newlineadder(strtoset, 0.3)
            strtoset = strtoset.replace('_', ' ')
            self.ids.desig.text = strtoset
        if check_exists_description(pcrs, user):
            query = sql.SQL(
                "SELECT description FROM business_special_info WHERE (business_name = \'" + user + "\')")
            print("Query: ")
            print(query)
            pcrs.execute(query)
            description = str(pcrs.fetchall()).replace('[', '').replace(']', '').replace('(', '').replace(')',
                                                                                                          '').replace(
                ',', '').replace('\'', '')
            print(description)
            description = newlineadder(description, 0.5)
            self.ids.descr.text = description
        if check_exists_loc(pcrs, user):
            query = sql.SQL(
                "SELECT lat, lon FROM business_special_info WHERE (business_name = \'" + user + "\')")
            print("Query: ")
            print(query)
            pcrs.execute(query)
            latlon = str(pcrs.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(', '').replace(
                ')', '').replace('\'', '')
            lat = latlon[0:latlon.find(',')]
            lon = latlon[latlon.find(',') + 1:len(latlon)]
            print(lat)
            print(lon)
            self.ids.latlon.text = "Latitude: " + lat + "\nLongitude: " + lon

    def setdescription(self):
        sdpu = SetDescriptionPopUp()
        sdpu.sdloader()


class MapHintPopUp(Popup):
    pass


class BusinessLocationSetter(Screen):
    def showhint(self):
        g = MapHintPopUp()
        g.open()

    def askconfirm(self):
        lat = self.ids.mapper.lat
        lon = self.ids.mapper.lon
        print("LAT")
        print(lat)
        print("LON")
        print(lon)
        n = MapMarkerPopup(lat=lat, lon=lon)
        b = BoxLayout(orientation='vertical')
        l = Label(text="Confirm this as your business location?\n(You will be able to edit this later.)")
        g = GridLayout(cols=2, rows=1)
        btnone = Button(text="Cancel", on_release=lambda p: self.ids.mapper.remove_widget(n))
        btntwo = Button(text="Confirm", on_release=lambda q: self.confirmlocation(str(lat), str(lon)))
        g.add_widget(btnone)
        g.add_widget(btntwo)
        b.add_widget(l)
        b.add_widget(g)
        n.add_widget(b)
        self.ids.mapper.add_widget(n)

    def confirmlocation(self, lat, lon):
        global con
        global user
        lcrsr = con.cursor()
        if row_exists_bspecialinfo(lcrsr, user):
            str_to_execute = "UPDATE business_special_info SET lat = \'" + lat + "\' WHERE business_name = \'" + \
                             user + "\'"
            lcrsr.execute(str_to_execute)
            lcrsr.execute("COMMIT")
            str_to_execute = "UPDATE business_special_info SET lon = \'" + lon + "\' WHERE business_name = \'" + \
                             user + "\'"
            lcrsr.execute(str_to_execute)
            lcrsr.execute("COMMIT")
        else:
            query = sql.SQL(
                "INSERT INTO business_special_info(lat, lon, business_name) VALUES (\'" + lat + "\', \'" + lon + "\', \'" + user + "\')")
            print("Query: ")
            print(query)
            lcrsr.execute(query)
            lcrsr.execute("COMMIT")
        self.manager.current = "businessdemand"


class HintPopUp(Popup):
    def opener(self, icon):
        h = defineifmissinghint()
        with open(h, 'r') as hh:
            p = hh.readlines()
            hh.close()
        self.ids.hinttext.text = p[0].strip()
        self.open()
        self.ids.btnforbind.bind(on_release=lambda d: self.declare_icon(icon=icon))
        print("Bind successful!")

    def declare_icon(self, icon):
        print("Declare")
        print(icon)
        s = defineifmissingtocheckforpopupadd()
        with open(s, 'w') as ss:
            ss.write(icon + "\n")
            ss.close()
        self.dismiss()


def find_color(crs, icon_nme):
    query = sql.SQL(
        "SELECT icon_color FROM icon_colors WHERE (icon_nme = \'" + icon_nme + "\')")
    print("Query: ")
    print(query)
    crs.execute(query)
    icon_color = str(crs.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(', '') \
        .replace(')', '').replace(',', '').replace('\'', '')
    return icon_color


class OriginalBusinessIconSelection(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upperbox = {}
        self.lowerbox = {}

    def fixreplication(self):
        global con
        global user
        cursor = con.cursor()
        print("IS THIS IT?")
        query = sql.SQL(
            "SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
        print("Query: ")
        print(query)
        cursor.execute(query)
        icons = fixalldatabasedonnewline(str(cursor.fetchall()))
        print(icons)
        i = 0
        while i < len(icons) - 1:
            p = i + 1
            while p < len(icons):
                if icons[i] == icons[p]:
                    icons.pop(p)
                    p -= 1
                p += 1
            i += 1
        strtoset = ""
        for u in icons:
            strtoset += str(u) + "\n"
        str_to_execute = "UPDATE business_special_info SET icon_names = \'" + strtoset + "\' WHERE business_name = \'" + \
                         user + "\'"
        print(str_to_execute)
        cursor.execute(str_to_execute)
        cursor.execute("COMMIT")

    def scheduleronenter(self):
        print("Schedule")
        Clock.schedule_interval(self.desigchecker, 0.5)

    def desigchecker(self, *args):
        print("desigcheck")
        s = defineifmissingtocheckforpopupadd()
        sss = ""
        with open(s, 'r') as ss:
            try:
                sss = ss.readlines()[0].strip()
            except IndexError:
                pass
            ss.close()
        print(sss)
        if sss != "" and sss is not None:
            print("Made it")
            self.selectme(sss)
            with open(s, 'w') as ssss:
                ssss.write("\n")
                ssss.close()

    def descheduleronleave(self):
        print("Unschedule")
        Clock.unschedule(self.desigchecker, 0.5)

    def rmv(self):
        self.ids.icons_for_selection.clear_widgets()
        self.ids.icons_selected.clear_widgets()
        self.descheduleronleave()
        self.fixreplication()

    def hintopener(self, icn, cursor):
        query = sql.SQL("SELECT icon_hint FROM icon_colors WHERE (icon_nme = \'" + icn + "\')")
        print("Query: ")
        print(query)
        cursor.execute(query)
        hint = str(cursor.fetchall()).replace('\'', '').replace(',', '').replace('[', '').replace('(', '') \
            .replace(')', '').replace(']', '').replace('\\n', '\n')
        h = defineifmissinghint()
        with open(h, 'w') as hh:
            hh.write(hint + "\n")
            hh.close()
        hinter = HintPopUp()
        hinter.opener(icon=icn)

    def selectme(self, icn):
        global con
        global user
        ip = 0
        iccn = ""
        clr = ""
        icursor = con.cursor()
        while ip < len(self.lowerbox):
            print(ip)
            print("self.lowerbox[ip].icon")
            print(self.lowerbox[ip].icon)
            print("icn")
            print(icn)
            if self.lowerbox[ip].icon == icn:
                iccn = self.lowerbox[ip].icon
                clr = self.lowerbox[ip].md_bg_color
                self.ids.icons_for_selection.remove_widget(self.lowerbox[ip])
                if check_exists_icons(icursor, user):
                    query = sql.SQL(
                        "SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
                    print("Query: ")
                    print(query)
                    icursor.execute(query)
                    set_to = str(icursor.fetchall()).replace('[', '').replace('(', '').replace('\'', '').replace(')',
                                                                                                                 '').replace(
                        ']', '').replace('None', '').replace(',', '').replace('\\n', '\n') + iccn + '\n'
                    str_to_execute = "UPDATE business_special_info SET icon_names = \'" + set_to + "\' WHERE business_name = \'" + \
                                     user + "\'"
                    print(str_to_execute)
                    icursor.execute(str_to_execute)
                    icursor.execute("COMMIT")
                else:
                    str_to_execute = "UPDATE business_special_info " \
                                     "SET icon_names = \'" + iccn.replace('None',
                                                                          '') + "\n" + "\' WHERE business_name = \'" + user + "\'"
                    icursor.execute(str_to_execute)
                    icursor.execute("COMMIT")
            ip += 1
        p = len(self.upperbox)
        self.upperbox[p] = MDFloatingActionButton(icon=iccn, md_bg_color=clr,
                                                  on_release=lambda x: self.deselectme(iccn))
        self.ids.icons_selected.add_widget(self.upperbox[p])

    def deselectme(self, icn):
        global con
        global user
        print("Deselect")
        ip = 0
        iccn = ""
        clr = ""
        icursor = con.cursor()
        while ip < len(self.upperbox):
            print(ip)
            print("self.upperbox[ip].icon")
            print(self.upperbox[ip].icon)
            print("icn")
            print(icn)
            if self.upperbox[ip].icon == icn:
                iccn = self.upperbox[ip].icon
                clr = self.upperbox[ip].md_bg_color
                self.ids.icons_selected.remove_widget(self.upperbox[ip])
                query = sql.SQL("SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
                print("Query: ")
                print(query)
                icursor.execute(query)
                set_to = str(icursor.fetchall()).replace('[', '').replace('None', '').replace('(', '').replace('\'',
                                                                                                               '').replace(
                    ')', '') \
                    .replace(']', '').replace(',', '').replace('\\n', '\n').replace((iccn + "\n"), '')
                str_to_execute = "UPDATE business_special_info SET icon_names = \'" + set_to + "\' WHERE business_name = \'" + \
                                 user + "\'"
                print(str_to_execute)
                icursor.execute(str_to_execute)
                icursor.execute("COMMIT")
            ip += 1
        p = len(self.lowerbox)
        self.lowerbox[p] = MDFloatingActionButton(icon=iccn, md_bg_color=clr,
                                                  on_release=lambda f: self.hintopener(iccn, icursor))
        self.ids.icons_for_selection.add_widget(self.lowerbox[p])
        print("Done Deselecting")

    def checkforicons(self):
        global user
        global con
        self.scheduleronenter()
        icon_cursor = con.cursor()
        query = sql.SQL("SELECT full_icon_list FROM all_icons")
        print("Query: ")
        print(query)
        icon_cursor.execute(query)
        lowerbox_icons = fixalldatabasedonnewline(str(icon_cursor.fetchall()))
        print("lowerbox_icons:")
        print(lowerbox_icons)
        if check_exists_icons(icon_cursor, user):
            query = sql.SQL("SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
            print("Query: ")
            print(query)
            icon_cursor.execute(query)
            icons = fixalldatabasedonnewline(str(icon_cursor.fetchall()))
            print("icons:")
            print(icons)
            indx = 0
            while indx < len(icons):
                if indx == 0:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[0],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[0]),
                                                                 on_release=lambda x: self.deselectme(icons[0]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[0]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 1:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[1],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[1]),
                                                                 on_release=lambda x: self.deselectme(icons[1]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[1]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 2:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[2],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[2]),
                                                                 on_release=lambda x: self.deselectme(icons[2]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[2]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 3:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[3],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[3]),
                                                                 on_release=lambda x: self.deselectme(icons[3]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[3]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 4:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[4],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[4]),
                                                                 on_release=lambda x: self.deselectme(icons[4]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[4]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 5:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[5],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[5]),
                                                                 on_release=lambda x: self.deselectme(icons[5]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[5]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 6:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[6],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[6]),
                                                                 on_release=lambda x: self.deselectme(icons[6]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[6]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 7:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[7],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[7]),
                                                                 on_release=lambda x: self.deselectme(icons[7]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[7]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 8:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[8],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[8]),
                                                                 on_release=lambda x: self.deselectme(icons[8]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[8]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 9:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[9],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[9]),
                                                                 on_release=lambda x: self.deselectme(icons[9]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[9]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 10:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[10],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[10]),
                                                                 on_release=lambda x: self.deselectme(icons[10]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[10]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 11:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[11],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[11]),
                                                                 on_release=lambda x: self.deselectme(icons[11]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[11]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 12:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[12],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[12]),
                                                                 on_release=lambda x: self.deselectme(icons[12]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[12]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 13:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[13],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[13]),
                                                                 on_release=lambda x: self.deselectme(icons[13]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[13]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 14:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[14],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[14]),
                                                                 on_release=lambda x: self.deselectme(icons[14]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[14]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 15:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[15],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[15]),
                                                                 on_release=lambda x: self.deselectme(icons[15]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[15]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 16:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[16],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[16]),
                                                                 on_release=lambda x: self.deselectme(icons[16]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[16]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 17:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[17],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[17]),
                                                                 on_release=lambda x: self.deselectme(icons[17]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[17]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 18:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[18],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[18]),
                                                                 on_release=lambda x: self.deselectme(icons[18]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[18]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 19:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[19],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[19]),
                                                                 on_release=lambda x: self.deselectme(icons[19]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[19]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 20:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[20],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[20]),
                                                                 on_release=lambda x: self.deselectme(icons[20]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[20]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 21:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[21],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[21]),
                                                                 on_release=lambda x: self.deselectme(icons[21]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[21]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 22:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[22],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[22]),
                                                                 on_release=lambda x: self.deselectme(icons[22]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[22]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 23:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[23],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[23]),
                                                                 on_release=lambda x: self.deselectme(icons[23]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[23]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 24:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[24],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[24]),
                                                                 on_release=lambda x: self.deselectme(icons[24]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[24]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 25:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[25],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[25]),
                                                                 on_release=lambda x: self.deselectme(icons[25]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[25]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 26:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[26],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[26]),
                                                                 on_release=lambda x: self.deselectme(icons[26]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[26]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 27:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[27],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[27]),
                                                                 on_release=lambda x: self.deselectme(icons[27]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[27]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 28:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[28],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[28]),
                                                                 on_release=lambda x: self.deselectme(icons[28]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[28]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 29:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[29],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[29]),
                                                                 on_release=lambda x: self.deselectme(icons[29]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[29]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 30:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[30],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[30]),
                                                                 on_release=lambda x: self.deselectme(icons[30]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[30]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 31:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[31],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[31]),
                                                                 on_release=lambda x: self.deselectme(icons[31]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[31]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 32:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[32],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[32]),
                                                                 on_release=lambda x: self.deselectme(icons[32]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[32]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 33:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[33],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[33]),
                                                                 on_release=lambda x: self.deselectme(icons[33]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[33]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 34:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[34],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[34]),
                                                                 on_release=lambda x: self.deselectme(icons[34]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[34]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 35:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[35],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[35]),
                                                                 on_release=lambda x: self.deselectme(icons[35]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[35]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 36:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[36],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[36]),
                                                                 on_release=lambda x: self.deselectme(icons[36]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[36]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 37:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[37],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[37]),
                                                                 on_release=lambda x: self.deselectme(icons[37]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[37]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 38:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[38],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[38]),
                                                                 on_release=lambda x: self.deselectme(icons[38]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[38]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 39:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[39],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[39]),
                                                                 on_release=lambda x: self.deselectme(icons[39]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[39]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 40:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[40],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[40]),
                                                                 on_release=lambda x: self.deselectme(icons[40]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[40]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 41:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[41],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[41]),
                                                                 on_release=lambda x: self.deselectme(icons[41]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[41]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 42:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[42],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[42]),
                                                                 on_release=lambda x: self.deselectme(icons[42]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[42]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 43:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[43],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[43]),
                                                                 on_release=lambda x: self.deselectme(icons[43]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[43]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 44:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[44],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[44]),
                                                                 on_release=lambda x: self.deselectme(icons[44]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[44]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 45:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[45],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[45]),
                                                                 on_release=lambda x: self.deselectme(icons[45]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[45]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 46:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[46],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[46]),
                                                                 on_release=lambda x: self.deselectme(icons[46]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[46]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 47:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[47],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[47]),
                                                                 on_release=lambda x: self.deselectme(icons[47]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[47]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 48:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[48],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[48]),
                                                                 on_release=lambda x: self.deselectme(icons[48]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[48]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 49:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[49],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[49]),
                                                                 on_release=lambda x: self.deselectme(icons[49]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[49]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 50:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[50],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[50]),
                                                                 on_release=lambda x: self.deselectme(icons[50]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[50]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 51:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[51],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[51]),
                                                                 on_release=lambda x: self.deselectme(icons[51]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[51]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 52:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[52],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[52]),
                                                                 on_release=lambda x: self.deselectme(icons[52]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[52]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 53:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[53],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[53]),
                                                                 on_release=lambda x: self.deselectme(icons[53]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[53]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 54:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[54],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[54]),
                                                                 on_release=lambda x: self.deselectme(icons[54]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[54]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 55:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[55],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[55]),
                                                                 on_release=lambda x: self.deselectme(icons[55]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[55]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 56:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[56],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[56]),
                                                                 on_release=lambda x: self.deselectme(icons[56]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[56]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 57:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[57],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[57]),
                                                                 on_release=lambda x: self.deselectme(icons[57]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[57]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
        indx = 0
        while indx < len(lowerbox_icons):
            if indx == 0:
                print("indx:")
                print(indx)
                print(lowerbox_icons[0])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[0],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[0]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[0],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 1:
                print("indx:")
                print(indx)
                print(lowerbox_icons[1])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[1],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[1]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[1],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 2:
                print("indx:")
                print(indx)
                print(lowerbox_icons[2])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[2],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[2]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[2],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 3:
                print("indx:")
                print(indx)
                print(lowerbox_icons[0])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[3],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[3]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[3],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 4:
                print("indx:")
                print(indx)
                print(lowerbox_icons[4])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[4],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[4]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[4],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 5:
                print("indx:")
                print(indx)
                print(lowerbox_icons[5])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[5],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[5]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[5],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 6:
                print("indx:")
                print(indx)
                print(lowerbox_icons[6])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[6],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[6]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[6],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 7:
                print("indx:")
                print(indx)
                print(lowerbox_icons[7])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[7],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[7]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[7],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 8:
                print("indx:")
                print(indx)
                print(lowerbox_icons[8])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[8],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[8]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[8],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 9:
                print("indx:")
                print(indx)
                print(lowerbox_icons[9])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[9],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[9]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[9],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 10:
                print("indx:")
                print(indx)
                print(lowerbox_icons[10])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[10],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[10]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[10],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 11:
                print("indx:")
                print(indx)
                print(lowerbox_icons[0])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[11],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[11]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[11],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 12:
                print("indx:")
                print(indx)
                print(lowerbox_icons[12])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[12],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[12]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[12],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 13:
                print("indx:")
                print(indx)
                print(lowerbox_icons[13])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[13],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[13]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[13],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 14:
                print("indx:")
                print(indx)
                print(lowerbox_icons[14])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[14],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[14]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[14],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 15:
                print("indx:")
                print(indx)
                print(lowerbox_icons[15])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[15],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[15]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[15],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 16:
                print("indx:")
                print(indx)
                print(lowerbox_icons[16])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[16],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[16]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[16],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 17:
                print("indx:")
                print(indx)
                print(lowerbox_icons[17])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[17],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[17]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[17],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 18:
                print("indx:")
                print(indx)
                print(lowerbox_icons[18])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[18],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[18]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[18],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 19:
                print("indx:")
                print(indx)
                print(lowerbox_icons[19])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[19],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[19]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[19],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 20:
                print("indx:")
                print(indx)
                print(lowerbox_icons[20])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[20],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[20]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[20],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 21:
                print("indx:")
                print(indx)
                print(lowerbox_icons[21])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[21],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[21]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[21],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 22:
                print("indx:")
                print(indx)
                print(lowerbox_icons[22])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[22],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[22]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[22],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 23:
                print("indx:")
                print(indx)
                print(lowerbox_icons[23])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[23],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[23]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[23],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 24:
                print("indx:")
                print(indx)
                print(lowerbox_icons[24])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[24],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[24]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[24],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 25:
                print("indx:")
                print(indx)
                print(lowerbox_icons[25])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[25],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[25]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[25],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 26:
                print("indx:")
                print(indx)
                print(lowerbox_icons[26])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[26],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[26]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[26],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 27:
                print("indx:")
                print(indx)
                print(lowerbox_icons[27])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[27],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[27]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[27],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 28:
                print("indx:")
                print(indx)
                print(lowerbox_icons[28])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[28],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[28]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[28],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 29:
                print("indx:")
                print(indx)
                print(lowerbox_icons[29])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[29],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[29]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[29],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 30:
                print("indx:")
                print(indx)
                print(lowerbox_icons[30])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[30],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[30]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[30],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 31:
                print("indx:")
                print(indx)
                print(lowerbox_icons[31])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[31],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[31]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[31],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 32:
                print("indx:")
                print(indx)
                print(lowerbox_icons[32])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[32],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[32]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[32],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 33:
                print("indx:")
                print(indx)
                print(lowerbox_icons[33])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[33],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[33]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[33],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 34:
                print("indx:")
                print(indx)
                print(lowerbox_icons[34])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[34],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[34]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[34],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 35:
                print("indx:")
                print(indx)
                print(lowerbox_icons[35])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[35],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[35]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[35],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 36:
                print("indx:")
                print(indx)
                print(lowerbox_icons[36])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[36],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[36]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[36],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 37:
                print("indx:")
                print(indx)
                print(lowerbox_icons[37])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[37],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[37]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[37],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 38:
                print("indx:")
                print(indx)
                print(lowerbox_icons[38])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[38],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[38]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[38],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 39:
                print("indx:")
                print(indx)
                print(lowerbox_icons[39])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[39],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[39]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[39],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 40:
                print("indx:")
                print(indx)
                print(lowerbox_icons[40])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[40],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[40]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[40],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 41:
                print("indx:")
                print(indx)
                print(lowerbox_icons[41])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[41],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[41]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[41],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 42:
                print("indx:")
                print(indx)
                print(lowerbox_icons[42])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[42],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[42]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[42],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 43:
                print("indx:")
                print(indx)
                print(lowerbox_icons[43])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[43],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[43]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[43],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 44:
                print("indx:")
                print(indx)
                print(lowerbox_icons[44])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[44],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[44]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[44],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 45:
                print("indx:")
                print(indx)
                print(lowerbox_icons[45])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[45],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[45]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[45],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 46:
                print("indx:")
                print(indx)
                print(lowerbox_icons[46])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[46],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[46]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[46],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 47:
                print("indx:")
                print(indx)
                print(lowerbox_icons[47])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[47],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[47]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[47],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 48:
                print("indx:")
                print(indx)
                print(lowerbox_icons[48])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[48],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[48]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[48],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 49:
                print("indx:")
                print(indx)
                print(lowerbox_icons[49])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[49],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[49]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[49],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 50:
                print("indx:")
                print(indx)
                print(lowerbox_icons[50])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[50],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[50]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[50],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 51:
                print("indx:")
                print(indx)
                print(lowerbox_icons[51])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[51],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[51]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[51],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 52:
                print("indx:")
                print(indx)
                print(lowerbox_icons[52])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[52],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[52]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[52],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 53:
                print("indx:")
                print(indx)
                print(lowerbox_icons[53])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[53],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[53]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[53],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 54:
                print("indx:")
                print(indx)
                print(lowerbox_icons[54])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[54],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[54]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[54],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 55:
                print("indx:")
                print(indx)
                print(lowerbox_icons[55])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[55],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[55]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[55],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 56:
                print("indx:")
                print(indx)
                print(lowerbox_icons[56])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[56],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[56]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[56],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 57:
                print("indx:")
                print(indx)
                print(lowerbox_icons[57])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[57],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[57]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[57],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            indx += 1


class BusinessIconSelection(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upperbox = {}
        self.lowerbox = {}

    def fixreplication(self):
        global con
        global user
        cursor = con.cursor()
        query = sql.SQL(
            "SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
        print("Query: ")
        print(query)
        cursor.execute(query)
        icons = fixalldatabasedonnewline(str(cursor.fetchall()))
        print(icons)
        i = 0
        while i < len(icons) - 1:
            p = i + 1
            while p < len(icons):
                if icons[i] == icons[p]:
                    icons.pop(p)
                    p -= 1
                p += 1
            i += 1
        strtoset = ""
        for u in icons:
            strtoset += str(u) + "\n"
        str_to_execute = "UPDATE business_special_info SET icon_names = \'" + strtoset + "\' WHERE business_name = \'" + \
                         user + "\'"
        print(str_to_execute)
        cursor.execute(str_to_execute)
        cursor.execute("COMMIT")

    def scheduleronenter(self):
        print("Schedule")
        Clock.schedule_interval(self.desigchecker, 0.5)

    def desigchecker(self, *args):
        print("desigcheck")
        s = defineifmissingtocheckforpopupadd()
        sss = ""
        with open(s, 'r') as ss:
            try:
                sss = ss.readlines()[0].strip()
            except IndexError:
                pass
            ss.close()
        print(sss)
        if sss != "" and sss is not None:
            print("Made it")
            self.selectme(sss)
            with open(s, 'w') as ssss:
                ssss.write("\n")
                ssss.close()

    def descheduleronleave(self):
        print("Unschedule")
        Clock.unschedule(self.desigchecker, 0.5)

    def rmvtwo(self):
        self.ids.icons_for_selection.clear_widgets()
        self.ids.icons_selected.clear_widgets()
        self.descheduleronleave()
        self.fixreplication()

    def hintopener(self, icn, cursor):
        query = sql.SQL("SELECT icon_hint FROM icon_colors WHERE (icon_nme = \'" + icn + "\')")
        print("Query: ")
        print(query)
        cursor.execute(query)
        hint = str(cursor.fetchall()).replace('\'', '').replace(',', '').replace('[', '').replace('(', '') \
            .replace(')', '').replace(']', '').replace('\\n', '\n')
        h = defineifmissinghint()
        with open(h, 'w') as hh:
            hh.write(hint + "\n")
            hh.close()
        hinter = HintPopUp()
        hinter.opener(icon=icn)

    def selectme(self, icn):
        global con
        global user
        ip = 0
        iccn = ""
        clr = ""
        icursor = con.cursor()
        while ip < len(self.lowerbox):
            print(ip)
            print("self.lowerbox[ip].icon")
            print(self.lowerbox[ip].icon)
            print("icn")
            print(icn)
            if self.lowerbox[ip].icon == icn:
                iccn = self.lowerbox[ip].icon
                clr = self.lowerbox[ip].md_bg_color
                self.ids.icons_for_selection.remove_widget(self.lowerbox[ip])
                if check_exists_icons(icursor, user):
                    query = sql.SQL(
                        "SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
                    print("Query: ")
                    print(query)
                    icursor.execute(query)
                    set_to = str(icursor.fetchall()).replace('[', '').replace('None', '').replace('(', '').replace('\'',
                                                                                                                   '').replace(
                        ')', '').replace(']', '').replace(',', '').replace('\\n', '\n') + iccn + '\n'
                    str_to_execute = "UPDATE business_special_info SET icon_names = \'" + set_to + "\' WHERE business_name = \'" + \
                                     user + "\'"
                    print(str_to_execute)
                    icursor.execute(str_to_execute)
                    icursor.execute("COMMIT")
                else:
                    str_to_execute = "UPDATE business_special_info SET " \
                                     "icon_names = \'" + iccn.replace('None',
                                                                      '') + "\n" + "\' WHERE business_name = \'" + user + "\'"
                    icursor.execute(str_to_execute)
                    icursor.execute("COMMIT")
            ip += 1
        p = len(self.upperbox)
        self.upperbox[p] = MDFloatingActionButton(icon=iccn, md_bg_color=clr,
                                                  on_release=lambda x: self.deselectme(iccn))
        self.ids.icons_selected.add_widget(self.upperbox[p])

    def deselectme(self, icn):
        global con
        global user
        print("Deselect")
        ip = 0
        iccn = ""
        clr = ""
        icursor = con.cursor()
        while ip < len(self.upperbox):
            print(ip)
            print("self.upperbox[ip].icon")
            print(self.upperbox[ip].icon)
            print("icn")
            print(icn)
            if self.upperbox[ip].icon == icn:
                iccn = self.upperbox[ip].icon
                clr = self.upperbox[ip].md_bg_color
                self.ids.icons_selected.remove_widget(self.upperbox[ip])
                query = sql.SQL("SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
                print("Query: ")
                print(query)
                icursor.execute(query)
                set_to = str(icursor.fetchall()).replace('[', '').replace('None', '').replace('(', '').replace('\'',
                                                                                                               '').replace(
                    ')', '').replace(']', '').replace(',', '').replace('\\n', '\n').replace((iccn + "\n"), '')
                str_to_execute = "UPDATE business_special_info SET icon_names = \'" + set_to + "\' WHERE business_name = \'" + \
                                 user + "\'"
                print(str_to_execute)
                icursor.execute(str_to_execute)
                icursor.execute("COMMIT")
            ip += 1
        p = len(self.lowerbox)
        self.lowerbox[p] = MDFloatingActionButton(icon=iccn, md_bg_color=clr,
                                                  on_release=lambda f: self.hintopener(iccn, icursor))
        self.ids.icons_for_selection.add_widget(self.lowerbox[p])
        print("Done Deselecting")

    def checkforiconstwo(self):
        global user
        global con
        self.scheduleronenter()
        icon_cursor = con.cursor()
        query = sql.SQL("SELECT full_icon_list FROM all_icons")
        print("Query: ")
        print(query)
        icon_cursor.execute(query)
        lowerbox_icons = fixalldatabasedonnewline(str(icon_cursor.fetchall()))
        print("lowerbox_icons:")
        print(lowerbox_icons)
        if check_exists_icons(icon_cursor, user):
            query = sql.SQL("SELECT icon_names FROM business_special_info WHERE (business_name = \'" + user + "\')")
            print("Query: ")
            print(query)
            icon_cursor.execute(query)
            icons = fixalldatabasedonnewline(str(icon_cursor.fetchall()))
            print("icons:")
            print(icons)
            indx = 0
            while indx < len(icons):
                if indx == 0:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[0],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[0]),
                                                                 on_release=lambda x: self.deselectme(icons[0]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[0]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 1:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[1],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[1]),
                                                                 on_release=lambda x: self.deselectme(icons[1]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[1]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 2:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[2],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[2]),
                                                                 on_release=lambda x: self.deselectme(icons[2]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[2]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 3:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[3],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[3]),
                                                                 on_release=lambda x: self.deselectme(icons[3]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[3]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 4:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[4],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[4]),
                                                                 on_release=lambda x: self.deselectme(icons[4]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[4]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 5:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[5],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[5]),
                                                                 on_release=lambda x: self.deselectme(icons[5]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[5]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 6:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[6],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[6]),
                                                                 on_release=lambda x: self.deselectme(icons[6]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[6]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 7:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[7],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[7]),
                                                                 on_release=lambda x: self.deselectme(icons[7]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[7]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 8:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[8],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[8]),
                                                                 on_release=lambda x: self.deselectme(icons[8]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[8]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 9:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[9],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[9]),
                                                                 on_release=lambda x: self.deselectme(icons[9]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[9]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 10:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[10],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[10]),
                                                                 on_release=lambda x: self.deselectme(icons[10]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[10]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 11:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[11],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[11]),
                                                                 on_release=lambda x: self.deselectme(icons[11]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[11]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 12:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[12],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[12]),
                                                                 on_release=lambda x: self.deselectme(icons[12]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[12]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 13:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[13],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[13]),
                                                                 on_release=lambda x: self.deselectme(icons[13]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[13]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 14:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[14],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[14]),
                                                                 on_release=lambda x: self.deselectme(icons[14]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[14]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 15:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[15],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[15]),
                                                                 on_release=lambda x: self.deselectme(icons[15]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[15]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 16:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[16],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[16]),
                                                                 on_release=lambda x: self.deselectme(icons[16]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[16]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 17:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[17],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[17]),
                                                                 on_release=lambda x: self.deselectme(icons[17]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[17]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 18:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[18],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[18]),
                                                                 on_release=lambda x: self.deselectme(icons[18]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[18]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 19:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[19],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[19]),
                                                                 on_release=lambda x: self.deselectme(icons[19]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[19]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 20:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[20],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[20]),
                                                                 on_release=lambda x: self.deselectme(icons[20]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[20]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 21:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[21],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[21]),
                                                                 on_release=lambda x: self.deselectme(icons[21]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[21]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 22:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[22],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[22]),
                                                                 on_release=lambda x: self.deselectme(icons[22]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[22]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 23:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[23],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[23]),
                                                                 on_release=lambda x: self.deselectme(icons[23]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[23]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 24:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[24],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[24]),
                                                                 on_release=lambda x: self.deselectme(icons[24]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[24]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 25:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[25],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[25]),
                                                                 on_release=lambda x: self.deselectme(icons[25]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[25]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 26:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[26],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[26]),
                                                                 on_release=lambda x: self.deselectme(icons[26]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[26]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 27:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[27],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[27]),
                                                                 on_release=lambda x: self.deselectme(icons[27]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[27]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 28:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[28],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[28]),
                                                                 on_release=lambda x: self.deselectme(icons[28]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[28]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 29:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[29],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[29]),
                                                                 on_release=lambda x: self.deselectme(icons[29]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[29]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 30:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[30],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[30]),
                                                                 on_release=lambda x: self.deselectme(icons[30]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[30]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 31:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[31],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[31]),
                                                                 on_release=lambda x: self.deselectme(icons[31]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[31]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 32:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[32],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[32]),
                                                                 on_release=lambda x: self.deselectme(icons[32]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[32]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 33:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[33],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[33]),
                                                                 on_release=lambda x: self.deselectme(icons[33]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[33]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 34:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[34],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[34]),
                                                                 on_release=lambda x: self.deselectme(icons[34]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[34]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 35:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[35],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[35]),
                                                                 on_release=lambda x: self.deselectme(icons[35]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[35]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 36:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[36],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[36]),
                                                                 on_release=lambda x: self.deselectme(icons[36]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[36]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 37:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[37],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[37]),
                                                                 on_release=lambda x: self.deselectme(icons[37]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[37]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 38:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[38],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[38]),
                                                                 on_release=lambda x: self.deselectme(icons[38]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[38]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 39:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[39],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[39]),
                                                                 on_release=lambda x: self.deselectme(icons[39]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[39]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 40:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[40],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[40]),
                                                                 on_release=lambda x: self.deselectme(icons[40]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[40]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 41:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[41],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[41]),
                                                                 on_release=lambda x: self.deselectme(icons[41]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[41]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 42:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[42],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[42]),
                                                                 on_release=lambda x: self.deselectme(icons[42]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[42]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 43:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[43],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[43]),
                                                                 on_release=lambda x: self.deselectme(icons[43]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[43]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 44:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[44],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[44]),
                                                                 on_release=lambda x: self.deselectme(icons[44]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[44]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 45:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[45],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[45]),
                                                                 on_release=lambda x: self.deselectme(icons[45]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[45]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 46:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[46],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[46]),
                                                                 on_release=lambda x: self.deselectme(icons[46]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[46]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 47:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[47],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[47]),
                                                                 on_release=lambda x: self.deselectme(icons[47]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[47]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 48:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[48],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[48]),
                                                                 on_release=lambda x: self.deselectme(icons[48]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[48]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 49:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[49],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[49]),
                                                                 on_release=lambda x: self.deselectme(icons[49]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[49]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 50:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[50],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[50]),
                                                                 on_release=lambda x: self.deselectme(icons[50]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[50]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 51:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[51],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[51]),
                                                                 on_release=lambda x: self.deselectme(icons[51]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[51]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 52:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[52],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[52]),
                                                                 on_release=lambda x: self.deselectme(icons[52]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[52]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 53:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[53],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[53]),
                                                                 on_release=lambda x: self.deselectme(icons[53]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[53]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 54:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[54],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[54]),
                                                                 on_release=lambda x: self.deselectme(icons[54]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[54]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 55:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[55],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[55]),
                                                                 on_release=lambda x: self.deselectme(icons[55]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[55]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 56:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[56],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[56]),
                                                                 on_release=lambda x: self.deselectme(icons[56]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[56]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
                elif indx == 57:
                    self.upperbox[indx] = MDFloatingActionButton(icon=icons[57],
                                                                 md_bg_color=find_color(crs=icon_cursor,
                                                                                        icon_nme=icons[57]),
                                                                 on_release=lambda x: self.deselectme(icons[57]))
                    self.ids.icons_selected.add_widget(self.upperbox[indx])
                    u = 0
                    while u < len(lowerbox_icons):
                        if lowerbox_icons[u] == icons[57]:
                            lowerbox_icons.pop(u)
                            u -= 1
                        u += 1
                    indx += 1
        indx = 0
        while indx < len(lowerbox_icons):
            if indx == 0:
                print("indx:")
                print(indx)
                print(lowerbox_icons[0])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[0],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[0]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[0],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 1:
                print("indx:")
                print(indx)
                print(lowerbox_icons[1])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[1],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[1]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[1],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 2:
                print("indx:")
                print(indx)
                print(lowerbox_icons[2])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[2],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[2]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[2],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 3:
                print("indx:")
                print(indx)
                print(lowerbox_icons[0])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[3],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[3]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[3],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 4:
                print("indx:")
                print(indx)
                print(lowerbox_icons[4])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[4],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[4]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[4],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 5:
                print("indx:")
                print(indx)
                print(lowerbox_icons[5])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[5],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[5]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[5],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 6:
                print("indx:")
                print(indx)
                print(lowerbox_icons[6])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[6],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[6]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[6],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 7:
                print("indx:")
                print(indx)
                print(lowerbox_icons[7])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[7],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[7]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[7],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 8:
                print("indx:")
                print(indx)
                print(lowerbox_icons[8])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[8],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[8]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[8],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 9:
                print("indx:")
                print(indx)
                print(lowerbox_icons[9])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[9],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[9]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[9],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 10:
                print("indx:")
                print(indx)
                print(lowerbox_icons[10])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[10],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[10]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[10],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 11:
                print("indx:")
                print(indx)
                print(lowerbox_icons[0])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[11],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[11]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[11],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 12:
                print("indx:")
                print(indx)
                print(lowerbox_icons[12])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[12],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[12]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[12],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 13:
                print("indx:")
                print(indx)
                print(lowerbox_icons[13])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[13],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[13]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[13],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 14:
                print("indx:")
                print(indx)
                print(lowerbox_icons[14])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[14],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[14]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[14],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 15:
                print("indx:")
                print(indx)
                print(lowerbox_icons[15])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[15],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[15]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[15],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 16:
                print("indx:")
                print(indx)
                print(lowerbox_icons[16])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[16],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[16]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[16],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 17:
                print("indx:")
                print(indx)
                print(lowerbox_icons[17])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[17],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[17]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[17],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 18:
                print("indx:")
                print(indx)
                print(lowerbox_icons[18])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[18],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[18]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[18],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 19:
                print("indx:")
                print(indx)
                print(lowerbox_icons[19])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[19],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[19]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[19],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 20:
                print("indx:")
                print(indx)
                print(lowerbox_icons[20])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[20],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[20]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[20],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 21:
                print("indx:")
                print(indx)
                print(lowerbox_icons[21])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[21],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[21]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[21],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 22:
                print("indx:")
                print(indx)
                print(lowerbox_icons[22])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[22],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[22]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[22],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 23:
                print("indx:")
                print(indx)
                print(lowerbox_icons[23])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[23],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[23]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[23],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 24:
                print("indx:")
                print(indx)
                print(lowerbox_icons[24])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[24],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[24]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[24],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 25:
                print("indx:")
                print(indx)
                print(lowerbox_icons[25])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[25],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[25]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[25],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 26:
                print("indx:")
                print(indx)
                print(lowerbox_icons[26])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[26],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[26]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[26],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 27:
                print("indx:")
                print(indx)
                print(lowerbox_icons[27])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[27],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[27]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[27],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 28:
                print("indx:")
                print(indx)
                print(lowerbox_icons[28])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[28],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[28]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[28],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 29:
                print("indx:")
                print(indx)
                print(lowerbox_icons[29])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[29],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[29]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[29],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 30:
                print("indx:")
                print(indx)
                print(lowerbox_icons[30])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[30],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[30]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[30],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 31:
                print("indx:")
                print(indx)
                print(lowerbox_icons[31])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[31],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[31]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[31],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 32:
                print("indx:")
                print(indx)
                print(lowerbox_icons[32])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[32],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[32]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[32],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 33:
                print("indx:")
                print(indx)
                print(lowerbox_icons[33])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[33],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[33]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[33],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 34:
                print("indx:")
                print(indx)
                print(lowerbox_icons[34])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[34],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[34]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[34],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 35:
                print("indx:")
                print(indx)
                print(lowerbox_icons[35])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[35],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[35]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[35],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 36:
                print("indx:")
                print(indx)
                print(lowerbox_icons[36])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[36],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[36]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[36],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 37:
                print("indx:")
                print(indx)
                print(lowerbox_icons[37])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[37],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[37]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[37],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 38:
                print("indx:")
                print(indx)
                print(lowerbox_icons[38])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[38],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[38]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[38],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 39:
                print("indx:")
                print(indx)
                print(lowerbox_icons[39])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[39],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[39]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[39],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 40:
                print("indx:")
                print(indx)
                print(lowerbox_icons[40])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[40],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[40]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[40],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 41:
                print("indx:")
                print(indx)
                print(lowerbox_icons[41])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[41],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[41]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[41],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 42:
                print("indx:")
                print(indx)
                print(lowerbox_icons[42])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[42],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[42]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[42],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 43:
                print("indx:")
                print(indx)
                print(lowerbox_icons[43])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[43],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[43]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[43],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 44:
                print("indx:")
                print(indx)
                print(lowerbox_icons[44])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[44],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[44]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[44],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 45:
                print("indx:")
                print(indx)
                print(lowerbox_icons[45])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[45],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[45]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[45],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 46:
                print("indx:")
                print(indx)
                print(lowerbox_icons[46])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[46],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[46]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[46],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 47:
                print("indx:")
                print(indx)
                print(lowerbox_icons[47])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[47],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[47]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[47],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 48:
                print("indx:")
                print(indx)
                print(lowerbox_icons[48])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[48],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[48]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[48],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 49:
                print("indx:")
                print(indx)
                print(lowerbox_icons[49])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[49],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[49]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[49],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 50:
                print("indx:")
                print(indx)
                print(lowerbox_icons[50])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[50],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[50]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[50],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 51:
                print("indx:")
                print(indx)
                print(lowerbox_icons[51])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[51],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[51]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[51],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 52:
                print("indx:")
                print(indx)
                print(lowerbox_icons[52])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[52],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[52]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[52],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 53:
                print("indx:")
                print(indx)
                print(lowerbox_icons[53])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[53],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[53]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[53],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 54:
                print("indx:")
                print(indx)
                print(lowerbox_icons[54])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[54],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[54]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[54],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 55:
                print("indx:")
                print(indx)
                print(lowerbox_icons[55])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[55],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[55]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[55],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 56:
                print("indx:")
                print(indx)
                print(lowerbox_icons[56])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[56],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[56]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[56],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            elif indx == 57:
                print("indx:")
                print(indx)
                print(lowerbox_icons[57])
                self.lowerbox[indx] = MDFloatingActionButton(icon=lowerbox_icons[57],
                                                             md_bg_color=find_color(crs=icon_cursor,
                                                                                    icon_nme=lowerbox_icons[57]),
                                                             on_release=lambda f: self.hintopener(lowerbox_icons[57],
                                                                                                  icon_cursor))
                self.ids.icons_for_selection.add_widget(self.lowerbox[indx])
            indx += 1


def follow(business_name):
    global user
    global con
    crs = con.cursor()

    """Business Side"""
    if followers_exist(crs, business_name):
        query = sql.SQL(
            "SELECT followers FROM business_followers WHERE (business_name = \'" + business_name + "\')")
        print("Query: ")
        print(query)
        crs.execute(query)
        followers = str(crs.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(', '').replace(')',
                                                                                                                    '').replace(
            ',', '').replace('\'', '')
        followers += str(user) + "\n"
        str_to_execute = "UPDATE business_followers SET followers = \'" + followers + "\' WHERE business_name = \'" + \
                         business_name + "\'"
        print(str_to_execute)
        crs.execute(str_to_execute)
        crs.execute("COMMIT")
    else:
        #################################### FIX THIS: NO WHERE IN INSERT
        """ Original Here
        str_to_execute = \
            "INSERT INTO business_followers(followers) VALUES(\'" + user + "\n" + "\') WHERE (business_name = \'" + business_name + "\')"
        crs.execute(str_to_execute)
        crs.execute("COMMIT")
        """
        # New?
        str_to_execute = \
            "INSERT INTO business_followers(followers, business_name) VALUES(\'" + user + "\n" + "\', \'" + business_name + "\')"
        crs.execute(str_to_execute)
        crs.execute("COMMIT")

    """Follower Side"""
    if followers_exist_c(crs, user):
        query = sql.SQL(
            "SELECT followings FROM client_followings WHERE (username = \'" + user + "\')")
        print("Query: ")
        print(query)
        crs.execute(query)
        followings = str(crs.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(', '').replace(
            ')', '').replace(',', '').replace('\'', '')
        followings += str(business_name) + "\n"
        str_to_execute = "UPDATE client_followings SET following = \'" + followings + "\' WHERE username = \'" + \
                         user + "\'"
        print(str_to_execute)
        crs.execute(str_to_execute)
        crs.execute("COMMIT")
    else:
        ############################################################################################## FIX THIS
        str_to_execute = \
            "INSERT INTO client_followings(following, username) VALUES(\'" + business_name + "\n" + "\', \'" + user + "\')"
        crs.execute(str_to_execute)
        crs.execute("COMMIT")

        """num_followers"""
        query = sql.SQL(
            "SELECT num_followers FROM business_followers WHERE (business_name = \'" + business_name + "\')")
        print("Query: ")
        print(query)
        crs.execute(query)
        numfollowers = int(
            str(crs.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(', '').replace(')',
                                                                                                            '').replace(
                ',', '').replace('\'', ''))
        numfollowers += 1
        str_to_execute = "UPDATE business_followers SET num_followers = (" + str(
            numfollowers) + ") WHERE business_name = \'" + \
                         business_name + "\'"
        print(str_to_execute)
        crs.execute(str_to_execute)
        crs.execute("COMMIT")


def find_nearby_businesses_in_order(username):
    global con
    cursorr = con.cursor()
    query = sql.SQL(
        "SELECT lat, lon FROM special_info WHERE (username = \'" + username + "\')")
    print("Query: ")
    print(query)
    cursorr.execute(query)
    v = str(cursorr.fetchall())
    print(v)
    # query = sql.SQL(
    #     "SELECT ALL business_name FROM business_special_info WHERE (lat < " + str(float(clat) + 2) +" AND lat > " + str(float(clat) - 2) + " AND lon < " + str(float(clon) + 2) + " AND lon > " + str(float(clon) - 2) + ")")
    # print("Query: ")
    # print(query)
    # cursorr.execute(query)
    # v = str(cursorr.fetchall())
    # print(v)
    return []


class FollowPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.box = {}
        self.lb = {}
        self.gc = {}
        self.b = {}
        self.b2 = {}
        self.b3 = {}

    def bfllwremover(self):
        p = 0
        while p < len(self.box):
            fdname = self.b2[p].name[0:len(self.b2[p].name)]
            if self.box[p].name[0:len(self.box[p].name)] == fdname:
                self.name.forbusinesses.remove_widget(self.box[p])
            p += 1

    def recordx(self, fd_name):
        c = defineifmissingbusinesstofollowsession()
        with open(c, 'a') as f:
            f.write(fd_name.text + "\n")
            f.close()

    def viewprofile(self):
        pass

    def tempfollow(self, lb, bfllwlstelement):
        follow(bfllwlstelement)

    def onloadbfllw(self):
        lst = []
        # b_partialestref()
        global con
        global user
        crs = con.cursor()
        query = sql.SQL(
            "SELECT num_following FROM client_followings WHERE (username = \'" + user + "\')")
        print("Query: ")
        print(query)
        crs.execute(query)
        numcrfollowing = str(crs.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(',
                                                                                                        '').replace(')',
                                                                                                                    '').replace(
            ',', '')
        query = sql.SQL(
            "SELECT following FROM client_followings WHERE (username = \'" + user + "\')")
        print("Query: ")
        print(query)
        crs.execute(query)
        crbfllw = fixalldatabasedonnewline(str(crs.fetchall()))
        nearby_businesses_in_order = find_nearby_businesses_in_order(user)
        bfllwlst = nearby_businesses_in_order
        b = 0
        while b < len(bfllwlst):
            u = 0
            while u < len(crbfllw):
                if crbfllw[u] == bfllwlst[b]:
                    bfllwlst.pop(b)
                    b -= 1
                    break
                u += 1
            b += 1
        fllwfile = defineifmissingbusinesstofollowsession()
        print("x")
        with open(fllwfile, 'r') as fdf:
            bfllw = fdf.readlines()
            fdf.close()
        if len(bfllw) == 0:
            with open(fllwfile, 'w') as p:
                p.close()
        bindx = 0
        print("x")
        while bindx < len(bfllwlst):
            if not isinbfllw(bfllw, bindx, bfllwlst):
                ind = bindx
                self.box[ind] = (GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                            padding=[10, 2, 10, 2]))
                self.lb[ind] = (Label(text=bfllwlst[bindx], size_hint_y=0.1))
                self.gc[ind] = (GridLayout(cols=1, rows=3))
                """View Buttons START"""
                if ind == 0:
                    # if self.lb[0].text != '' and self.lb[0] is not None:
                    #    self.recordx(self.lb[0])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[0], bfllwlst[0])))
                elif ind == 1:
                    # if self.lb[1].text != '' and self.lb[1] is not None:
                    #    self.recordx(self.lb[1])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[1], bfllwlst[1])))
                elif ind == 2:
                    # if self.lb[2].text != '' and self.lb[2] is not None:
                    #    self.recordx(self.lb[2])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[2], bfllwlst[2])))
                elif ind == 3:
                    # if self.lb[3].text != '' and self.lb[3] is not None:
                    #    self.recordx(self.lb[3])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[3], bfllwlst[3])))
                elif ind == 4:
                    # if self.lb[4].text != '' and self.lb[4] is not None:
                    #     self.recordx(self.lb[4])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[4], bfllwlst[4])))
                elif ind == 5:
                    # if self.lb[5].text != '' and self.lb[5] is not None:
                    #     self.recordx(self.lb[5])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[5], bfllwlst[5])))
                elif ind == 6:
                    # if self.lb[6].text != '' and self.lb[6] is not None:
                    #     self.recordx(self.lb[6])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[6], bfllwlst[6])))
                elif ind == 7:
                    # if self.lb[7].text != '' and self.lb[7] is not None:
                    #     self.recordx(self.lb[7])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[7], bfllwlst[7])))
                elif ind == 8:
                    # if self.lb[8].text != '' and self.lb[8] is not None:
                    #     self.recordx(self.lb[8])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[8], bfllwlst[8])))
                elif ind == 9:
                    # if self.lb[9].text != '' and self.lb[9] is not None:
                    #     self.recordx(self.lb[9])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[9], bfllwlst[9])))
                elif ind == 10:
                    # if self.lb[10].text != '' and self.lb[10] is not None:
                    #     self.recordx(self.lb[10])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[10], bfllwlst[10])))
                elif ind == 11:
                    # if self.lb[11].text != '' and self.lb[11] is not None:
                    #     self.recordx(self.lb[11])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[11], bfllwlst[11])))
                elif ind == 12:
                    # if self.lb[12].text != '' and self.lb[12] is not None:
                    #     self.recordx(self.lb[12])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[12], bfllwlst[12])))
                elif ind == 13:
                    # if self.lb[13].text != '' and self.lb[13] is not None:
                    #     self.recordx(self.lb[13])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[13], bfllwlst[13])))
                elif ind == 14:
                    # if self.lb[14].text != '' and self.lb[14] is not None:
                    #     self.recordx(self.lb[14])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[14], bfllwlst[14])))
                elif ind == 15:
                    # if self.lb[15].text != '' and self.lb[15] is not None:
                    #     self.recordx(self.lb[15])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[15], bfllwlst[15])))
                elif ind == 16:
                    # if self.lb[16].text != '' and self.lb[16] is not None:
                    #     self.recordx(self.lb[16])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[16], bfllwlst[16])))
                elif ind == 17:
                    # if self.lb[17].text != '' and self.lb[17] is not None:
                    #     self.recordx(self.lb[17])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[17], bfllwlst[17])))
                elif ind == 18:
                    # if self.lb[18].text != '' and self.lb[18] is not None:
                    #     self.recordx(self.lb[18])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[18], bfllwlst[18])))
                elif ind == 19:
                    # if self.lb[19].text != '' and self.lb[19] is not None:
                    #     self.recordx(self.lb[19])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[19], bfllwlst[19])))
                elif ind == 20:
                    # if self.lb[20].text != '' and self.lb[20] is not None:
                    #     self.recordx(self.lb[20])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[20], bfllwlst[20])))
                elif ind == 21:
                    # if self.lb[21].text != '' and self.lb[21] is not None:
                    #     self.recordx(self.lb[21])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[21], bfllwlst[21])))
                elif ind == 22:
                    # if self.lb[22].text != '' and self.lb[22] is not None:
                    #     self.recordx(self.lb[22])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[22], bfllwlst[22])))
                elif ind == 23:
                    # if self.lb[23].text != '' and self.lb[23] is not None:
                    #     self.recordx(self.lb[23])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[23], bfllwlst[23])))
                elif ind == 24:
                    # if self.lb[24].text != '' and self.lb[24] is not None:
                    #     self.recordx(self.lb[24])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[24], bfllwlst[24])))
                elif ind == 25:
                    # if self.lb[25].text != '' and self.lb[25] is not None:
                    #     self.recordx(self.lb[25])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[25], bfllwlst[25])))
                elif ind == 26:
                    # if self.lb[26].text != '' and self.lb[26] is not None:
                    #     self.recordx(self.lb[26])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[26], bfllwlst[26])))
                elif ind == 27:
                    # if self.lb[27].text != '' and self.lb[27] is not None:
                    #     self.recordx(self.lb[27])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[27], bfllwlst[27])))
                elif ind == 28:
                    # if self.lb[28].text != '' and self.lb[28] is not None:
                    #     self.recordx(self.lb[28])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[28], bfllwlst[28])))
                elif ind == 29:
                    # if self.lb[29].text != '' and self.lb[29] is not None:
                    #     self.recordx(self.lb[29])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[29], bfllwlst[29])))
                elif ind == 30:
                    # if self.lb[30].text != '' and self.lb[30] is not None:
                    #     self.recordx(self.lb[30])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[30], bfllwlst[30])))
                elif ind == 31:
                    # if self.lb[31].text != '' and self.lb[31] is not None:
                    #     self.recordx(self.lb[31])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[31], bfllwlst[31])))
                elif ind == 32:
                    # if self.lb[32].text != '' and self.lb[32] is not None:
                    #     self.recordx(self.lb[32])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[32], bfllwlst[32])))
                elif ind == 33:
                    # if self.lb[33].text != '' and self.lb[33] is not None:
                    #     self.recordx(self.lb[33])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[33], bfllwlst[33])))
                elif ind == 34:
                    # if self.lb[34].text != '' and self.lb[34] is not None:
                    #     self.recordx(self.lb[34])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[34], bfllwlst[34])))
                elif ind == 35:
                    # if self.lb[35].text != '' and self.lb[35] is not None:
                    #     self.recordx(self.lb[35])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[35], bfllwlst[35])))
                elif ind == 36:
                    # if self.lb[36].text != '' and self.lb[36] is not None:
                    #     self.recordx(self.lb[36])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[36], bfllwlst[36])))
                elif ind == 37:
                    # if self.lb[37].text != '' and self.lb[37] is not None:
                    #     self.recordx(self.lb[37])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[37], bfllwlst[37])))
                elif ind == 38:
                    # if self.lb[38].text != '' and self.lb[38] is not None:
                    #     self.recordx(self.lb[38])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[38], bfllwlst[38])))
                elif ind == 39:
                    # if self.lb[39].text != '' and self.lb[39] is not None:
                    #     self.recordx(self.lb[39])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[39], bfllwlst[39])))
                elif ind == 40:
                    # if self.lb[40].text != '' and self.lb[40] is not None:
                    #     self.recordx(self.lb[40])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[40], bfllwlst[40])))
                elif ind == 41:
                    # if self.lb[41].text != '' and self.lb[41] is not None:
                    #     self.recordx(self.lb[41])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[41], bfllwlst[41])))
                elif ind == 42:
                    # if self.lb[42].text != '' and self.lb[42] is not None:
                    #     self.recordx(self.lb[42])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[42], bfllwlst[42])))
                elif ind == 43:
                    # if self.lb[43].text != '' and self.lb[43] is not None:
                    #     self.recordx(self.lb[43])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[43], bfllwlst[43])))
                elif ind == 44:
                    # if self.lb[44].text != '' and self.lb[44] is not None:
                    #     self.recordx(self.lb[44])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[44], bfllwlst[44])))
                elif ind == 45:
                    # if self.lb[45].text != '' and self.lb[45] is not None:
                    #     self.recordx(self.lb[45])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[45], bfllwlst[45])))
                elif ind == 46:
                    # if self.lb[46].text != '' and self.lb[46] is not None:
                    #     self.recordx(self.lb[46])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[46], bfllwlst[46])))
                elif ind == 47:
                    # if self.lb[47].text != '' and self.lb[47] is not None:
                    #     self.recordx(self.lb[47])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[47], bfllwlst[47])))
                elif ind == 48:
                    # if self.lb[48].text != '' and self.lb[48] is not None:
                    #     self.recordx(self.lb[48])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[48], bfllwlst[48])))
                elif ind == 49:
                    # if self.lb[49].text != '' and self.lb[49] is not None:
                    #     self.recordx(self.lb[49])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[49], bfllwlst[49])))
                else:
                    # if self.lb[50].text != '' and self.lb[50] is not None:
                    #     self.recordx(self.lb[50])
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.viewprofile(self.lb[50], bfllwlst[50])))
                """View Buttons END"""
                """Follow Buttons START"""
                if ind == 0:
                    # if self.lb[0].text != '' and self.lb[0] is not None:
                    #    self.recordx(self.lb[0])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[0], bfllwlst[0])))
                elif ind == 1:
                    # if self.lb[1].text != '' and self.lb[1] is not None:
                    #    self.recordx(self.lb[1])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[1], bfllwlst[1])))
                elif ind == 2:
                    # if self.lb[2].text != '' and self.lb[2] is not None:
                    #    self.recordx(self.lb[2])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[2], bfllwlst[2])))
                elif ind == 3:
                    # if self.lb[3].text != '' and self.lb[3] is not None:
                    #    self.recordx(self.lb[3])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[3], bfllwlst[3])))
                elif ind == 4:
                    # if self.lb[4].text != '' and self.lb[4] is not None:
                    #     self.recordx(self.lb[4])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[4], bfllwlst[4])))
                elif ind == 5:
                    # if self.lb[5].text != '' and self.lb[5] is not None:
                    #     self.recordx(self.lb[5])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[5], bfllwlst[5])))
                elif ind == 6:
                    # if self.lb[6].text != '' and self.lb[6] is not None:
                    #     self.recordx(self.lb[6])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[6], bfllwlst[6])))
                elif ind == 7:
                    # if self.lb[7].text != '' and self.lb[7] is not None:
                    #     self.recordx(self.lb[7])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[7], bfllwlst[7])))
                elif ind == 8:
                    # if self.lb[8].text != '' and self.lb[8] is not None:
                    #     self.recordx(self.lb[8])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[8], bfllwlst[8])))
                elif ind == 9:
                    # if self.lb[9].text != '' and self.lb[9] is not None:
                    #     self.recordx(self.lb[9])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[9], bfllwlst[9])))
                elif ind == 10:
                    # if self.lb[10].text != '' and self.lb[10] is not None:
                    #     self.recordx(self.lb[10])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[10], bfllwlst[10])))
                elif ind == 11:
                    # if self.lb[11].text != '' and self.lb[11] is not None:
                    #     self.recordx(self.lb[11])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[11], bfllwlst[11])))
                elif ind == 12:
                    # if self.lb[12].text != '' and self.lb[12] is not None:
                    #     self.recordx(self.lb[12])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[12], bfllwlst[12])))
                elif ind == 13:
                    # if self.lb[13].text != '' and self.lb[13] is not None:
                    #     self.recordx(self.lb[13])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[13], bfllwlst[13])))
                elif ind == 14:
                    # if self.lb[14].text != '' and self.lb[14] is not None:
                    #     self.recordx(self.lb[14])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[14], bfllwlst[14])))
                elif ind == 15:
                    # if self.lb[15].text != '' and self.lb[15] is not None:
                    #     self.recordx(self.lb[15])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[15], bfllwlst[15])))
                elif ind == 16:
                    # if self.lb[16].text != '' and self.lb[16] is not None:
                    #     self.recordx(self.lb[16])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[16], bfllwlst[16])))
                elif ind == 17:
                    # if self.lb[17].text != '' and self.lb[17] is not None:
                    #     self.recordx(self.lb[17])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[17], bfllwlst[17])))
                elif ind == 18:
                    # if self.lb[18].text != '' and self.lb[18] is not None:
                    #     self.recordx(self.lb[18])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[18], bfllwlst[18])))
                elif ind == 19:
                    # if self.lb[19].text != '' and self.lb[19] is not None:
                    #     self.recordx(self.lb[19])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[19], bfllwlst[19])))
                elif ind == 20:
                    # if self.lb[20].text != '' and self.lb[20] is not None:
                    #     self.recordx(self.lb[20])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[20], bfllwlst[20])))
                elif ind == 21:
                    # if self.lb[21].text != '' and self.lb[21] is not None:
                    #     self.recordx(self.lb[21])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[21], bfllwlst[21])))
                elif ind == 22:
                    # if self.lb[22].text != '' and self.lb[22] is not None:
                    #     self.recordx(self.lb[22])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[22], bfllwlst[22])))
                elif ind == 23:
                    # if self.lb[23].text != '' and self.lb[23] is not None:
                    #     self.recordx(self.lb[23])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[23], bfllwlst[23])))
                elif ind == 24:
                    # if self.lb[24].text != '' and self.lb[24] is not None:
                    #     self.recordx(self.lb[24])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[24], bfllwlst[24])))
                elif ind == 25:
                    # if self.lb[25].text != '' and self.lb[25] is not None:
                    #     self.recordx(self.lb[25])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[25], bfllwlst[25])))
                elif ind == 26:
                    # if self.lb[26].text != '' and self.lb[26] is not None:
                    #     self.recordx(self.lb[26])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[26], bfllwlst[26])))
                elif ind == 27:
                    # if self.lb[27].text != '' and self.lb[27] is not None:
                    #     self.recordx(self.lb[27])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[27], bfllwlst[27])))
                elif ind == 28:
                    # if self.lb[28].text != '' and self.lb[28] is not None:
                    #     self.recordx(self.lb[28])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[28], bfllwlst[28])))
                elif ind == 29:
                    # if self.lb[29].text != '' and self.lb[29] is not None:
                    #     self.recordx(self.lb[29])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[29], bfllwlst[29])))
                elif ind == 30:
                    # if self.lb[30].text != '' and self.lb[30] is not None:
                    #     self.recordx(self.lb[30])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[30], bfllwlst[30])))
                elif ind == 31:
                    # if self.lb[31].text != '' and self.lb[31] is not None:
                    #     self.recordx(self.lb[31])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[31], bfllwlst[31])))
                elif ind == 32:
                    # if self.lb[32].text != '' and self.lb[32] is not None:
                    #     self.recordx(self.lb[32])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[32], bfllwlst[32])))
                elif ind == 33:
                    # if self.lb[33].text != '' and self.lb[33] is not None:
                    #     self.recordx(self.lb[33])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[33], bfllwlst[33])))
                elif ind == 34:
                    # if self.lb[34].text != '' and self.lb[34] is not None:
                    #     self.recordx(self.lb[34])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[34], bfllwlst[34])))
                elif ind == 35:
                    # if self.lb[35].text != '' and self.lb[35] is not None:
                    #     self.recordx(self.lb[35])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[35], bfllwlst[35])))
                elif ind == 36:
                    # if self.lb[36].text != '' and self.lb[36] is not None:
                    #     self.recordx(self.lb[36])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[36], bfllwlst[36])))
                elif ind == 37:
                    # if self.lb[37].text != '' and self.lb[37] is not None:
                    #     self.recordx(self.lb[37])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[37], bfllwlst[37])))
                elif ind == 38:
                    # if self.lb[38].text != '' and self.lb[38] is not None:
                    #     self.recordx(self.lb[38])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[38], bfllwlst[38])))
                elif ind == 39:
                    # if self.lb[39].text != '' and self.lb[39] is not None:
                    #     self.recordx(self.lb[39])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[39], bfllwlst[39])))
                elif ind == 40:
                    # if self.lb[40].text != '' and self.lb[40] is not None:
                    #     self.recordx(self.lb[40])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[40], bfllwlst[40])))
                elif ind == 41:
                    # if self.lb[41].text != '' and self.lb[41] is not None:
                    #     self.recordx(self.lb[41])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[41], bfllwlst[41])))
                elif ind == 42:
                    # if self.lb[42].text != '' and self.lb[42] is not None:
                    #     self.recordx(self.lb[42])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[42], bfllwlst[42])))
                elif ind == 43:
                    # if self.lb[43].text != '' and self.lb[43] is not None:
                    #     self.recordx(self.lb[43])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[43], bfllwlst[43])))
                elif ind == 44:
                    # if self.lb[44].text != '' and self.lb[44] is not None:
                    #     self.recordx(self.lb[44])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[44], bfllwlst[44])))
                elif ind == 45:
                    # if self.lb[45].text != '' and self.lb[45] is not None:
                    #     self.recordx(self.lb[45])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[45], bfllwlst[45])))
                elif ind == 46:
                    # if self.lb[46].text != '' and self.lb[46] is not None:
                    #     self.recordx(self.lb[46])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[46], bfllwlst[46])))
                elif ind == 47:
                    # if self.lb[47].text != '' and self.lb[47] is not None:
                    #     self.recordx(self.lb[47])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[47], bfllwlst[47])))
                elif ind == 48:
                    # if self.lb[48].text != '' and self.lb[48] is not None:
                    #     self.recordx(self.lb[48])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[48], bfllwlst[48])))
                elif ind == 49:
                    # if self.lb[49].text != '' and self.lb[49] is not None:
                    #     self.recordx(self.lb[49])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[49], bfllwlst[49])))
                else:
                    # if self.lb[50].text != '' and self.lb[50] is not None:
                    #     self.recordx(self.lb[50])
                    self.b2[ind] = (Button(text="Follow", size_hint_y=0.1,
                                           on_press=lambda x:
                                           self.tempfollow(self.lb[50], bfllwlst[50])))
                """Follow Buttons END"""
                """Delete Buttons START"""
                if ind == 0:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[0])))
                elif ind == 1:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[1])))
                elif ind == 2:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[2])))
                elif ind == 3:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[3])))
                elif ind == 4:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[4])))
                elif ind == 5:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[5])))
                elif ind == 6:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[6])))
                elif ind == 7:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[7])))
                elif ind == 8:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[8])))
                elif ind == 9:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[9])))
                elif ind == 10:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[10])))
                elif ind == 11:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[11])))
                elif ind == 12:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[12])))
                elif ind == 13:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[13])))
                elif ind == 14:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[14])))
                elif ind == 15:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[15])))
                elif ind == 16:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[16])))
                elif ind == 17:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[17])))
                elif ind == 18:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[18])))
                elif ind == 19:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[19])))
                elif ind == 20:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[20])))
                elif ind == 21:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[21])))
                elif ind == 22:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[22])))
                elif ind == 23:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[23])))
                elif ind == 24:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[24])))
                elif ind == 25:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[25])))
                elif ind == 26:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[26])))
                elif ind == 27:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[27])))
                elif ind == 28:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[28])))
                elif ind == 29:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[29])))
                elif ind == 30:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[30])))
                elif ind == 31:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[31])))
                elif ind == 32:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[32])))
                elif ind == 33:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[33])))
                elif ind == 34:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[34])))
                elif ind == 35:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[35])))
                elif ind == 36:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[36])))
                elif ind == 37:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[37])))
                elif ind == 38:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[38])))
                elif ind == 39:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[39])))
                elif ind == 40:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[40])))
                elif ind == 41:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[41])))
                elif ind == 42:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[42])))
                elif ind == 43:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[43])))
                elif ind == 44:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[44])))
                elif ind == 45:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[45])))
                elif ind == 46:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[46])))
                elif ind == 47:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[47])))
                elif ind == 48:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[48])))
                elif ind == 49:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[49])))
                else:
                    self.b3[ind] = (Button(text="Hide", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[50])))
                """Delete Buttons END"""
                self.gc[ind].add_widget(self.b[ind])
                self.gc[ind].add_widget(self.b2[ind])
                self.gc[ind].add_widget(self.b3[ind])
                self.box[ind].add_widget(self.lb[ind])
                self.box[ind].add_widget(self.gc[ind])
                self.ids.forbusinesses.add_widget(self.box[ind])
                lst.append([self.box[ind], self.lb[ind], self.gc[ind], self.b[ind], self.b2[ind]])
                print(lst[ind][0])
                print(self.ids.forbusinesses.children)
                addfdnametotext(bfllwlst[ind], fllwfile)
            bindx += 1
        print(lst)


def config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db


class SignUpPage(Screen):
    def connectme(self):
        global user
        cuser = None
        global con
        try:
            cur = con.cursor()
            cur.execute('SELECT version()')
            db_version = cur.fetchone()
            print(db_version)
            cuser = self.ids.usernameu.text
            user = cuser
            password = self.ids.passwordu.text
            query = sql.SQL("CREATE USER {username} WITH PASSWORD {password}").format(
                username=sql.Identifier(cuser),
                password=sql.Placeholder()
            )
            cur.execute(query, (password,))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL ON dietfriend_client_food_data TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL ON client_settings TO {0}").format(
                sql.Identifier(cuser)
            )
            print("absent")
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL ON business_food_search TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            ###########################
            query = sql.SQL("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            ###########################
            str_to_execute = \
                "INSERT INTO dietfriend_usernames_and_passwords(username, password) VALUES(\'" + cuser + "\', \'" + \
                password + "\')"
            cur.execute(str_to_execute)
            cur.execute("COMMIT")
            picture = "blank-account.png"
            str_to_execute = \
                "INSERT INTO client_settings(profile_picture) VALUES(\'" + picture + "\')"
            cur.execute(str_to_execute)
            cur.execute("COMMIT")
            con.close()
            print('Database connection closed.')
            con = psycopg2.connect(database="dietfriendcab", user=cuser,
                                   password=password, host='127.0.0.1', port='5432')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        global theme
        global primary_p
        cur = con.cursor()
        past = defineifmissing_prev_insecure_settings()
        if row_exists_theme(cur, user):
            query = sql.SQL(
                "SELECT bg_theme FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            theme_setting = str(cur.fetchall())
            print("theme_setting:")
            print(theme_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_theme_index = int(random() * 100) % 2
            if random_theme_index == 0:
                theme = "Light"
            else:
                theme = "Dark"
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        if row_exists_primary_p(cur, user):
            query = sql.SQL(
                "SELECT primary_p FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            primary_p_setting = str(cur.fetchall())
            print("primary_p_setting:")
            print(primary_p_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_primary_p_index = int(random() * 100) % 10
            if random_primary_p_index == 0:
                primary_p = "Teal"
            elif random_primary_p_index == 1:
                primary_p = "Red"
            elif random_primary_p_index == 2:
                primary_p = "Pink"
            elif random_primary_p_index == 3:
                primary_p = "Indigo"
            elif random_primary_p_index == 4:
                primary_p = "Blue"
            elif random_primary_p_index == 5:
                primary_p = "LightBlue"
            elif random_primary_p_index == 6:
                primary_p = "Lime"
            elif random_primary_p_index == 7:
                primary_p = "Yellow"
            elif random_primary_p_index == 8:
                primary_p = "Orange"
            else:
                primary_p = "Amber"
            query = sql.SQL(
                "INSERT INTO client_settings (username, primary_p, bg_theme) VALUES (\'" + user + "\', \'" + primary_p + "\', \'" + theme + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            cur.execute("COMMIT")
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        global universal_list

        try:
            print("Signed in, constructing universal_list")
            with open(str(datetime.datetime.now())[0:10] + str(user) + ".txt", 'r') as p:
                lines = p.readlines()
                p.close()
            words = []
            for i in lines:
                words.append(i[0:i.find(' ')])
            with open(str(datetime.datetime.now())[0:10] + str(user) + "_type.txt", 'r') as p:
                lines_two = p.readlines()
                p.close()
            numservings = []
            h = 0
            while h < len(lines_two):
                numservings.append(float(lines_two[h][lines_two[h].find(',') + 1:lines_two[h].find('\n')]))
                h += 1
            universal_list = [[], words, numservings]
            print("universal_list: ")
            print(universal_list)
        except:
            universal_list = [[], [], []]
        # UNCOMMENT ABOVE + COMMENT LINE BELOW THIS FOR REAL PRODUCT

        # universal_list = [[], ['pop_tarts', 'turkey_sticks', 'calcium_powder', 'chili_magic', 'planters_peanuts',
        #                        'buddig_beef', 'buddig_beef', 'tuna_can', 'wolf_brand_chili_magic'],
        #                   [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]]

        # to_add_to_dietfriend_usernames_and_passwords = sql.SQL("")
        # conn = psycopg2.connect(
        #     database="dietfriendcab", user=user, password=password, host='127.0.0.1', port='5432'
        # )


class PrivacyPolicy(Screen):
    pass


class ChildApp(Screen):
    def tempselectimg(self):
        # estdec()
        partialestref()
        newimgsnamesnumservings = getimgs()
        dayfordisplay = doprocess(newimgsnamesnumservings)
        self.calories.text = str(dayfordisplay.totalsfoodlist[1])
        self.total_fat.text = str(dayfordisplay.totalsfoodlist[2])
        self.saturated_fat.text = str(dayfordisplay.totalsfoodlist[3])
        self.trans_fat.text = str(dayfordisplay.totalsfoodlist[4])
        self.cholesterol.text = str(dayfordisplay.totalsfoodlist[5])
        self.sodium.text = str(dayfordisplay.totalsfoodlist[6])
        self.total_carb.text = str(dayfordisplay.totalsfoodlist[7])
        self.fiber.text = str(dayfordisplay.totalsfoodlist[8])
        self.total_sugars.text = str(dayfordisplay.totalsfoodlist[9])
        self.added_sugars.text = str(dayfordisplay.totalsfoodlist[10])
        self.protein.text = str(dayfordisplay.totalsfoodlist[11])
        self.calcium.text = str(dayfordisplay.totalsfoodlist[12])
        self.iron.text = str(dayfordisplay.totalsfoodlist[13])
        self.potassium.text = str(dayfordisplay.totalsfoodlist[14])
        self.vitamin_a.text = str(dayfordisplay.totalsfoodlist[15])
        self.vitamin_b.text = str(dayfordisplay.totalsfoodlist[16])
        self.vitamin_c.text = str(dayfordisplay.totalsfoodlist[17])
        self.vitamin_d.text = str(dayfordisplay.totalsfoodlist[18])

    def selectimg(self):
        newimgsnamesnumservings = getimgs()
        dayfordisplay = doprocess(newimgsnamesnumservings)
        self.calories.text = str(dayfordisplay.totalsfoodlist[1])
        self.total_fat.text = str(dayfordisplay.totalsfoodlist[2])
        self.saturated_fat.text = str(dayfordisplay.totalsfoodlist[3])
        self.trans_fat.text = str(dayfordisplay.totalsfoodlist[4])
        self.cholesterol.text = str(dayfordisplay.totalsfoodlist[5])
        self.sodium.text = str(dayfordisplay.totalsfoodlist[6])
        self.total_carb.text = str(dayfordisplay.totalsfoodlist[7])
        self.fiber.text = str(dayfordisplay.totalsfoodlist[8])
        self.total_sugars.text = str(dayfordisplay.totalsfoodlist[9])
        self.added_sugars.text = str(dayfordisplay.totalsfoodlist[10])
        self.protein.text = str(dayfordisplay.totalsfoodlist[11])
        self.calcium.text = str(dayfordisplay.totalsfoodlist[12])
        self.iron.text = str(dayfordisplay.totalsfoodlist[13])
        self.potassium.text = str(dayfordisplay.totalsfoodlist[14])
        self.vitamin_a.text = str(dayfordisplay.totalsfoodlist[15])
        self.vitamin_b.text = str(dayfordisplay.totalsfoodlist[16])
        self.vitamin_c.text = str(dayfordisplay.totalsfoodlist[17])
        self.vitamin_d.text = str(dayfordisplay.totalsfoodlist[18])

    def estandrefall(self):
        dayfordisplayestandref = doprocessestref()
        dayfordisplayestandrefs = dayfordisplayestandref
        dayfordisplayestandrefs = doprocessestref()
        self.calories.text = str(dayfordisplayestandrefs.totalsfoodlist[1])
        self.total_fat.text = str(dayfordisplayestandrefs.totalsfoodlist[2])
        self.saturated_fat.text = str(dayfordisplayestandrefs.totalsfoodlist[3])
        self.trans_fat.text = str(dayfordisplayestandrefs.totalsfoodlist[4])
        self.cholesterol.text = str(dayfordisplayestandrefs.totalsfoodlist[5])
        self.sodium.text = str(dayfordisplayestandrefs.totalsfoodlist[6])
        self.total_carb.text = str(dayfordisplayestandrefs.totalsfoodlist[7])
        self.fiber.text = str(dayfordisplayestandrefs.totalsfoodlist[8])
        self.total_sugars.text = str(dayfordisplayestandrefs.totalsfoodlist[9])
        self.added_sugars.text = str(dayfordisplayestandrefs.totalsfoodlist[10])
        self.protein.text = str(dayfordisplayestandrefs.totalsfoodlist[11])
        self.calcium.text = str(dayfordisplayestandrefs.totalsfoodlist[12])
        self.iron.text = str(dayfordisplayestandrefs.totalsfoodlist[13])
        self.potassium.text = str(dayfordisplayestandrefs.totalsfoodlist[14])
        self.vitamin_a.text = str(dayfordisplayestandrefs.totalsfoodlist[15])
        self.vitamin_b.text = str(dayfordisplayestandrefs.totalsfoodlist[16])
        self.vitamin_c.text = str(dayfordisplayestandrefs.totalsfoodlist[17])
        self.vitamin_d.text = str(dayfordisplayestandrefs.totalsfoodlist[18])

    def frevertall(self):
        dayfordisplayfullreverted = doprocessrevertall()
        self.calories.text = str(dayfordisplayfullreverted.totalsfoodlist[1])
        self.total_fat.text = str(dayfordisplayfullreverted.totalsfoodlist[2])
        self.saturated_fat.text = str(dayfordisplayfullreverted.totalsfoodlist[3])
        self.trans_fat.text = str(dayfordisplayfullreverted.totalsfoodlist[4])
        self.cholesterol.text = str(dayfordisplayfullreverted.totalsfoodlist[5])
        self.sodium.text = str(dayfordisplayfullreverted.totalsfoodlist[6])
        self.total_carb.text = str(dayfordisplayfullreverted.totalsfoodlist[7])
        self.fiber.text = str(dayfordisplayfullreverted.totalsfoodlist[8])
        self.total_sugars.text = str(dayfordisplayfullreverted.totalsfoodlist[9])
        self.added_sugars.text = str(dayfordisplayfullreverted.totalsfoodlist[10])
        self.protein.text = str(dayfordisplayfullreverted.totalsfoodlist[11])
        self.calcium.text = str(dayfordisplayfullreverted.totalsfoodlist[12])
        self.iron.text = str(dayfordisplayfullreverted.totalsfoodlist[13])
        self.potassium.text = str(dayfordisplayfullreverted.totalsfoodlist[14])
        self.vitamin_a.text = str(dayfordisplayfullreverted.totalsfoodlist[15])
        self.vitamin_b.text = str(dayfordisplayfullreverted.totalsfoodlist[16])
        self.vitamin_c.text = str(dayfordisplayfullreverted.totalsfoodlist[17])
        self.vitamin_d.text = str(dayfordisplayfullreverted.totalsfoodlist[18])

    def mibuttonchoice(self, typeg):
        typefile: str = defineifmissingmisessiontype()
        with open(typefile, 'w') as w:
            w.write(typeg)
            w.close()
        return typefile


class ImageSelection(Screen):
    def selected(self, filename):
        try:
            self.ids.imgtoselect.source = filename[0]
            print(filename[0])
        except:
            pass

    def select(self, filename):
        global universal_list
        global user
        try:
            ppath = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
            fname_list = listdir(ppath)
            disconnected_fname_list = fname_list
            if len(fname_list) > 0:
                fname = priorname(priorfname=disconnected_fname_list[0], disconnected_fname_list=disconnected_fname_list)
            else:
                fname = '1_food_picture.jpg'
            path_img = os.path.join(os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user), fname)
            cwd = os.getcwd()
            os.chdir(os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user))
            print("IMAGE PATH")
            print(path_img)
            filenamed = cv2.imread(filename[0])
            picture = cv2.imwrite(path_img, filenamed)
            os.chdir(cwd)
            print("ORIGINAL universal_list:")
            print(universal_list)
            universal_list[0].append(fname)
            self.manager.current = "foodnamer"
        except:
            pass


class BusinessImageSelection(Screen):
    def selected(self, filename):
        try:
            self.ids.imgtoselect.source = filename[0]
            print(filename[0])
        except:
            pass

    def select(self, filename):
        global universal_list
        global user
        try:
            ppath = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
            fname_list = listdir(ppath)
            disconnected_fname_list = fname_list
            if len(fname_list) > 0:
                fname = priorname(priorfname=disconnected_fname_list[0], disconnected_fname_list=disconnected_fname_list)
            else:
                fname = '1_food_picture.jpg'
            # Original Below line for all path_img alike: os.path.dirname(os.path.realpath('DietFriend')) + "\\DietFriend_Pictures" + user + "\\" + fname
            path_img = os.path.join(os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user), fname)
            cwd = os.getcwd()
            os.chdir(os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user))
            print("IMAGE PATH")
            print(path_img)
            filenamed = cv2.imread(filename[0])
            picture = cv2.imwrite(path_img, filenamed)
            os.chdir(cwd)
            print("ORIGINAL universal_list:")
            print(universal_list)
            universal_list[0].append(fname)
            self.manager.current = "foodnamer"
        except:
            pass

# def img_take():
#     videocaptureobject = cv2.VideoCapture(0)
#     result = True
#     while result:
#         ret, frame = videocaptureobject.read()
#         datename = str(datetime.datetime.now())
#         datename = datename.replace(" ", "_")
#         datenamer = datename + ".jpg"
#         cv2.imwrite(datenamer, frame)
#         result = False
#     videocaptureobject.release()
#     cv2.destroyAllWindows()
#     return videocaptureobject
#
#
# def defineifmissingimgtaken():
#     crd = doprocess(getimgs())
#     curdayupdtd = str(crd.date_time)
#
#     try:
#         fo = open(curtxt, 'r')
#         fo.close()
#     except FileNotFoundError:
#         fm = open(curtxt, 'w')
#         fm.close()
#     return curtxt
#
#
# def getlen(vle):
#     with open(vle, 'r') as vre:
#         xer = vre.readlines()
#         vre.close()
#     print("No error?")
#     print(len(xer))
#     return len(xer)
#
#
# def updateimgtke():
#     imgtke = defineifmissingimgtaken()
#     with open(imgtke, 'a') as q:
#         q.write(dtme+'\n')
#         q.close()


def adjust_for_prior(fname):
    fnamestr = str(fname)
    fnamestr = '9' + fnamestr
    # fnamestr = fnamestr[0:fnamestr.find('_')] + '9' + fnamestr[fnamestr.find('_'):fnamestr.find('.')] + '9' + fnamestr[fnamestr.find('.'):len(fnamestr)]
    return fnamestr
    # TRY:return '0_'+fnamestr


def priorname(priorfname, disconnected_fname_list):
    priorfname = adjust_for_prior(priorfname)
    disconnected_fname_list.append(priorfname)
    print("Priorfname: ")
    print(priorfname)
    disconnected_fname_list = sorted(disconnected_fname_list)
    print("Priorfname after sorted: ")
    print(priorfname)
    print(disconnected_fname_list)
    if disconnected_fname_list[len(disconnected_fname_list) - 1] == priorfname:
        print("RETURN")
        print(priorfname)
        return priorfname
    else:
        j = 0
        for i in disconnected_fname_list:
            if i == priorfname:
                disconnected_fname_list.pop(j)
            j += 1
        priorname(priorfname=priorfname, disconnected_fname_list=disconnected_fname_list)


class ImageTake(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.streamvideocaptureobject = None
        self.image_frame = None
        # self.flip_image_frame = None

    def streame(self, *args):
        """Some cache edit still required"""
        self.streamvideocaptureobject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)

    def load_video(self, *args):
        ret, frame = self.streamvideocaptureobject.read()
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tobytes()
        # self.flip_image_frame = cv2.flip(frame, 0)
        # noinspection PyArgumentList
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.curimage.texture = texture

    def turn_off(self):
        Clock.unschedule(self.load_video, 1.0 / 30.0)
        self.streamvideocaptureobject.release()
        cv2.destroyAllWindows()

    # cv2.imshow('Capturing Video', frame)
    # image = pygame.image.fromstring(frame.tobytes(), (640, 480), "RGB")  # convert received image from string

    # pygame.image.save(image, "imagefilerdf.jpg")
    # data = io.BytesIO(open("imagefilerdf.jpg", "rb").read())
    # im = CoreImage(data, ext="jpg")
    # im.save("imagefilerdf.jpg")
    # self.ids.curimage.reload()

    # if (cv2.waitKey(1) & 0xFF == ord('q')) or getlen(imgt) > gl:
    #         #    self.streamvideocaptureobject.release()
    #         #    cv2.destroyAllWindows()
    #         #    break

    def take_picture(self, *args):
        global universal_list
        global user
        ppath = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
        fname_list = listdir(ppath)
        disconnected_fname_list = fname_list
        if len(fname_list) > 0:
            fname = priorname(priorfname=disconnected_fname_list[0], disconnected_fname_list=disconnected_fname_list)
        else:
            fname = '1_food_picture.jpg'
        path_img = os.path.join(os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user), fname)
        cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user))
        print("IMAGE PATH")
        print(path_img)
        picture = cv2.imwrite(path_img, self.image_frame)
        os.chdir(cwd)
        print("ORIGINAL universal_list:")
        print(universal_list)
        universal_list[0].append(fname)


#    def updatecanvas(self):
#        imgtke = defineifmissingimgtaken()
#        x = getlen(imgtke)
#        self.streame(imgtke, x)
# Create new img file to add to DietFriend_Pictures; then doprocess(getimgs())


class BusinessImageTake(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.streamvideocaptureobject = None
        self.image_frame = None
        # self.flip_image_frame = None

    def streame(self, *args):
        """Some cache edit still required"""
        self.streamvideocaptureobject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)

    def load_video(self, *args):
        ret, frame = self.streamvideocaptureobject.read()
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tobytes()
        # self.flip_image_frame = cv2.flip(frame, 0)
        # noinspection PyArgumentList
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.curimage.texture = texture

    def turn_off(self):
        Clock.unschedule(self.load_video, 1.0 / 30.0)
        self.streamvideocaptureobject.release()
        cv2.destroyAllWindows()

    def take_picture(self, *args):
        global universal_list
        global user
        ppath = os.path.join(os.path.dirname(__file__), "B_DietFriend_Pictures"+user)
        fname_list = listdir(ppath)
        disconnected_fname_list = fname_list
        if len(fname_list) > 0:
            fname = priorname(priorfname=disconnected_fname_list[0], disconnected_fname_list=disconnected_fname_list)
        else:
            fname = '1_food_picture.jpg'
        path_img = os.path.join(os.path.join(os.path.dirname(__file__), "B_DietFriend_Pictures"+user), fname)
        cwd = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), "B_DietFriend_Pictures"+user))
        print("IMAGE PATH")
        print(path_img)
        picture = cv2.imwrite(path_img, self.image_frame)
        os.chdir(cwd)
        print("ORIGINAL universal_list:")
        print(universal_list)
        universal_list[0].append(fname)


class FoodNamingAfterImage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fnai_dropdown = None
        self.fnai_dd_item_dict = {}
        self.fnai_items_to_add = []
        self.fnai_menu = []

    def update_suggestions_menu(self):
        if self.fnai_dropdown is not None:
            self.destroy_()
        items = autofind(self.ids.fd_name.text)
        print("items in order for update:")
        print(items)
        for item in items:
            self.fnai_items_to_add.append(item[0])
        self.dropdown_()

    def dropdown_(self):
        i = 0
        while i < len(self.fnai_items_to_add):
            if len(self.fnai_dd_item_dict) <= 10:
                try:
                    self.fnai_dd_item_dict[i] = self.fnai_items_to_add[i]
                except IndexError:
                    self.fnai_dd_item_dict[i] = ''
            if i == 0:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[0],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[0]),
                })
            elif i == 1:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[1],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[1]),
                })
            elif i == 2:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[2],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[2]),
                })
            elif i == 3:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[3],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[3]),
                })
            elif i == 4:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[4],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[4]),
                })
            elif i == 5:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[5],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[5]),
                })
            elif i == 6:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[6],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[6]),
                })
            elif i == 7:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[7],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[7]),
                })
            elif i == 8:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[8],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[8]),
                })
            elif i == 9:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[9],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[9]),
                })
            else:
                self.fnai_menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.fnai_dd_item_dict[10],
                    "on_release": lambda: self.menu_callback(instance=self.fnai_dd_item_dict[10]),
                })
            i += 1
        self.fnai_dropdown = MDDropdownMenu(caller=self.ids.fd_name, items=self.fnai_menu, width_mult=4)
        print("self.dropdown.items: ")
        print(self.fnai_dropdown.items)
        self.fnai_dropdown.open()

    def destroy_(self):
        self.fnai_dropdown.dismiss()
        self.fnai_dd_item_dict = {}
        self.fnai_menu.clear()
        self.fnai_items_to_add.clear()
        self.fnai_dropdown.clear_widgets()
        self.fnai_dropdown.items.clear()
        print("Should be []")
        print(self.fnai_dropdown.items)

    def menu_callback(self, instance):
        print(instance)
        self.ids.fd_name.text = instance

    def namefood(self):
        global universal_list
        universal_list[1].append(self.ids.fd_name.text.replace(" ", "_"))
        tmplst = universal_list[2]
        tmplst.append(float(self.ids.fd_serving.text))
        universal_list[2] = tmplst
        print("NEW universal_list:")
        print(universal_list)
        self.manager.current = "childapp"


class BusinessFoodNamingAfterImage(Screen):
    def namefood(self):
        global universal_list
        universal_list[1].append(self.ids.fd_name.text.replace(" ", "_"))
        tmplst = universal_list[2]
        tmplst.append(1.00)
        universal_list[2] = tmplst
        print("NEW universal_list:")
        print(universal_list)
        self.manager.current = "businesshome"


class ForEdit:
    def __init__(self, name):
        self.name = name


def checkes(strfc, cd, indx):
    lstforcheck = []
    m = 0
    numrs = -1
    while m < len(strfc) - 2:
        h = strfc[m:m + 1]
        g = strfc[m + 1:m + 2]
        if h == 'r' and g == ',':
            numrs += 1
            numhg = strfc[m + 2:strfc[m + 2:len(strfc)].find(' ') + m + 2]
            lstforcheck.append([numrs, m, numhg])
        m += 1
    print(lstforcheck)
    q = defineifmissingestref(cd)
    with open(q, 'r') as qrs:
        w = qrs.readlines()
        qrs.close()
    fdest = getfd(w[indx])
    g = 2
    while g < len(lstforcheck):
        if float(lstforcheck[g][2]) == float(getattr(fdest, foodattrnamelst[g - 2])):
            print("CORRECT!")
            strfc = strfc[0:lstforcheck[g][1]] + 'e' + strfc[lstforcheck[g][1] + 1: len(strfc)]
        g += 1
    print(strfc)
    return strfc


def b_checkes(strfc, cd, indx):
    lstforcheck = []
    m = 0
    numrs = -1
    while m < len(strfc) - 2:
        h = strfc[m:m + 1]
        g = strfc[m + 1:m + 2]
        if h == 'r' and g == ',':
            numrs += 1
            numhg = strfc[m + 2:strfc[m + 2:len(strfc)].find(' ') + m + 2]
            lstforcheck.append([numrs, m, numhg])
        m += 1
    print(lstforcheck)
    q = b_defineifmissingestref()
    with open(q, 'r') as qrs:
        w = qrs.readlines()
        qrs.close()
    fdest = getfd(w[indx])
    g = 2
    while g < len(lstforcheck):
        if float(lstforcheck[g][2]) == float(getattr(fdest, foodattrnamelst[g - 2])):
            print("CORRECT!")
            strfc = strfc[0:lstforcheck[g][1]] + 'e' + strfc[lstforcheck[g][1] + 1: len(strfc)]
        g += 1
    print(strfc)
    return strfc


def estimatefromcustomchecker(indx, individualsf, cd):
    """Returns True if custom data is actually entirely equal to estimated data, else False"""
    # Prerequisites: individualsf = defineifmissingindividuals(cd)
    with open(individualsf, 'r') as iec:
        yf = iec.readlines()
        iec.close()
    strforchecking = yf[indx]
    strforchecking = checkes(strforchecking, cd, indx)
    """Once \'e,\''s are set..."""
    total = 0
    check = 0
    while check < len(strforchecking) - 2:
        e = strforchecking[check:check + 1]
        p = strforchecking[check + 1:check + 2]
        if e == 'e' and p == ',':
            total += 1
        check += 1
    print(total)
    if total == 19:
        return True
    return False


def b_estimatefromcustomchecker(indx, individualsf, cd):
    """Returns True if custom data is actually entirely equal to estimated data, else False"""
    # Prerequisites: individualsf = defineifmissingindividuals(cd)
    with open(individualsf, 'r') as iec:
        yf = iec.readlines()
        iec.close()
    strforchecking = yf[indx]
    strforchecking = b_checkes(strforchecking, cd, indx)
    """Once \'e,\''s are set..."""
    total = 0
    check = 0
    while check < len(strforchecking) - 2:
        e = strforchecking[check:check + 1]
        p = strforchecking[check + 1:check + 2]
        if e == 'e' and p == ',':
            total += 1
        check += 1
    print(total)
    if total == 19:
        return True
    return False


def getfd(strt):
    tstrt = strt
    lstfdset = []
    n = 0
    print(tstrt)
    tstrt.replace('\n', ' ')
    tstrt += '\n'
    while n < len(tstrt):
        if tstrt[n:n + 1] == ' ' or tstrt[n:n + 1] == '\n':
            c = ''
            try:
                c = tstrt[0:n]
            except:
                pass
            lstfdset.append(c)
            print(c)
            tstrt = tstrt[n + 1: len(tstrt)]
            n = -1
        n += 1
    lstfdset.pop(21)
    print(lstfdset)
    fd = Food(lstfdset[0], lstfdset[1], lstfdset[2], lstfdset[3], lstfdset[4], lstfdset[5], lstfdset[6], lstfdset[7],
              lstfdset[8], lstfdset[9], lstfdset[10], lstfdset[11], lstfdset[12], lstfdset[13], lstfdset[14],
              lstfdset[15], lstfdset[16], lstfdset[17], lstfdset[18], lstfdset[19], lstfdset[20])
    return fd


def individualssuperwriteest(cd, bxstoch, ftindd):
    def isinx(val, lsttup):
        xerr = 0
        while xerr < len(lsttup):
            if lsttup[xerr][1] == val:
                return True
            xerr += 1
        return False

    k = defineifmissingindividuals(cd)
    with open(k, 'r') as kk:
        kkk = kk.readlines()
        kk.close()
    indicestorep = []
    o = 0
    while o < len(bxstoch):
        indicestorep.append(bxstoch[o][1])
        o += 1
    indexgetterstr = kkk[ftindd]
    indicesofnon_e_r_s = []
    p = 0
    x = -1
    while p < len(indexgetterstr) - 1:
        curchar = indexgetterstr[p:p + 1]
        if curchar == 'r' or curchar == 'n' or curchar == 'e':
            x += 1
        if curchar == 'r' or curchar == 'n':
            indicesofnon_e_r_s.append((p, x))
        p += 1
    w = 0
    xx = -1
    while w < len(indexgetterstr) - 1:
        curchar = indexgetterstr[w:w + 1]
        if curchar == 'r' or curchar == 'e' or curchar == 'n' and indexgetterstr[w + 1:w + 2] == ',':
            xx += 1
        i = 0
        while i < len(indicesofnon_e_r_s):
            if w == indicesofnon_e_r_s[i][0] and xx == indicesofnon_e_r_s[i][1] and isinx(xx, bxstoch):
                indexgetterstr = indexgetterstr[0:w] + 'e' + indexgetterstr[w + 1:len(indexgetterstr)]
            i += 1
        w += 1
    writetolineindividuals(cd, ftindd + 1, '\n')
    writetolineindividuals(cd, ftindd + 1, indexgetterstr)


def b_individualssuperwriteest(cd, bxstoch, ftindd):
    def isinx(val, lsttup):
        xerr = 0
        while xerr < len(lsttup):
            if lsttup[xerr][1] == val:
                return True
            xerr += 1
        return False

    k = b_defineifmissingindividuals()
    with open(k, 'r') as kk:
        kkk = kk.readlines()
        kk.close()
    indicestorep = []
    o = 0
    while o < len(bxstoch):
        indicestorep.append(bxstoch[o][1])
        o += 1
    indexgetterstr = kkk[ftindd]
    indicesofnon_e_r_s = []
    p = 0
    x = -1
    while p < len(indexgetterstr) - 1:
        curchar = indexgetterstr[p:p + 1]
        if curchar == 'r' or curchar == 'n' or curchar == 'e':
            x += 1
        if curchar == 'r' or curchar == 'n':
            indicesofnon_e_r_s.append((p, x))
        p += 1
    w = 0
    xx = -1
    while w < len(indexgetterstr) - 1:
        curchar = indexgetterstr[w:w + 1]
        if curchar == 'r' or curchar == 'e' or curchar == 'n' and indexgetterstr[w + 1:w + 2] == ',':
            xx += 1
        i = 0
        while i < len(indicesofnon_e_r_s):
            if w == indicesofnon_e_r_s[i][0] and xx == indicesofnon_e_r_s[i][1] and isinx(xx, bxstoch):
                indexgetterstr = indexgetterstr[0:w] + 'e' + indexgetterstr[w + 1:len(indexgetterstr)]
            i += 1
        w += 1
    b_writetolineindividuals(cd, ftindd + 1, '\n')
    b_writetolineindividuals(cd, ftindd + 1, indexgetterstr)


class FoodListPopUp(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.box = None
        self.gc = {}
        self.lb = {}
        self.estonlyindividual = {}
        self.textinput = {}
        self.estallindividual = None
        self.btn = None

    def setter(self, cd, bxstoch, ftindd):
        individualssuperwriteest(cd, bxstoch, ftindd)
        est = defineifmissingestref(cd)
        estdec()
        with open(est, 'r') as d:
            dd = d.readlines()
            d.close()
        forester = dd[ftindd]
        print("HERE FORESTER")
        ester = getfd(forester)
        global est_serving_size
        est_serving_size = getattr(ester, foodattrnamelst[0])
        z = 1
        while z < len(bxstoch):
            self.textinput[z].text = getattr(ester, foodattrnamelst[z])
            z += 1

    def loader(self, ftind):
        cd = doprocess(getimgs())
        global est_serving_size
        est_serving_size = None
        self.title = "Edit Food: " + cd.usedfoodsincount[ftind].food_name + ":"
        ind = 0
        self.box = (BoxLayout(orientation='vertical', padding=[10, 10, 10, 10]))
        self.estallindividual = Button(text="Estimate All",
                                       on_release=lambda x: self.setter(cd, [(self.textinput[0].text, 2),
                                                                             (self.textinput[1].text, 3),
                                                                             (self.textinput[2].text, 4),
                                                                             (self.textinput[3].text, 5),
                                                                             (self.textinput[4].text, 6),
                                                                             (self.textinput[5].text, 7),
                                                                             (self.textinput[6].text, 8),
                                                                             (self.textinput[7].text, 9),
                                                                             (self.textinput[8].text, 10),
                                                                             (self.textinput[9].text, 11),
                                                                             (self.textinput[10].text, 12),
                                                                             (self.textinput[11].text, 13),
                                                                             (self.textinput[12].text, 14),
                                                                             (self.textinput[13].text, 15),
                                                                             (self.textinput[14].text, 16),
                                                                             (self.textinput[15].text, 17),
                                                                             (self.textinput[16].text, 18),
                                                                             (self.textinput[17].text, 19),
                                                                             (self.textinput[18].text, 20)], ftind))
        self.box.add_widget(self.estallindividual)
        while ind < len(foodattrnamelst):
            self.gc[ind] = (GridLayout(cols=3))
            self.lb[ind] = (Label(text=foodattrnamelst[ind], size_hint_y=0.1))
            self.estonlyindividual[ind] = (Button(text="Estimate",
                                                  on_release=lambda x: self.setter(cd,
                                                                                   [(self.textinput[ind].text, ind + 2)],
                                                                                                    ftind)))
            self.textinput[ind] = (TextInput(text=str(getattr(cd.usedfoodsincount[ftind], foodattrnamelst[ind])),
                                             size_hint_y=0.1))
            self.gc[ind].add_widget(self.lb[ind])
            self.gc[ind].add_widget(self.textinput[ind])
            self.box.add_widget(self.gc[ind])
            ind += 1
        lk = defineifmissingtype(cd)
        with open(lk, 'r') as llk:
            cllk = llk.readlines()[ftind].replace('t', '').replace('r', '').replace('e', '').replace('n', '').replace(',', '').replace('\n', '')
            llk.close()
        self.textinput[0].text = str(cllk)
        self.btn = Button(text="Apply Changes", on_release=lambda x: self.returner(cd, ftind,
                                                                                   [self.textinput[0].text,
                                                                                    self.textinput[1].text,
                                                                                    self.textinput[2].text,
                                                                                    self.textinput[3].text,
                                                                                    self.textinput[4].text,
                                                                                    self.textinput[5].text,
                                                                                    self.textinput[6].text,
                                                                                    self.textinput[7].text,
                                                                                    self.textinput[8].text,
                                                                                    self.textinput[9].text,
                                                                                    self.textinput[10].text,
                                                                                    self.textinput[11].text,
                                                                                    self.textinput[12].text,
                                                                                    self.textinput[13].text,
                                                                                    self.textinput[14].text,
                                                                                    self.textinput[15].text,
                                                                                    self.textinput[16].text,
                                                                                    self.textinput[17].text,
                                                                                    self.textinput[18].text
                                                                                    ]))
        self.box.add_widget(self.btn)
        self.add_widget(self.box)
        self.open()

    def returner(self, cd, f_ind, textinputlst):
        global est_serving_size
        global con
        global user
        curcursor = con.cursor()
        lstofchanged = []
        y = 1
        while y < len(foodattrnamelst):
            if float(textinputlst[y]) != float(getattr(cd.usedfoodsincount[f_ind], foodattrnamelst[y])):
                lstofchanged.append((foodattrnamelst[y], float(textinputlst[y])))
            y += 1
        if est_serving_size is not None:
            fd = Food(getattr(cd.usedfoodsincount[f_ind], 'food_name'),
                      getattr(cd.usedfoodsincount[f_ind], 'food_datetime'),
                      float(est_serving_size), float(textinputlst[1]), float(textinputlst[2]), float(textinputlst[3]),
                      float(textinputlst[4]), float(textinputlst[5]), float(textinputlst[6]), float(textinputlst[7]),
                      float(textinputlst[8]), float(textinputlst[9]), float(textinputlst[10]), float(textinputlst[11]),
                      float(textinputlst[12]), float(textinputlst[13]), float(textinputlst[14]), float(textinputlst[15]),
                      float(textinputlst[16]), float(textinputlst[17]), float(textinputlst[18]))
        else:
            fd = Food(getattr(cd.usedfoodsincount[f_ind], 'food_name'),
                      getattr(cd.usedfoodsincount[f_ind], 'food_datetime'),
                      getattr(cd.usedfoodsincount[f_ind], 'serving'), float(textinputlst[1]), float(textinputlst[2]),
                      float(textinputlst[3]),
                      float(textinputlst[4]), float(textinputlst[5]), float(textinputlst[6]), float(textinputlst[7]),
                      float(textinputlst[8]), float(textinputlst[9]), float(textinputlst[10]), float(textinputlst[11]),
                      float(textinputlst[12]), float(textinputlst[13]), float(textinputlst[14]),
                      float(textinputlst[15]),
                      float(textinputlst[16]), float(textinputlst[17]), float(textinputlst[18]))
        tempfd = cd.usedfoodsincount[f_ind]
        cd.usedfoodsincount[f_ind] = fd
        fileindivid = defineifmissingindividuals(cd)
        strtowrite = ""
        i = 0
        while i < 21:
            if i >= 2:
                ii = i - 2
                strtowrite += ('r,' + str(getattr(cd.usedfoodsincount[f_ind], foodattrnamelst[ii])) + ' ')
            elif i == 1:
                strtowrite += ('r,' + str(getattr(cd.usedfoodsincount[f_ind], 'food_datetime')) + ' ')
            else:
                strtowrite += ('r,' + str(getattr(cd.usedfoodsincount[f_ind], 'food_name')) + ' ')
            i += 1
        strtowrite += '\n'
        writetolineindividuals(cd, f_ind + 1, '\n')
        writetolineindividuals(cd, f_ind + 1, strtowrite)
        foodsubtractpreadd(cd, tempfd, fd)
        dst = defineifmissingtype(cd)
        with open(dst, 'r') as dstr:
            f = dstr.readlines()
            dstr.close()
        strforwrite = f[f_ind]
        # orignumservings = strforwrite[strforwrite.find(',')+1:strforwrite.find('\n')]
        strforwrite = strforwrite.replace('t', 'r')
        strforwrite = strforwrite.replace('e', 'r')
        strforwrite = strforwrite[0:strforwrite.find(',')+1] + str(textinputlst[0]).replace('\n', '') + "\n"
        templst = universal_list[2]
        templst[f_ind] = float(str(textinputlst[0]).replace('\n', ''))
        universal_list[2] = templst
        # ratio = float(str(textinputlst[0]).replace('\n', ''))/float(orignumservings)
        # dxt = 1
        # while dxt < len(foodattrnamelst):
        #     strcreate = '_im_two' + foodattrnamelst[dxt]
        #     dxtx = defineifmissingmisessiontwo(cd, foodattrnamelst[dxt])
        #     if row_exists_moreinfo(curcursor, user, str(datetime.datetime.now())[0:10]):
        #         query = sql.SQL(
        #             "SELECT " + strcreate + " FROM client_moreinfo_value_storage WHERE (username = \'" + user + "\') AND (datetime = \'" + str(
        #                 datetime.datetime.now())[0:10] + "\')")
        #         print("Query: ")
        #         print(query)
        #         curcursor.execute(query)
        #         towrt = str(curcursor.fetchall()).replace('\'', '').replace('None', '').replace('[', '').replace('(',
        #                                                                                                       '').replace(
        #             ')', '').replace(']', '').replace('\\n', '\n')
        #         print("TOWRT")
        #         print(towrt)
        #         if towrt.find('x') == -1 and towrt.find('i') == -1:
        #             pass
        #         else:
        #             with open(dxtx, 'w') as mfltwo:
        #                 mfltwo.write(towrt)
        #                 mfltwo.close()
        #     else:
        #         query = sql.SQL(
        #             "INSERT INTO client_moreinfo_value_storage (username, datetime) VALUES (\'" + user + "\', \'" + str(
        #                 datetime.datetime.now())[0:10] + "\')")
        #         print("Query: ")
        #         print(query)
        #         curcursor.execute(query)
        #         curcursor.execute("COMMIT")
        #     try:
        #         with open(dxtx, 'r') as dxtxr:
        #             hghf = dxtxr.readlines()
        #             dxtxr.close()
        #         hghf[f_ind] = hghf[f_ind][0:hghf[f_ind].find(',')+1] + \
        #                       str(float(hghf[f_ind][hghf[f_ind].find(',') + 1:hghf[f_ind].find('\n')])*ratio) + "\n"
        #         stringer = ""
        #         for p in hghf:
        #             stringer += p
        #         with open(dxtx, 'w') as dxtxw:
        #             dxtxw.write(stringer)
        #             dxtxw.close()
        #         str_to_execute = "UPDATE client_moreinfo_value_storage SET " + strcreate + " = \'" + stringer.replace(
        #             'None',
        #             '').replace(
        #             '\\n', '\n') + "\' WHERE username = \'" + user + "\' AND datetime = \'" + str(datetime.datetime.now())[0:10] + "\'"
        #         print(str_to_execute)
        #         curcursor.execute(str_to_execute)
        #         curcursor.execute("COMMIT")
        #     except IndexError:
        #         pass
        #     dxt += 1
        writetoline(cd, f_ind + 1, '\n')
        writetoline(cd, f_ind + 1, strforwrite)
        if estimatefromcustomchecker(f_ind, fileindivid, cd):
            with open(dst, 'r') as dstr:
                f = dstr.readlines()
                dstr.close()
            strforwrite = f[f_ind]
            strforwrite = strforwrite.replace('r', 'e')
            writetoline(cd, f_ind + 1, '\n')
            writetoline(cd, f_ind + 1, strforwrite)

        datestring = cd.date_time
        datestring = str(datestring)
        datestring = datestring[0:10]
        indtext = defineifmissingindividuals(cd)
        with open(indtext, 'r') as readind:
            indtextlist = readind.readlines()
            print("indtextlist: ")
            readind.close()
        indtxt = ""
        u = 0
        while u < len(indtextlist):
            indtxt += indtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET individualstextfile_r = \'" + indtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        typtext = defineifmissingtype(cd)
        with open(typtext, 'r') as readtyp:
            typtextlist = readtyp.readlines()
            print("typtextlist: ")
            readtyp.close()
        typtxt = ""
        u = 0
        while u < len(typtextlist):
            typtxt += typtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")
        estermoreinfos_focused_basedonlst(cd, lstofchanged, f_ind)
        self.remove_widget(self.box)
        self.dismiss()


def foodsubtractpreadd(cd, tfd, nfd):
    uyt = 0
    while uyt < len(cd.totalsfoodlist):
        if getattr(tfd, foodattrnamelst[uyt]) != -2000.0:
            cd.totalsfoodlist[uyt] -= getattr(tfd, foodattrnamelst[uyt])
        uyt += 1
    uyt = 0
    while uyt < len(cd.totalsfoodlist):
        if getattr(nfd, foodattrnamelst[uyt]) != -2000.0:
            cd.totalsfoodlist[uyt] += getattr(nfd, foodattrnamelst[uyt])
        uyt += 1


def getservingamt(x, crd):
    """Returns NumServings for Food"""
    h = defineifmissingtype(crd)
    with open(h, 'r') as rd:
        p = rd.readlines()
        rd.close()
    stringx = p[x]
    stringxp = stringx[stringx.find(',') + 1:len(stringx)]
    return stringxp


def getallservingamts(date):
    p = defineifmissing_type_date(date)
    with open(p, 'r') as hl:
        yy = hl.readlines()
        hl.close()
    t = 0
    while t < len(yy):
        y = yy[t].strip()
        y = y[y.find(',') + 1:len(y)]
        yy[t] = y
        t += 1
    return yy


def popupeditgetter(ndx, crd):
    """Constructs FoodListPopUp, Returns Array of New Attributes or newFood that ~replaces~ old food"""
    fdlstpopup = FoodListPopUp()
    fdlstpopup.loader(ndx)
    # time.sleep(16)
    # return finisher(fdlstpopup, crd)


def b_popupeditgetter(ndx, crd):
    """Constructs B_MiSpecPopUp, Returns Array of New Attributes or newFood that ~replaces~ old food"""
    bmispecpopup = B_MiSpecPopUp()
    bmispecpopup.b_loader(ndx)


# def finisher(fdlstpopup, crd):
#     Clock.schedule_once(lambda p: fdlstpopup.dismiss())
#     dyi = defineifmissingindividuals(crd)
#     with open(dyi, 'r') as dyir:
#         st = dyir.readlines()
#         dyir.close()
#     # newfd = converttofullfood(st[ndx])
#     # return newfd
#     return Food('', '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def usedfoodfinder(crd, duz):
    u = 0
    while u < len(crd.usedfoodsincount):
        if crd.usedfoodsincount[u].food_name == duz.food_name:
            return u
        u += 1
    return -1


def applyindedits(lb, duz):
    crd = doprocess(getimgs())
    xinused = usedfoodfinder(crd, duz)
    staticfood = duz
    # nwfd =
    amtsrvngs = getservingamt(xinused, crd)
    popupeditgetter(xinused, crd)
    # adjustcurfoodlstandtotalsfoodlst(duz, nwfd)
    # lb.text = nwfd.food_name  # Index of .food_name


def b_applyindedits(lb, duz):
    crd = business_doprocess()
    xinused = usedfoodfinder(crd, duz)
    # amtsrvngs = getservingamt(xinused, crd)
    b_popupeditgetter(xinused, crd)


class FoodDeletePopUp(Popup):
    def opener(self, indofbox, fdname):
        self.ids.hinttext.text = fdname
        self.open()
        self.ids.btnforbind.bind(on_release=lambda d: self.confirm_delete(indofbox))
        print("Bind successful!")

    def confirm_delete(self, indofbox):
        global indforfdlstpopup
        indforfdlstpopup = int(indofbox)
        self.dismiss()


class FoodListPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.box = {}
        self.lb = {}
        self.gc = {}
        self.b = {}
        self.b2 = {}

    def scheduleit(self):
        Clock.schedule_interval(self.food_delete, 1.0)

    def descheduleit(self):
        Clock.unschedule(self.food_delete)

    def areyousure(self, indofbox, fdname):
        fdd = FoodDeletePopUp()
        fdd.opener(indofbox, fdname)

    def food_delete(self, *args):
        d = str(datetime.datetime.now())[0:10]
        global indforfdlstpopup
        global universal_list
        global user
        global con
        concur = con.cursor()
        if indforfdlstpopup is not None:
            self.ids.forfoods.remove_widget(self.box[indforfdlstpopup])
            cwd = os.getcwd()
            path = os.path.join(os.path.dirname(__file__), "DietFriend_Pictures"+user)
            os.chdir(path)
            imgfilelst = os.listdir(path)
            try:
                os.remove(imgfilelst[int(indforfdlstpopup)])
            except IndexError:
                pass
            os.chdir(cwd)
            cl = [defineifmissingdt(d), defineifmissingtypedt(d), defineifmissingestrefdt(d),
                  defineifmissingindividualsdt(d), defineifmissingflsession()]
            t = 1
            while t < len(foodattrnamelst):
                strforadd = defineifmissingmisessiontwodt(d, foodattrnamelst[t])
                cl.append(strforadd)
                t += 1
            count = -1
            for u in cl:
                count += 1
                try:
                    with open(u, 'r') as pr:
                        prp = pr.readlines()
                        pr.close()
                    prp.pop(int(indforfdlstpopup))
                    strforwrite = ""
                    for y in prp:
                        strforwrite += y
                    with open(u, 'w') as rp:
                        rp.write(strforwrite)
                        rp.close()
                    listforupdatedb[count][3] = strforwrite
                    listforupdatedb[count][5] = user
                    listforupdatedb[count][7] = str(datetime.datetime.now())[0:10]
                    emit = listforupdatedb[count]
                    if len(emit) > 6:
                        num_params = emit[0]
                        table = emit[1]
                        col_name = emit[2]
                        set_to = emit[3]
                        condition_1_name = emit[4]
                        cond_1 = emit[5]
                        condition_2_name = emit[6]
                        cond_2 = emit[7]
                    else:
                        num_params = emit[0]
                        table = emit[1]
                        col_name = emit[2]
                        set_to = emit[3]
                        condition_1_name = emit[4]
                        cond_1 = emit[5]
                    if num_params == 2:
                        if parameter_based_check_db_row(cursor=concur, table=table, condition_1_name=condition_1_name, cond_1=cond_1, condition_2_name=condition_2_name, cond_2=cond_2):
                            str_to_execute = "UPDATE "+table+" SET "+col_name+" = \'" + set_to + "\' WHERE "+condition_1_name+" = \'" + \
                            cond_1 + "\' AND "+condition_2_name+" = \'" + cond_2 + "\'"
                            print(str_to_execute)
                            concur.execute(str_to_execute)
                            concur.execute("COMMIT")
                        else:
                            query = sql.SQL(
                                "INSERT INTO "+table+" ("+condition_1_name+", "+condition_2_name+") VALUES (\'" + cond_1 + "\', \'" + cond_2 + "\')")
                            print("Query: ")
                            print(query)
                            concur.execute(query)
                            concur.execute("COMMIT")
                    elif num_params == 1:
                        if parameter_based_check_db_row(cursor=concur, table=table, condition_1_name=condition_1_name, cond_1=cond_1):
                            str_to_execute = "UPDATE "+table+" SET "+col_name+" = \'" + set_to + "\' WHERE "+condition_1_name+" = \'" + \
                            cond_1 + "\'"
                            print(str_to_execute)
                            concur.execute(str_to_execute)
                            concur.execute("COMMIT")
                        else:
                            query = sql.SQL("INSERT INTO "+table+" ("+condition_1_name+") VALUES (\'" + cond_1 + "\')")
                            print("Query: ")
                            print(query)
                            concur.execute(query)
                            concur.execute("COMMIT")
                    else:
                        pass
                except IndexError:
                    pass
            templstone = universal_list[0]
            templsttwo = universal_list[1]
            templstthree = universal_list[2]
            templstone.pop(indforfdlstpopup)
            templsttwo.pop(indforfdlstpopup)
            templstthree.pop(indforfdlstpopup)
            universal_list[0] = templstone
            universal_list[1] = templsttwo
            universal_list[2] = templstthree
            indforfdlstpopup = None

    def remover(self):
        p = 0
        while p < len(self.box):
            fdname = self.b2[p].name[0:len(self.b2[p].name)]
            if self.box[p].name[0:len(self.box[p].name)] == fdname:
                self.name.forfoods.remove_widget(self.box[p])
            p += 1

    def onloadfdl(self):
        lst = []
        global indforfdlstpopup
        indforfdlstpopup = None
        newimgsnamesnumservings = getimgs()
        dayfordisplay = doprocess(newimgsnamesnumservings)
        fdfile = defineifmissingflsession()
        print("x")
        with open(fdfile, 'r') as fdf:
            fdl = fdf.readlines()
            fdf.close()
        if len(fdl) == 0:
            with open(fdfile, 'w') as p:
                p.close()
        fd = 0
        print("x")
        while fd < len(dayfordisplay.usedfoodsincount):
            if not isinfd(fdl, fd, dayfordisplay.usedfoodsincount):
                ind = fd
                self.box[ind] = (GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                            padding=[10, 10, 10, 10]))
                self.lb[ind] = (Label(text=dayfordisplay.usedfoodsincount[ind].food_name, size_hint_y=0.1))
                self.gc[ind] = (GridLayout(cols=1, rows=2))
                """Edit Buttons START"""
                if ind == 0:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[0], dayfordisplay.usedfoodsincount[0])))
                elif ind == 1:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[1], dayfordisplay.usedfoodsincount[1])))
                elif ind == 2:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[2], dayfordisplay.usedfoodsincount[2])))
                elif ind == 3:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[3], dayfordisplay.usedfoodsincount[3])))
                elif ind == 4:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[4], dayfordisplay.usedfoodsincount[4])))
                elif ind == 5:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[5], dayfordisplay.usedfoodsincount[5])))
                elif ind == 6:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[6], dayfordisplay.usedfoodsincount[6])))
                elif ind == 7:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[7], dayfordisplay.usedfoodsincount[7])))
                elif ind == 8:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[8], dayfordisplay.usedfoodsincount[8])))
                elif ind == 9:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[9], dayfordisplay.usedfoodsincount[9])))
                elif ind == 10:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[10], dayfordisplay.usedfoodsincount[10])))
                elif ind == 11:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[11], dayfordisplay.usedfoodsincount[11])))
                elif ind == 12:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[12], dayfordisplay.usedfoodsincount[12])))
                elif ind == 13:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[13], dayfordisplay.usedfoodsincount[13])))
                elif ind == 14:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[14], dayfordisplay.usedfoodsincount[14])))
                elif ind == 15:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[15], dayfordisplay.usedfoodsincount[15])))
                elif ind == 16:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[16], dayfordisplay.usedfoodsincount[16])))
                elif ind == 17:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[17], dayfordisplay.usedfoodsincount[17])))
                elif ind == 18:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[18], dayfordisplay.usedfoodsincount[18])))
                elif ind == 19:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[19], dayfordisplay.usedfoodsincount[19])))
                elif ind == 20:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[20], dayfordisplay.usedfoodsincount[20])))
                elif ind == 21:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[21], dayfordisplay.usedfoodsincount[21])))
                elif ind == 22:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[22], dayfordisplay.usedfoodsincount[22])))
                elif ind == 23:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[23], dayfordisplay.usedfoodsincount[23])))
                elif ind == 24:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[24], dayfordisplay.usedfoodsincount[24])))
                elif ind == 25:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[25], dayfordisplay.usedfoodsincount[25])))
                elif ind == 26:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[26], dayfordisplay.usedfoodsincount[26])))
                elif ind == 27:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[27], dayfordisplay.usedfoodsincount[27])))
                elif ind == 28:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[28], dayfordisplay.usedfoodsincount[28])))
                elif ind == 29:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[29], dayfordisplay.usedfoodsincount[29])))
                elif ind == 30:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[30], dayfordisplay.usedfoodsincount[30])))
                elif ind == 31:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[31], dayfordisplay.usedfoodsincount[31])))
                elif ind == 32:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[32], dayfordisplay.usedfoodsincount[32])))
                elif ind == 33:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[33], dayfordisplay.usedfoodsincount[33])))
                elif ind == 34:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[34], dayfordisplay.usedfoodsincount[34])))
                elif ind == 35:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[35], dayfordisplay.usedfoodsincount[35])))
                elif ind == 36:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[36], dayfordisplay.usedfoodsincount[36])))
                elif ind == 37:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[37], dayfordisplay.usedfoodsincount[37])))
                elif ind == 38:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[38], dayfordisplay.usedfoodsincount[38])))
                elif ind == 39:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[39], dayfordisplay.usedfoodsincount[39])))
                elif ind == 40:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[40], dayfordisplay.usedfoodsincount[40])))
                elif ind == 41:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[41], dayfordisplay.usedfoodsincount[41])))
                elif ind == 42:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[42], dayfordisplay.usedfoodsincount[42])))
                elif ind == 43:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[43], dayfordisplay.usedfoodsincount[43])))
                elif ind == 44:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[44], dayfordisplay.usedfoodsincount[44])))
                elif ind == 45:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[45], dayfordisplay.usedfoodsincount[45])))
                elif ind == 46:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[46], dayfordisplay.usedfoodsincount[46])))
                elif ind == 47:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[47], dayfordisplay.usedfoodsincount[47])))
                elif ind == 48:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[48], dayfordisplay.usedfoodsincount[48])))
                elif ind == 49:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[49], dayfordisplay.usedfoodsincount[49])))
                else:
                    self.b[ind] = (Button(text="Edit", size_hint_y=0.1,
                                          on_press=lambda x:
                                          applyindedits(self.lb[50], dayfordisplay.usedfoodsincount[50])))
                """Edit Buttons END"""
                """Delete Buttons START"""
                if ind == 0:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(0, self.lb[0].text)))
                elif ind == 1:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(1, self.lb[1].text)))
                elif ind == 2:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(2, self.lb[2].text)))
                elif ind == 3:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(3, self.lb[3].text)))
                elif ind == 4:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(4, self.lb[4].text)))
                elif ind == 5:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(5, self.lb[5].text)))
                elif ind == 6:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(6, self.lb[6].text)))
                elif ind == 7:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(7, self.lb[7].text)))
                elif ind == 8:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(8, self.lb[8].text)))
                elif ind == 9:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(9, self.lb[9].text)))
                elif ind == 10:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(10, self.lb[10].text)))
                elif ind == 11:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(11, self.lb[11].text)))
                elif ind == 12:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(12, self.lb[12].text)))
                elif ind == 13:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(13, self.lb[13].text)))
                elif ind == 14:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(14, self.lb[14].text)))
                elif ind == 15:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(15, self.lb[15].text)))
                elif ind == 16:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(16, self.lb[16].text)))
                elif ind == 17:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(17, self.lb[17].text)))
                elif ind == 18:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(18, self.lb[18].text)))
                elif ind == 19:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(19, self.lb[19].text)))
                elif ind == 20:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(20, self.lb[20].text)))
                elif ind == 21:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(21, self.lb[21].text)))
                elif ind == 22:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(22, self.lb[22].text)))
                elif ind == 23:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(23, self.lb[23].text)))
                elif ind == 24:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(24, self.lb[24].text)))
                elif ind == 25:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(25, self.lb[25].text)))
                elif ind == 26:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(26, self.lb[26].text)))
                elif ind == 27:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(27, self.lb[27].text)))
                elif ind == 28:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(28, self.lb[28].text)))
                elif ind == 29:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(29, self.lb[29].text)))
                elif ind == 30:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(30, self.lb[30].text)))
                elif ind == 31:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(31, self.lb[31].text)))
                elif ind == 32:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(32, self.lb[32].text)))
                elif ind == 33:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(33, self.lb[33].text)))
                elif ind == 34:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(34, self.lb[34].text)))
                elif ind == 35:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(35, self.lb[35].text)))
                elif ind == 36:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(36, self.lb[36].text)))
                elif ind == 37:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(37, self.lb[37].text)))
                elif ind == 38:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(38, self.lb[38].text)))
                elif ind == 39:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(39, self.lb[39].text)))
                elif ind == 40:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(40, self.lb[40].text)))
                elif ind == 41:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(41, self.lb[41].text)))
                elif ind == 42:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(42, self.lb[42].text)))
                elif ind == 43:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(43, self.lb[43].text)))
                elif ind == 44:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(44, self.lb[44].text)))
                elif ind == 45:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(45, self.lb[45].text)))
                elif ind == 46:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(46, self.lb[46].text)))
                elif ind == 47:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(47, self.lb[47].text)))
                elif ind == 48:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(48, self.lb[48].text)))
                elif ind == 49:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(49, self.lb[49].text)))
                else:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.areyousure(50, self.lb[50].text)))
                """Delete Buttons END"""
                self.gc[ind].add_widget(self.b[ind])
                self.gc[ind].add_widget(self.b2[ind])
                self.box[ind].add_widget(self.lb[ind])
                self.box[ind].add_widget(self.gc[ind])
                self.ids.forfoods.add_widget(self.box[ind])
                lst.append([self.box[ind], self.lb[ind], self.gc[ind], self.b[ind], self.b2[ind]])
                # print(lst[ind][0])
                print(self.ids.forfoods.children)
                addfdnametotext(dayfordisplay.usedfoodsincount[ind].food_name, fdfile)
            fd += 1
        print(lst)
        self.scheduleit()


""" CURRENT:
on_press=lambda x: self.ids.forfoods.remove_widget(self.box[ind]),
OBSOLETE:
on_press=lambda x: self.ids.forfoods.remove_widget(self.boxcont)) works,
on_press=lambda x: self.ids.forfoods.remove_widget(lst[fd-1][0]) works,
on_press=lambda x: self.ids.forfoods.remove_widget(self.ids.forfoods.children[0]) works"""
"""
    def onloadfdl(self):
        lst = []
        newimgsnamesnumservings = getimgs()
        dayfordisplay = doprocess(newimgsnamesnumservings)
        fdfile = defineifmissingflsession()
        print("x")
        with open(fdfile, 'r') as fdf:
            fdl = fdf.readlines()
            fdf.close()
        if len(fdl) == 0:
            with open(fdfile, 'w') as p:
                p.close()
        fd = 0
        print("x")
        while fd < len(dayfordisplay.usedfoodsincount):
            if not isinfd(fdl, fd, dayfordisplay.usedfoodsincount):
                self.boxcont = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y), padding=[10, 10, 10, 10])
                self.label = Label(text=dayfordisplay.usedfoodsincount[fd].food_name, size_hint_y=0.1)
                self.gridcont = GridLayout(cols=1, rows=2)
                self.button = Button(text="Edit", size_hint_y=0.1)
                self.button2 = Button(
                text="Delete", size_hint_y=0.1, on_press=lambda x: self.ids.forfoods.remove_widget(lst[fd][0]))
                self.gridcont.add_widget(self.button)
                self.gridcont.add_widget(self.button2)
                self.boxcont.add_widget(self.label)
                self.boxcont.add_widget(self.gridcont)
                self.ids.forfoods.add_widget(self.boxcont)
                lst.append([self.boxcont, self.label, self.gridcont, self.button, self.button2])
                print(lst[fd][0])
                print(self.ids.forfoods.children)
                addfdnametotext(dayfordisplay.usedfoodsincount[fd].food_name, fdfile)
            fd += 1
        print(lst)

on_press=lambda x: self.ids.forfoods.remove_widget(self.boxcont)) works"""


# I JUST DON'T KNOW
# def proxyonloadfdl(self, fdf, fdfile):
#     lst = []
#     with open(fdfile, 'r') as fdp:
#         fdl = fdp.readlines()
#         fdp.close()
#     if len(fdl) == 0:
#         with open(fdfile, 'w') as p:
#             p.close()
#     fd = 0
#     print("x")
#     """, ids=str(fd)"""
#     while fd < len(fdf.usedfoodsincount):
#         if not isinfd(fdl, fd, fdf.usedfoodsincount):
#             ind = fd
#             self.box.append(GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y), padding=[10, 10, 10, 10]))
#             self.lb.append(Label(text=fdf.usedfoodsincount[ind].food_name, size_hint_y=0.1))
#             self.gc.append(GridLayout(cols=1, rows=2))
#             self.b.append(Button(text="Edit", size_hint_y=0.1))
#             self.b2.append(Button(text="Delete", size_hint_y=0.1,
#                                   on_release=self.removeandreplace(self.ids, fdf, fdfile), ids=str(ind)))
#             self.gc[ind].add_widget(self.b[ind])
#             self.gc[ind].add_widget(self.b2[ind])
#             self.box[ind].add_widget(self.lb[ind])
#             self.box[ind].add_widget(self.gc[ind])
#             self.ids.forfoods.add_widget(self.box[ind])
#             lst.append([self.box[ind], self.lb[ind], self.gc[ind], self.b[ind], self.b2[ind]])
#             print(lst[ind][0])
#             print(self.ids.forfoods.children)
#             addfdnametotext(fdf.usedfoodsincount[ind].food_name, fdfile)
#         fd += 1
#     print(lst)
#
#
# def removeandreplace(self, ida, dfd, fdfil):
#     self.ids.forfoods.clear_widgets(children=None)
#     num = int(ida)
#     print(num)
#     dfd.usedfoodsincount.pop(num)
#     removefromtext(num, fdfil)
#     self.proxyonloadfdl(dfd, fdfil)
#
#
# def clk(self, stringid, dayfordisplay, fdfile):
#     nnum = int(stringid)
#     self.b2[nnum].bind(on_press=lambda x: self.removeandreplace(stringid, dayfordisplay, fdfile))
#
#
# def onloadfdl(self):
#     lst = []
#     newimgsnamesnumservings = getimgs()
#     dayfordisplay = doprocess(newimgsnamesnumservings)
#     fdfile = defineifmissingflsession()
#     print("x")
#     with open(fdfile, 'r') as fdf:
#         fdl = fdf.readlines()
#         fdf.close()
#     if len(fdl) == 0:
#         with open(fdfile, 'w') as p:
#             p.close()
#     fd = 0
#     print("x")
#     """, ids=str(fd)"""
#     while fd < len(dayfordisplay.usedfoodsincount):
#         if not isinfd(fdl, fd, dayfordisplay.usedfoodsincount):
#             ind = fd
#             self.box.append(GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y), padding=[10, 10, 10, 10]))
#             self.lb.append(Label(text=dayfordisplay.usedfoodsincount[ind].food_name, size_hint_y=0.1))
#             self.gc.append(GridLayout(cols=1, rows=2))
#             self.b.append(Button(text="Edit", size_hint_y=0.1))
#             stringid = str(ind)
#             self.b2.append(Button(text="Delete", size_hint_y=0.1,
#                                   on_press=self.clk(stringid, dayfordisplay, fdfile), ids=stringid))
#             self.gc[ind].add_widget(self.b[ind])
#             self.gc[ind].add_widget(self.b2[ind])
#             self.box[ind].add_widget(self.lb[ind])
#             self.box[ind].add_widget(self.gc[ind])
#             self.ids.forfoods.add_widget(self.box[ind])
#             lst.append([self.box[ind], self.lb[ind], self.gc[ind], self.b[ind], self.b2[ind]])
#             print(lst[ind][0])
#             print(self.ids.forfoods.children)
#             addfdnametotext(dayfordisplay.usedfoodsincount[ind].food_name, fdfile)
#         fd += 1
#     print(lst)


class MIFoodEditor(Popup):
    def opener(self, attribute, index, current_day):
        self.open()
        self.ids.for__food_name.text = current_day.usedfoodsincount[index].food_name
        self.ids.to_edit.text = str(getattr(current_day.usedfoodsincount[index], attribute))
        self.ids.forbnd.bind(on_release=lambda f: self.setvals(attribute, index))

    def setvals(self, attribute, index):
        er = defineifmissingmisessionpopup()
        with open(er, 'w') as oer:
            oer.write(attribute + "\n" + str(index) + "\n" + self.ids.to_edit.text.replace('\n', '') + "\n")
            oer.close()
        self.dismiss()


def not_in(x, p):
    for i in p:
        if i == x:
            return False
    return True


# def empty(a):
#     i = 0
#     for b in a:
#         i += 1
#     if i != 0:
#         return False
#     return True


class MoreInfo(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bx = {}
        self.gci = {}
        self.label = {}
        self.button = {}
        self.button2 = {}
        self.button3 = {}

        # TO-DO: Edit all i,% on im_two***** such that % changes to estimated or reverted on those respective buttons,
        # possibly change x,% to i,altered-% as well

    def scheduler(self):
        Clock.schedule_interval(self.checkermip, 0.5)

    def descheduler(self):
        Clock.unschedule(self.checkermip)

    def retrieve(self):
        hr = defineifmissingmisessionpopup()
        with open(hr, 'r') as q:
            qq = q.readlines()
            q.close()
        attribute = qq[0].strip()
        index = int(qq[1].strip())
        value = float(qq[2].strip())
        with open(hr, 'w') as q:
            q.write("")
            q.close()
        c = getimgs()
        current_day = doprocess(c)
        xdxf = defineifmissingmisessiontwo(current_day, attribute)
        with open(xdxf, 'r') as lc:
            new = lc.readlines()[index].replace('x', 'i')
            lc.close()
        new = new[0:new.find(',') + 1] + str(value) + "\n"
        writetolinemix(current_day, attribute, index + 1, new)
        fd = Food(getattr(current_day.usedfoodsincount[index], 'food_name'),
                  getattr(current_day.usedfoodsincount[index], 'food_datetime'),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[0]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[1]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[2]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[3]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[4]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[5]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[6]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[7]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[8]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[9]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[10]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[11]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[12]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[13]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[14]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[15]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[16]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[17]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[18]))
        setattr(fd, attribute, value)
        if str(value).find('-2000') == -1:
            self.label[index].text = current_day.usedfoodsincount[index].food_name + ": " + str(
                float(value) * universal_list[2][index])
        else:
            self.label[index].text = current_day.usedfoodsincount[index].food_name + ": " + str(value)
        tempfd = current_day.usedfoodsincount[index]
        current_day.usedfoodsincount[index] = fd
        fileindivid = defineifmissingindividuals(current_day)
        strtowrite = ""
        i = 0
        while i < 21:
            if i >= 2:
                ii = i - 2
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], foodattrnamelst[ii])) + ' ')
            elif i == 1:
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], 'food_datetime')) + ' ')
            else:
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], 'food_name')) + ' ')
            i += 1
        strtowrite += '\n'
        writetolineindividuals(current_day, index + 1, '\n')
        writetolineindividuals(current_day, index + 1, strtowrite)
        foodsubtractpreadd(current_day, tempfd, fd)
        dst = defineifmissingtype(current_day)
        with open(dst, 'r') as dstr:
            f = dstr.readlines()
            dstr.close()
        strforwrite = f[index]
        strforwrite = strforwrite.replace('t', 'r')
        strforwrite = strforwrite.replace('e', 'r')
        writetoline(current_day, index + 1, '\n')
        writetoline(current_day, index + 1, strforwrite)
        if estimatefromcustomchecker(index, fileindivid, current_day):
            with open(dst, 'r') as dstr:
                f = dstr.readlines()
                dstr.close()
            strforwrite = f[index]
            strforwrite = strforwrite.replace('r', 'e')
            writetoline(current_day, index + 1, '\n')
            writetoline(current_day, index + 1, strforwrite)

        global user
        global con
        datestring = current_day.date_time
        datestring = str(datestring)
        datestring = datestring[0:10]
        curcursor = con.cursor()
        indtext = defineifmissingindividuals(current_day)
        with open(indtext, 'r') as readind:
            indtextlist = readind.readlines()
            print("indtextlist: ")
            readind.close()
        indtxt = ""
        u = 0
        while u < len(indtextlist):
            indtxt += indtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET individualstextfile_r = \'" + indtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        typtext = defineifmissingtype(current_day)
        with open(typtext, 'r') as readtyp:
            typtextlist = readtyp.readlines()
            print("typtextlist: ")
            readtyp.close()
        typtxt = ""
        u = 0
        while u < len(typtextlist):
            typtxt += typtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        strtoselect = "_im_two" + attribute
        with open(xdxf, 'r') as mft:
            vv = mft.readlines()
            print(vv)
            if vv[len(vv) - 1].find('i') == -1 and vv[len(vv) - 1].find('x') == -1:
                vv.pop(len(vv) - 1)
            print(vv)
            mft.close()
        text = ""
        for h in vv:
            text += h
        str_to_execute = "UPDATE client_moreinfo_value_storage SET " + strtoselect + " = \'" + text.replace('None',
                                                                                                            '').replace(
            '\\n', '\n') + "\' WHERE username = \'" + user + "\' AND datetime = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")
        with open(xdxf, 'w') as mftw:
            mftw.write(text)
            mftw.close()

    def checkermip(self, *args):
        hr = defineifmissingmisessionpopup()
        with open(hr, 'r') as q:
            qq = q.readlines()
            q.close()
        try:
            if len(qq) >= 2:
                self.retrieve()
        except:
            pass

    def edit(self, attribute, index, current_day):
        popup = MIFoodEditor()
        popup.opener(attribute, index, current_day)

    def exclude(self, attribute, index, current_day):
        self.label[index].text = current_day.usedfoodsincount[index].food_name + ": [excluded]"
        xdxf = defineifmissingmisessiontwo(current_day, attribute)
        with open(xdxf, 'r') as lc:
            new = lc.readlines()[index].replace('i', 'x')
            lc.close()
        writetolinemix(current_day, attribute, index + 1, new)
        fd = Food(getattr(current_day.usedfoodsincount[index], 'food_name'),
                  getattr(current_day.usedfoodsincount[index], 'food_datetime'),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[0]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[1]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[2]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[3]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[4]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[5]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[6]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[7]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[8]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[9]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[10]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[11]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[12]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[13]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[14]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[15]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[16]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[17]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[18]))
        setattr(fd, attribute, -2000)
        tempfd = current_day.usedfoodsincount[index]
        current_day.usedfoodsincount[index] = fd
        fileindivid = defineifmissingindividuals(current_day)
        strtowrite = ""
        i = 0
        while i < 21:
            if i >= 2:
                ii = i - 2
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], foodattrnamelst[ii])) + ' ')
            elif i == 1:
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], 'food_datetime')) + ' ')
            else:
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], 'food_name')) + ' ')
            i += 1
        strtowrite += '\n'
        writetolineindividuals(current_day, index + 1, '\n')
        writetolineindividuals(current_day, index + 1, strtowrite)
        foodsubtractpreadd(current_day, tempfd, fd)
        dst = defineifmissingtype(current_day)
        with open(dst, 'r') as dstr:
            f = dstr.readlines()
            dstr.close()
        strforwrite = f[index]
        strforwrite = strforwrite.replace('t', 'r')
        strforwrite = strforwrite.replace('e', 'r')
        writetoline(current_day, index + 1, '\n')
        writetoline(current_day, index + 1, strforwrite)
        if estimatefromcustomchecker(index, fileindivid, current_day):
            with open(dst, 'r') as dstr:
                f = dstr.readlines()
                dstr.close()
            strforwrite = f[index]
            strforwrite = strforwrite.replace('r', 'e')
            writetoline(current_day, index + 1, '\n')
            writetoline(current_day, index + 1, strforwrite)

        global user
        global con
        datestring = current_day.date_time
        datestring = str(datestring)
        datestring = datestring[0:10]
        curcursor = con.cursor()
        indtext = defineifmissingindividuals(current_day)
        with open(indtext, 'r') as readind:
            indtextlist = readind.readlines()
            print("indtextlist: ")
            readind.close()
        indtxt = ""
        u = 0
        while u < len(indtextlist):
            indtxt += indtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET individualstextfile_r = \'" + indtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        typtext = defineifmissingtype(current_day)
        with open(typtext, 'r') as readtyp:
            typtextlist = readtyp.readlines()
            print("typtextlist: ")
            readtyp.close()
        typtxt = ""
        u = 0
        while u < len(typtextlist):
            typtxt += typtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        strtoselect = "_im_two" + attribute
        with open(xdxf, 'r') as mft:
            vv = mft.readlines()
            print(vv)
            if vv[len(vv) - 1].find('i') == -1 and vv[len(vv) - 1].find('x') == -1:
                vv.pop(len(vv) - 1)
            print(vv)
            mft.close()
        text = ""
        for h in vv:
            text += h
        str_to_execute = "UPDATE client_moreinfo_value_storage SET " + strtoselect + " = \'" + text.replace('None',
                                                                                                            '').replace(
            '\\n', '\n') + "\' WHERE username = \'" + \
                         user + "\' AND datetime = \'" + str(datetime.datetime.now())[0:10] + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")
        with open(xdxf, 'w') as mftw:
            mftw.write(text)
            mftw.close()

    def include(self, attribute, index, current_day):
        xdxf = defineifmissingmisessiontwo(current_day, attribute)
        with open(xdxf, 'r') as lc:
            orig = lc.readlines()[index]
            lc.close()
        orig = orig[2:len(orig)].strip()
        writetolinemix(current_day, attribute, index + 1, "i," + orig + "\n")
        if str(float(orig)).find('-2000') == -1:
            self.label[index].text = current_day.usedfoodsincount[index].food_name + ": " + str(
                float(orig) * universal_list[2][index])
        else:
            self.label[index].text = current_day.usedfoodsincount[index].food_name + ": " + str(float(orig))
        fd = Food(getattr(current_day.usedfoodsincount[index], 'food_name'),
                  getattr(current_day.usedfoodsincount[index], 'food_datetime'),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[0]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[1]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[2]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[3]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[4]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[5]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[6]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[7]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[8]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[9]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[10]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[11]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[12]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[13]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[14]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[15]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[16]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[17]),
                  getattr(current_day.usedfoodsincount[index], foodattrnamelst[18]))
        setattr(fd, attribute, float(orig))
        tempfd = current_day.usedfoodsincount[index]
        current_day.usedfoodsincount[index] = fd
        fileindivid = defineifmissingindividuals(current_day)
        strtowrite = ""
        i = 0
        while i < 21:
            if i >= 2:
                ii = i - 2
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], foodattrnamelst[ii])) + ' ')
            elif i == 1:
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], 'food_datetime')) + ' ')
            else:
                strtowrite += ('r,' + str(getattr(current_day.usedfoodsincount[index], 'food_name')) + ' ')
            i += 1
        strtowrite += '\n'
        writetolineindividuals(current_day, index + 1, '\n')
        writetolineindividuals(current_day, index + 1, strtowrite)
        foodsubtractpreadd(current_day, tempfd, fd)
        dst = defineifmissingtype(current_day)
        with open(dst, 'r') as dstr:
            f = dstr.readlines()
            dstr.close()
        strforwrite = f[index]
        strforwrite = strforwrite.replace('t', 'r')
        strforwrite = strforwrite.replace('e', 'r')
        writetoline(current_day, index + 1, '\n')
        writetoline(current_day, index + 1, strforwrite)
        if estimatefromcustomchecker(index, fileindivid, current_day):
            with open(dst, 'r') as dstr:
                f = dstr.readlines()
                dstr.close()
            strforwrite = f[index]
            strforwrite = strforwrite.replace('r', 'e')
            writetoline(current_day, index + 1, '\n')
            writetoline(current_day, index + 1, strforwrite)

        global user
        global con
        datestring = current_day.date_time
        datestring = str(datestring)
        datestring = datestring[0:10]
        curcursor = con.cursor()
        indtext = defineifmissingindividuals(current_day)
        with open(indtext, 'r') as readind:
            indtextlist = readind.readlines()
            print("indtextlist: ")
            readind.close()
        indtxt = ""
        u = 0
        while u < len(indtextlist):
            indtxt += indtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET individualstextfile_r = \'" + indtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        typtext = defineifmissingtype(current_day)
        with open(typtext, 'r') as readtyp:
            typtextlist = readtyp.readlines()
            print("typtextlist: ")
            readtyp.close()
        typtxt = ""
        u = 0
        while u < len(typtextlist):
            typtxt += typtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_client_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
            user + "\' AND date = \'" + datestring + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        strtoselect = "_im_two" + attribute
        with open(xdxf, 'r') as mft:
            vv = mft.readlines()
            print(vv)
            if vv[len(vv) - 1].find('i') == -1 and vv[len(vv) - 1].find('x') == -1:
                vv.pop(len(vv) - 1)
            print(vv)
            mft.close()
        text = ""
        for h in vv:
            text += h
        str_to_execute = "UPDATE client_moreinfo_value_storage SET " + strtoselect + " = \'" + text.replace('None',
                                                                                                            '').replace(
            '\\n', '\n') + "\' WHERE username = \'" + \
                         user + "\' AND datetime = \'" + str(datetime.datetime.now())[0:10] + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")
        with open(xdxf, 'w') as mftw:
            mftw.write(text)
            mftw.close()

    def onloadmi(self):
        global con
        global user
        concur = con.cursor()
        self.scheduler()
        filestr = str(running_id) + user + '_mitype.txt'
        with open(filestr, 'r') as rop:
            try:
                igd = rop.readlines()[0].strip()
            except IndexError:
                igd = 'calories'
            rop.close()
        newimgsnamesnumservings = getimgs()
        dayfordisplay = doprocess(newimgsnamesnumservings)
        for i in dayfordisplay.usedfoodsincount:
            print("Look:")
            print(i.saturated_fat)
        mifile = defineifmissingmisession(igd)
        mifiletwo = defineifmissingmisessiontwo(dayfordisplay, igd)
        print("x")
        with open(mifile, 'r') as mifdf:
            fdl = mifdf.readlines()
            mifdf.close()
        if len(fdl) == 0:
            with open(mifile, 'w') as mip:
                mip.close()
        strtoselect = '_im_two' + igd
        if row_exists_moreinfo(concur, user, str(datetime.datetime.now())[0:10]):
            query = sql.SQL(
                "SELECT " + strtoselect + " FROM client_moreinfo_value_storage WHERE (username = \'" + user + "\') AND (datetime = \'" + str(
                    datetime.datetime.now())[0:10] + "\')")
            print("Query: ")
            print(query)
            concur.execute(query)
            towrt = str(concur.fetchall()).replace('\'', '').replace('None', '').replace('[', '').replace('(',
                                                                                                          '').replace(
                ')', '').replace(']', '').replace('\\n', '\n')
            print("TOWRT")
            print(towrt)
            if towrt.find('x') == -1 and towrt.find('i') == -1:
                pass
            else:
                with open(mifiletwo, 'w') as mfltwo:
                    mfltwo.write(towrt)
                    mfltwo.close()
        else:
            query = sql.SQL(
                "INSERT INTO client_moreinfo_value_storage (username, datetime) VALUES (\'" + user + "\', \'" + str(
                    datetime.datetime.now())[0:10] + "\')")
            print("Query: ")
            print(query)
            concur.execute(query)
            concur.execute("COMMIT")
        with open(mifiletwo, 'r') as mfltwo:
            u = mfltwo.readlines()
            mfltwo.close()
        excluded = []
        k = 0
        while k < len(u):
            if u[k][0:1] == 'x':
                excluded.append(k)
            k += 1
        fd = 0
        print("x")
        while fd < len(dayfordisplay.usedfoodsincount):
            print("getattr")
            print(getattr(dayfordisplay.usedfoodsincount[fd], igd))
            if not isinfd(fdl, fd, dayfordisplay.usedfoodsincount):
                ind = fd
                if str(getattr(dayfordisplay.usedfoodsincount[fd], igd)).find('-2000') == -1:
                    forstr = dayfordisplay.usedfoodsincount[fd].food_name + ': ' + \
                             str(getattr(dayfordisplay.usedfoodsincount[fd], igd) * universal_list[2][fd])
                else:
                    forstr = dayfordisplay.usedfoodsincount[fd].food_name + ': ' + \
                             str(getattr(dayfordisplay.usedfoodsincount[fd], igd))
                forstrx = dayfordisplay.usedfoodsincount[fd].food_name + ': [excluded]'

                if ind == 0:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = g.readlines()
                            g.close()
                        if tttg == []:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[0], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = g.readlines()
                            g.close()
                        if tttg == []:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[0], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 0, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 0, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 0, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[0])
                    self.gci[ind].add_widget(self.button[0])
                    self.gci[ind].add_widget(self.button2[0])
                    self.gci[ind].add_widget(self.button3[0])
                    self.bx[ind].add_widget(self.gci[0])
                    self.ids.formoreinfo.add_widget(self.bx[0])
                elif ind == 1:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 2:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[1], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 2:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[1], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 1, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 1, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 1, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[1])
                    self.gci[ind].add_widget(self.button[1])
                    self.gci[ind].add_widget(self.button2[1])
                    self.gci[ind].add_widget(self.button3[1])
                    self.bx[ind].add_widget(self.gci[1])
                    self.ids.formoreinfo.add_widget(self.bx[1])
                elif ind == 2:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 3:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[2], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 3:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[2], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 2, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 2, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 2, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[2])
                    self.gci[ind].add_widget(self.button[2])
                    self.gci[ind].add_widget(self.button2[2])
                    self.gci[ind].add_widget(self.button3[2])
                    self.bx[ind].add_widget(self.gci[2])
                    self.ids.formoreinfo.add_widget(self.bx[2])
                elif ind == 3:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 4:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[3], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 4:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[3], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 3, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 3, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 3, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[3])
                    self.gci[ind].add_widget(self.button[3])
                    self.gci[ind].add_widget(self.button2[3])
                    self.gci[ind].add_widget(self.button3[3])
                    self.bx[ind].add_widget(self.gci[3])
                    self.ids.formoreinfo.add_widget(self.bx[3])
                elif ind == 4:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 5:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[4], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 5:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[4], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 4, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 4, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 4, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[4])
                    self.gci[ind].add_widget(self.button[4])
                    self.gci[ind].add_widget(self.button2[4])
                    self.gci[ind].add_widget(self.button3[4])
                    self.bx[ind].add_widget(self.gci[4])
                    self.ids.formoreinfo.add_widget(self.bx[4])
                elif ind == 5:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 6:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[5], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 6:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[5], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 5, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 5, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 5, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[5])
                    self.gci[ind].add_widget(self.button[5])
                    self.gci[ind].add_widget(self.button2[5])
                    self.gci[ind].add_widget(self.button3[5])
                    self.bx[ind].add_widget(self.gci[5])
                    self.ids.formoreinfo.add_widget(self.bx[5])
                elif ind == 6:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 7:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[6], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 7:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[6], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 6, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 6, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 6, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[6])
                    self.gci[ind].add_widget(self.button[6])
                    self.gci[ind].add_widget(self.button2[6])
                    self.gci[ind].add_widget(self.button3[6])
                    self.bx[ind].add_widget(self.gci[6])
                    self.ids.formoreinfo.add_widget(self.bx[6])
                elif ind == 7:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 8:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[7], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 8:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[7], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 7, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 7, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 7, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[7])
                    self.gci[ind].add_widget(self.button[7])
                    self.gci[ind].add_widget(self.button2[7])
                    self.gci[ind].add_widget(self.button3[7])
                    self.bx[ind].add_widget(self.gci[7])
                    self.ids.formoreinfo.add_widget(self.bx[7])
                elif ind == 8:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 9:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[8], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 9:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[8], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 8, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 8, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 8, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[8])
                    self.gci[ind].add_widget(self.button[8])
                    self.gci[ind].add_widget(self.button2[8])
                    self.gci[ind].add_widget(self.button3[8])
                    self.bx[ind].add_widget(self.gci[8])
                    self.ids.formoreinfo.add_widget(self.bx[8])
                elif ind == 9:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 10:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[9], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 10:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[9], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 9, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 9, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 9, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[9])
                    self.gci[ind].add_widget(self.button[9])
                    self.gci[ind].add_widget(self.button2[9])
                    self.gci[ind].add_widget(self.button3[9])
                    self.bx[ind].add_widget(self.gci[9])
                    self.ids.formoreinfo.add_widget(self.bx[9])
                elif ind == 10:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 11:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[10], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 11:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[10], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 10, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 10, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 10, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[10])
                    self.gci[ind].add_widget(self.button[10])
                    self.gci[ind].add_widget(self.button2[10])
                    self.gci[ind].add_widget(self.button3[10])
                    self.bx[ind].add_widget(self.gci[10])
                    self.ids.formoreinfo.add_widget(self.bx[10])
                elif ind == 11:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 12:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[11], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 12:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[11], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 11, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 11, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 11, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[11])
                    self.gci[ind].add_widget(self.button[11])
                    self.gci[ind].add_widget(self.button2[11])
                    self.gci[ind].add_widget(self.button3[11])
                    self.bx[ind].add_widget(self.gci[11])
                    self.ids.formoreinfo.add_widget(self.bx[11])
                elif ind == 12:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 13:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[12], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 13:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[12], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 12, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 12, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 12, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[12])
                    self.gci[ind].add_widget(self.button[12])
                    self.gci[ind].add_widget(self.button2[12])
                    self.gci[ind].add_widget(self.button3[12])
                    self.bx[ind].add_widget(self.gci[12])
                    self.ids.formoreinfo.add_widget(self.bx[12])
                elif ind == 13:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 14:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[13], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 14:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[13], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 13, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 13, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 13, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[13])
                    self.gci[ind].add_widget(self.button[13])
                    self.gci[ind].add_widget(self.button2[13])
                    self.gci[ind].add_widget(self.button3[13])
                    self.bx[ind].add_widget(self.gci[13])
                    self.ids.formoreinfo.add_widget(self.bx[13])
                elif ind == 14:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 15:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[14], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 15:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[14], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 14, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 14, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 14, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[14])
                    self.gci[ind].add_widget(self.button[14])
                    self.gci[ind].add_widget(self.button2[14])
                    self.gci[ind].add_widget(self.button3[14])
                    self.bx[ind].add_widget(self.gci[14])
                    self.ids.formoreinfo.add_widget(self.bx[14])
                elif ind == 15:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 16:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[15], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 16:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[15], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 15, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 15, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 15, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[15])
                    self.gci[ind].add_widget(self.button[15])
                    self.gci[ind].add_widget(self.button2[15])
                    self.gci[ind].add_widget(self.button3[15])
                    self.bx[ind].add_widget(self.gci[15])
                    self.ids.formoreinfo.add_widget(self.bx[15])
                elif ind == 16:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 17:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[16], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 17:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[16], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 16, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 16, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 16, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[16])
                    self.gci[ind].add_widget(self.button[16])
                    self.gci[ind].add_widget(self.button2[16])
                    self.gci[ind].add_widget(self.button3[16])
                    self.bx[ind].add_widget(self.gci[16])
                    self.ids.formoreinfo.add_widget(self.bx[16])
                elif ind == 17:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 18:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[17], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 18:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[17], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 17, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 17, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 17, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[17])
                    self.gci[ind].add_widget(self.button[17])
                    self.gci[ind].add_widget(self.button2[17])
                    self.gci[ind].add_widget(self.button3[17])
                    self.bx[ind].add_widget(self.gci[17])
                    self.ids.formoreinfo.add_widget(self.bx[17])
                elif ind == 18:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 19:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[18], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 19:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[18], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 18, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 18, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 18, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[18])
                    self.gci[ind].add_widget(self.button[18])
                    self.gci[ind].add_widget(self.button2[18])
                    self.gci[ind].add_widget(self.button3[18])
                    self.bx[ind].add_widget(self.gci[18])
                    self.ids.formoreinfo.add_widget(self.bx[18])
                elif ind == 19:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 20:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[19], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 20:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[19], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 19, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 19, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 19, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[19])
                    self.gci[ind].add_widget(self.button[19])
                    self.gci[ind].add_widget(self.button2[19])
                    self.gci[ind].add_widget(self.button3[19])
                    self.bx[ind].add_widget(self.gci[19])
                    self.ids.formoreinfo.add_widget(self.bx[19])
                elif ind == 20:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 21:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[20], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 21:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[20], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 20, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 20, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 20, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[20])
                    self.gci[ind].add_widget(self.button[20])
                    self.gci[ind].add_widget(self.button2[20])
                    self.gci[ind].add_widget(self.button3[20])
                    self.bx[ind].add_widget(self.gci[20])
                    self.ids.formoreinfo.add_widget(self.bx[20])
                elif ind == 21:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 22:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[21], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 22:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[21], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 21, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 21, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 21, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[21])
                    self.gci[ind].add_widget(self.button[21])
                    self.gci[ind].add_widget(self.button2[21])
                    self.gci[ind].add_widget(self.button3[21])
                    self.bx[ind].add_widget(self.gci[21])
                    self.ids.formoreinfo.add_widget(self.bx[21])
                elif ind == 22:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 23:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[22], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 23:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[22], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 22, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 22, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 22, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[22])
                    self.gci[ind].add_widget(self.button[22])
                    self.gci[ind].add_widget(self.button2[22])
                    self.gci[ind].add_widget(self.button3[22])
                    self.bx[ind].add_widget(self.gci[22])
                    self.ids.formoreinfo.add_widget(self.bx[22])
                elif ind == 23:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 24:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[23], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 24:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[23], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 23, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 23, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 23, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[23])
                    self.gci[ind].add_widget(self.button[23])
                    self.gci[ind].add_widget(self.button2[23])
                    self.gci[ind].add_widget(self.button3[23])
                    self.bx[ind].add_widget(self.gci[23])
                    self.ids.formoreinfo.add_widget(self.bx[23])
                elif ind == 24:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 25:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[24], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 25:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[24], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 24, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 24, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 24, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[24])
                    self.gci[ind].add_widget(self.button[24])
                    self.gci[ind].add_widget(self.button2[24])
                    self.gci[ind].add_widget(self.button3[24])
                    self.bx[ind].add_widget(self.gci[24])
                    self.ids.formoreinfo.add_widget(self.bx[24])
                elif ind == 25:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 26:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[25], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 26:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[25], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 25, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 25, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 25, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[25])
                    self.gci[ind].add_widget(self.button[25])
                    self.gci[ind].add_widget(self.button2[25])
                    self.gci[ind].add_widget(self.button3[25])
                    self.bx[ind].add_widget(self.gci[25])
                    self.ids.formoreinfo.add_widget(self.bx[25])
                elif ind == 26:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 27:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[26], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 27:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[26], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 26, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 26, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 26, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[26])
                    self.gci[ind].add_widget(self.button[26])
                    self.gci[ind].add_widget(self.button2[26])
                    self.gci[ind].add_widget(self.button3[26])
                    self.bx[ind].add_widget(self.gci[26])
                    self.ids.formoreinfo.add_widget(self.bx[26])
                elif ind == 27:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 28:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[27], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 28:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[27], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 27, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 27, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 27, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[27])
                    self.gci[ind].add_widget(self.button[27])
                    self.gci[ind].add_widget(self.button2[27])
                    self.gci[ind].add_widget(self.button3[27])
                    self.bx[ind].add_widget(self.gci[27])
                    self.ids.formoreinfo.add_widget(self.bx[27])
                elif ind == 28:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 29:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[28], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 29:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[28], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 28, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 28, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 28, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[28])
                    self.gci[ind].add_widget(self.button[28])
                    self.gci[ind].add_widget(self.button2[28])
                    self.gci[ind].add_widget(self.button3[28])
                    self.bx[ind].add_widget(self.gci[28])
                    self.ids.formoreinfo.add_widget(self.bx[28])
                elif ind == 29:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 30:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[29], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 30:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[29], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 29, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 29, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 29, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[29])
                    self.gci[ind].add_widget(self.button[29])
                    self.gci[ind].add_widget(self.button2[29])
                    self.gci[ind].add_widget(self.button3[29])
                    self.bx[ind].add_widget(self.gci[29])
                    self.ids.formoreinfo.add_widget(self.bx[29])
                elif ind == 30:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 31:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[30], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 31:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[30], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 30, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 30, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 30, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[30])
                    self.gci[ind].add_widget(self.button[30])
                    self.gci[ind].add_widget(self.button2[30])
                    self.gci[ind].add_widget(self.button3[30])
                    self.bx[ind].add_widget(self.gci[30])
                    self.ids.formoreinfo.add_widget(self.bx[30])
                elif ind == 31:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 32:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[31], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 32:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[31], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 31, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 31, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 31, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[31])
                    self.gci[ind].add_widget(self.button[31])
                    self.gci[ind].add_widget(self.button2[31])
                    self.gci[ind].add_widget(self.button3[31])
                    self.bx[ind].add_widget(self.gci[31])
                    self.ids.formoreinfo.add_widget(self.bx[31])
                elif ind == 32:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 33:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[32], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 33:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[32], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 32, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 32, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 32, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[32])
                    self.gci[ind].add_widget(self.button[32])
                    self.gci[ind].add_widget(self.button2[32])
                    self.gci[ind].add_widget(self.button3[32])
                    self.bx[ind].add_widget(self.gci[32])
                    self.ids.formoreinfo.add_widget(self.bx[32])
                elif ind == 33:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 34:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[33], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 34:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[33], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 33, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 33, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 33, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[33])
                    self.gci[ind].add_widget(self.button[33])
                    self.gci[ind].add_widget(self.button2[33])
                    self.gci[ind].add_widget(self.button3[33])
                    self.bx[ind].add_widget(self.gci[33])
                    self.ids.formoreinfo.add_widget(self.bx[50])
                elif ind == 34:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 35:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[34], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 35:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[34], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 34, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 34, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 34, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[34])
                    self.gci[ind].add_widget(self.button[34])
                    self.gci[ind].add_widget(self.button2[34])
                    self.gci[ind].add_widget(self.button3[34])
                    self.bx[ind].add_widget(self.gci[34])
                    self.ids.formoreinfo.add_widget(self.bx[34])
                elif ind == 35:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 36:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[35], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 36:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[35], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 35, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 35, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 35, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[35])
                    self.gci[ind].add_widget(self.button[35])
                    self.gci[ind].add_widget(self.button2[35])
                    self.gci[ind].add_widget(self.button3[35])
                    self.bx[ind].add_widget(self.gci[35])
                    self.ids.formoreinfo.add_widget(self.bx[35])
                elif ind == 36:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 37:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[36], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 37:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[36], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 36, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 36, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 36, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[36])
                    self.gci[ind].add_widget(self.button[36])
                    self.gci[ind].add_widget(self.button2[36])
                    self.gci[ind].add_widget(self.button3[36])
                    self.bx[ind].add_widget(self.gci[36])
                    self.ids.formoreinfo.add_widget(self.bx[36])
                elif ind == 37:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 38:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[37], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 38:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[37], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 37, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 37, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 37, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[37])
                    self.gci[ind].add_widget(self.button[37])
                    self.gci[ind].add_widget(self.button2[37])
                    self.gci[ind].add_widget(self.button3[37])
                    self.bx[ind].add_widget(self.gci[37])
                    self.ids.formoreinfo.add_widget(self.bx[37])
                elif ind == 38:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 39:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[38], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 39:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[38], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 38, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 38, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 38, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[38])
                    self.gci[ind].add_widget(self.button[38])
                    self.gci[ind].add_widget(self.button2[38])
                    self.gci[ind].add_widget(self.button3[38])
                    self.bx[ind].add_widget(self.gci[38])
                    self.ids.formoreinfo.add_widget(self.bx[38])
                elif ind == 39:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 40:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[39], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 40:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[39], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 39, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 39, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 39, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[39])
                    self.gci[ind].add_widget(self.button[39])
                    self.gci[ind].add_widget(self.button2[39])
                    self.gci[ind].add_widget(self.button3[39])
                    self.bx[ind].add_widget(self.gci[39])
                    self.ids.formoreinfo.add_widget(self.bx[39])
                elif ind == 40:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 41:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[40], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 41:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[40], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 40, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 40, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 40, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[40])
                    self.gci[ind].add_widget(self.button[40])
                    self.gci[ind].add_widget(self.button2[40])
                    self.gci[ind].add_widget(self.button3[40])
                    self.bx[ind].add_widget(self.gci[40])
                    self.ids.formoreinfo.add_widget(self.bx[40])
                elif ind == 41:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 42:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[41], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 42:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[41], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 41, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 41, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 41, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[41])
                    self.gci[ind].add_widget(self.button[41])
                    self.gci[ind].add_widget(self.button2[41])
                    self.gci[ind].add_widget(self.button3[41])
                    self.bx[ind].add_widget(self.gci[41])
                    self.ids.formoreinfo.add_widget(self.bx[41])
                elif ind == 42:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 43:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[42], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 43:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[42], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 42, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 42, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 42, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[42])
                    self.gci[ind].add_widget(self.button[42])
                    self.gci[ind].add_widget(self.button2[42])
                    self.gci[ind].add_widget(self.button3[42])
                    self.bx[ind].add_widget(self.gci[42])
                    self.ids.formoreinfo.add_widget(self.bx[42])
                elif ind == 43:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 44:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[43], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 44:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[43], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 43, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 43, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 43, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[43])
                    self.gci[ind].add_widget(self.button[43])
                    self.gci[ind].add_widget(self.button2[43])
                    self.gci[ind].add_widget(self.button3[43])
                    self.bx[ind].add_widget(self.gci[43])
                    self.ids.formoreinfo.add_widget(self.bx[43])
                elif ind == 44:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 45:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[44], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 45:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[44], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 44, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 44, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 44, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[44])
                    self.gci[ind].add_widget(self.button[44])
                    self.gci[ind].add_widget(self.button2[44])
                    self.gci[ind].add_widget(self.button3[44])
                    self.bx[ind].add_widget(self.gci[44])
                    self.ids.formoreinfo.add_widget(self.bx[44])
                elif ind == 45:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 46:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[45], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 46:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[45], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 45, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 45, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 45, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[45])
                    self.gci[ind].add_widget(self.button[45])
                    self.gci[ind].add_widget(self.button2[45])
                    self.gci[ind].add_widget(self.button3[45])
                    self.bx[ind].add_widget(self.gci[45])
                    self.ids.formoreinfo.add_widget(self.bx[45])
                elif ind == 46:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 47:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[46], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 47:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[46], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 46, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 46, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 46, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[46])
                    self.gci[ind].add_widget(self.button[46])
                    self.gci[ind].add_widget(self.button2[46])
                    self.gci[ind].add_widget(self.button3[46])
                    self.bx[ind].add_widget(self.gci[46])
                    self.ids.formoreinfo.add_widget(self.bx[46])
                elif ind == 47:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 48:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[47], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 48:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[47], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 47, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 47, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 47, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[47])
                    self.gci[ind].add_widget(self.button[47])
                    self.gci[ind].add_widget(self.button2[47])
                    self.gci[ind].add_widget(self.button3[47])
                    self.bx[ind].add_widget(self.gci[47])
                    self.ids.formoreinfo.add_widget(self.bx[47])
                elif ind == 48:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 49:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[48], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 49:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[48], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 48, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 48, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 48, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[48])
                    self.gci[ind].add_widget(self.button[48])
                    self.gci[ind].add_widget(self.button2[48])
                    self.gci[ind].add_widget(self.button3[48])
                    self.bx[ind].add_widget(self.gci[48])
                    self.ids.formoreinfo.add_widget(self.bx[48])
                elif ind == 49:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 50:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[49], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 50:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[49], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 49, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 49, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 49, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])
                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[49])
                    self.gci[ind].add_widget(self.button[49])
                    self.gci[ind].add_widget(self.button2[49])
                    self.gci[ind].add_widget(self.button3[49])
                    self.bx[ind].add_widget(self.gci[49])
                    self.ids.formoreinfo.add_widget(self.bx[49])
                elif ind == 50:
                    if not_in(ind, excluded):
                        self.label[ind] = Label(text=forstr, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 51:
                            addfdnametotext(("i," + str(getattr(dayfordisplay.usedfoodsincount[50], igd))), mifiletwo)
                    else:
                        self.label[ind] = Label(text=forstrx, size_hint_y=0.1)
                        with open(mifiletwo, 'r') as g:
                            tttg = len(g.readlines())
                            g.close()
                        if tttg < 51:
                            addfdnametotext(("x," + str(getattr(dayfordisplay.usedfoodsincount[50], igd))), mifiletwo)

                    self.button[ind] = Button(text="Edit", size_hint_y=0.1,
                                              on_release=lambda q: self.edit(igd, 50, dayfordisplay))
                    self.button2[ind] = Button(text="Include", size_hint_y=0.1,
                                               on_release=lambda p: self.include(igd, 50, dayfordisplay))
                    self.button3[ind] = Button(text="Exclude", size_hint_y=0.1,
                                               on_release=lambda d: self.exclude(igd, 50, dayfordisplay))
                    self.bx[ind] = GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                              padding=[10, 10, 10, 10])

                    self.gci[ind] = GridLayout(cols=1, rows=3)
                    self.bx[ind].add_widget(self.label[50])
                    self.gci[ind].add_widget(self.button[50])
                    self.gci[ind].add_widget(self.button2[50])
                    self.gci[ind].add_widget(self.button3[50])
                    self.bx[ind].add_widget(self.gci[50])
                    self.ids.formoreinfo.add_widget(self.bx[50])
                addfdnametotext(dayfordisplay.usedfoodsincount[fd].food_name, mifile)
            fd += 1
        with open(mifiletwo, 'r') as mft:
            vv = mft.readlines()
            print("VVVVVVVVVVVVVV")
            print(vv)
            if len(vv) > 0:
                if vv[len(vv) - 1].find('i') == -1 and vv[len(vv) - 1].find('x') == -1:
                    vv.pop(len(vv) - 1)
            print(vv)
            mft.close()
        text = ""
        for h in vv:
            text += h
        str_to_execute = "UPDATE client_moreinfo_value_storage SET " + strtoselect + " = \'" + text.replace('None',
                                                                                                            '').replace(
            '\\n', '\n') + "\' WHERE username = \'" + user + "\' AND datetime = \'" + str(datetime.datetime.now())[
                                                                                      0:10] + "\'"
        print(str_to_execute)
        concur.execute(str_to_execute)
        concur.execute("COMMIT")
        with open(mifiletwo, 'w') as mftw:
            mftw.write(text)
            mftw.close()
        print("x")

    def destroy(self):
        self.descheduler()
        filestr = str(running_id) + user + '_mitype.txt'
        with open(filestr, 'r') as rop:
            try:
                igd = rop.readlines()[0].strip()
            except IndexError:
                igd = 'calories'
            rop.close()
        mifl = defineifmissingmisession(igd)
        self.ids.formoreinfo.clear_widgets(children=None)
        with open(mifl, 'w') as mfle:
            mfle.write("")
            mfle.close()


class DealsPage(Screen):
    pass


class Setting(Screen):
    def logout(self):
        global loggedoutb4
        global loggedin
        loggedoutb4 = True
        loggedin = False
        self.manager.current = "signin"


class BusinessSignIn(Screen):
    def b_sign_in(self):
        cuser = self.ids.usernameb.text
        global user
        user = cuser
        password = self.ids.passwordb.text
        global con
        con = psycopg2.connect(
            database="dietfriendcab", user=cuser, password=password, host='127.0.0.1', port='5432'
        )
        cur = con.cursor()
        global theme
        global primary_p
        past = defineifmissing_prev_insecure_settings()
        if row_exists_theme(cur, user):
            query = sql.SQL(
                "SELECT bg_theme FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            theme_setting = str(cur.fetchall())
            print("theme_setting:")
            print(theme_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_theme_index = int(random() * 100) % 2
            if random_theme_index == 0:
                theme = "Light"
            else:
                theme = "Dark"
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        if row_exists_primary_p(cur, user):
            query = sql.SQL(
                "SELECT primary_p FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            primary_p_setting = str(cur.fetchall())
            print("primary_p_setting:")
            print(primary_p_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_primary_p_index = int(random() * 100) % 10
            if random_primary_p_index == 0:
                primary_p = "Teal"
            elif random_primary_p_index == 1:
                primary_p = "Red"
            elif random_primary_p_index == 2:
                primary_p = "Pink"
            elif random_primary_p_index == 3:
                primary_p = "Indigo"
            elif random_primary_p_index == 4:
                primary_p = "Blue"
            elif random_primary_p_index == 5:
                primary_p = "LightBlue"
            elif random_primary_p_index == 6:
                primary_p = "Lime"
            elif random_primary_p_index == 7:
                primary_p = "Yellow"
            elif random_primary_p_index == 8:
                primary_p = "Orange"
            else:
                primary_p = "Amber"
            query = sql.SQL(
                "INSERT INTO client_settings (username, primary_p, bg_theme) VALUES (\'" + user + "\', \'" + primary_p + "\', \'" + theme + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            cur.execute("COMMIT")
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
            ############## FIX THIS
            path = os.path.join(os.path.dirname(__file__), "B_DietFriend_Pictures")
            if not os.path.exists(path):
                os.makedirs(path)
        global universal_list
        """
        try:
            print("Signed in, constructing universal_list")
            with open(str(datetime.datetime.now())[0:10] + str(user) + ".txt", 'r') as p:
                lines = p.readlines()
                p.close()
            words = []
            for i in lines:
                words.append(i[0:i.find(' ')])
            with open(str(datetime.datetime.now())[0:10] + str(user) + "_type.txt", 'r') as p:
                lines_two = p.readlines()
                p.close()
            numservings = []
            h = 0
            while h < len(lines_two):
                numservings.append(float(lines_two[h][lines_two[h].find(',') + 1:lines_two[h].find('\n')]))
                h += 1
            universal_list = [[], words, numservings]
            print("universal_list: ")
            print(universal_list)
        except:
            universal_list = [[], [], []]
        # UNCOMMENT ABOVE + COMMENT LINE BELOW THIS FOR REAL PRODUCT
        """
        universal_list = [[], ['chili_magic', 'tuna_can', 'wolf_brand_chili_no_beans'], [1.00, 1.00, 1.00]]


class BusinessSignUp(Screen):
    def businessconnectme(self):
        global user
        cuser = None
        global con
        try:
            cur = con.cursor()
            cur.execute('SELECT version()')
            db_version = cur.fetchone()
            print(db_version)
            cuser = self.ids.usernameub.text
            user = cuser
            password = self.ids.passwordub.text
            query = sql.SQL("CREATE USER {username} WITH PASSWORD {password}").format(
                username=sql.Identifier(cuser),
                password=sql.Placeholder()
            )
            print("Is error here?")
            cur.execute(query, (password,))
            print("No")
            cur.execute("COMMIT")
            print("No")
            query = sql.SQL("GRANT ALL ON dietfriend_client_food_data TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL ON dietfriend_business_food_data TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL ON client_settings TO {0}").format(
                sql.Identifier(cuser)
            )
            print("absent")
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL ON business_followers TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL ON business_food_search TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            ######################
            query = sql.SQL("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            query = sql.SQL("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {0}").format(
                sql.Identifier(cuser)
            )
            cur.execute(query.as_string(con))
            cur.execute("COMMIT")
            ######################
            print("absent1")
            str_to_execute = "INSERT INTO dietfriend_usernames_and_passwords_business(username, password) VALUES(\'" \
                             + cuser + "\', \'" + password + "\')"
            cur.execute(str_to_execute)
            str_to_execute = "INSERT INTO business_followers(business_name, num_followers) VALUES(\'" \
                             + cuser + "\', \'" + str(0) + "\')"
            cur.execute(str_to_execute)
            print("absent2")
            con.close()
            print('Database connection closed.')
            con = psycopg2.connect(database="dietfriendcab", user=cuser,
                                   password=password, host='127.0.0.1', port='5432')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        global theme
        global primary_p
        cur = con.cursor()
        past = defineifmissing_prev_insecure_settings()
        if row_exists_theme(cur, user):
            query = sql.SQL(
                "SELECT bg_theme FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            theme_setting = str(cur.fetchall())
            print("theme_setting:")
            print(theme_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_theme_index = int(random() * 100) % 2
            if random_theme_index == 0:
                theme = "Light"
            else:
                theme = "Dark"
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        if row_exists_primary_p(cur, user):
            query = sql.SQL(
                "SELECT primary_p FROM client_settings WHERE (username = \'" + cuser + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            primary_p_setting = str(cur.fetchall())
            print("primary_p_setting:")
            print(primary_p_setting)
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        else:
            random_primary_p_index = int(random() * 100) % 10
            if random_primary_p_index == 0:
                primary_p = "Teal"
            elif random_primary_p_index == 1:
                primary_p = "Red"
            elif random_primary_p_index == 2:
                primary_p = "Pink"
            elif random_primary_p_index == 3:
                primary_p = "Indigo"
            elif random_primary_p_index == 4:
                primary_p = "Blue"
            elif random_primary_p_index == 5:
                primary_p = "LightBlue"
            elif random_primary_p_index == 6:
                primary_p = "Lime"
            elif random_primary_p_index == 7:
                primary_p = "Yellow"
            elif random_primary_p_index == 8:
                primary_p = "Orange"
            else:
                primary_p = "Amber"
            query = sql.SQL(
                "INSERT INTO client_settings (username, primary_p, bg_theme) VALUES (\'" + user + "\', \'" + primary_p + "\', \'" + theme + "\')")
            print("Query: ")
            print(query)
            cur.execute(query)
            cur.execute("COMMIT")
            with open(past[0], 'w') as xg:
                xg.write(user)
                xg.close()
        global universal_list
        """
        try:
            print("Signed in, constructing universal_list")
            with open(str(datetime.datetime.now())[0:10] + str(user) + ".txt", 'r') as p:
                lines = p.readlines()
                p.close()
            words = []
            for i in lines:
                words.append(i[0:i.find(' ')])
            with open(str(datetime.datetime.now())[0:10] + str(user) + "_type.txt", 'r') as p:
                lines_two = p.readlines()
                p.close()
            numservings = []
            h = 0
            while h < len(lines_two):
                numservings.append(float(lines_two[h][lines_two[h].find(',') + 1:lines_two[h].find('\n')]))
                h += 1
            universal_list = [[], words, numservings]
            print("universal_list: ")
            print(universal_list)
        except:
            universal_list = [[], [], []]
        # UNCOMMENT ABOVE + COMMENT LINE BELOW THIS FOR REAL PRODUCT
        """
        universal_list = [[], ['chili_magic', 'tuna_can', 'wolf_brand_chili_no_beans'], [1.00, 1.00, 1.00]]


class B_MiSpecPopUp(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.box = None
        self.gc = {}
        self.lb = {}
        self.estonlyindividual = {}
        self.textinput = {}
        self.estallindividual = None
        self.btn = None

    def b_setter(self, cd, bxstoch, ftindd):
        b_individualssuperwriteest(cd, bxstoch, ftindd)
        est = b_defineifmissingestref()
        b_estdec()
        with open(est, 'r') as d:
            dd = d.readlines()
            d.close()
        forester = dd[ftindd]
        print("HERE FORESTER")
        ester = getfd(forester)
        z = 0
        while z < len(bxstoch):
            self.textinput[z].text = getattr(ester, foodattrnamelst[z])
            z += 1

    def b_loader(self, ftind):
        cd = business_doprocess()
        self.title = "Edit Food: " + cd.usedfoodsincount[ftind].food_name + ":"
        ind = 0
        self.box = (BoxLayout(orientation='vertical', padding=[10, 10, 10, 10]))
        self.estallindividual = Button(text="Estimate All",
                                       on_release=lambda x: self.b_setter(cd,
                                                                          [(self.textinput[0].text, 2),
                                                                           (self.textinput[1].text, 3),
                                                                           (self.textinput[2].text, 4),
                                                                           (self.textinput[3].text, 5),
                                                                           (self.textinput[4].text, 6),
                                                                           (self.textinput[5].text, 7),
                                                                           (self.textinput[6].text, 8),
                                                                           (self.textinput[7].text, 9),
                                                                           (self.textinput[8].text, 10),
                                                                           (self.textinput[9].text, 11),
                                                                           (self.textinput[10].text, 12),
                                                                           (self.textinput[11].text, 13),
                                                                           (self.textinput[12].text, 14),
                                                                           (self.textinput[13].text, 15),
                                                                           (self.textinput[14].text, 16),
                                                                           (self.textinput[15].text, 17),
                                                                           (self.textinput[16].text, 18),
                                                                           (self.textinput[17].text, 19),
                                                                           (self.textinput[18].text, 20)], ftind))
        self.box.add_widget(self.estallindividual)
        while ind < len(foodattrnamelst):
            self.gc[ind] = (GridLayout(cols=3))
            self.lb[ind] = (Label(text=foodattrnamelst[ind], size_hint_y=0.1))
            self.estonlyindividual[ind] = (Button(text="Estimate",
                                                  on_release=lambda x: self.b_setter(cd,
                                                                                     [(self.textinput[ind].text,
                                                                                       ind + 2)], ftind)))
            self.textinput[ind] = (TextInput(text=str(getattr(cd.usedfoodsincount[ftind], foodattrnamelst[ind])),
                                             size_hint_y=0.1))
            self.gc[ind].add_widget(self.lb[ind])
            self.gc[ind].add_widget(self.textinput[ind])
            self.box.add_widget(self.gc[ind])
            ind += 1
        self.btn = Button(text="Apply Changes", on_release=lambda x: self.b_returner(cd, ftind,
                                                                                     [self.textinput[0].text,
                                                                                      self.textinput[1].text,
                                                                                      self.textinput[2].text,
                                                                                      self.textinput[3].text,
                                                                                      self.textinput[4].text,
                                                                                      self.textinput[5].text,
                                                                                      self.textinput[6].text,
                                                                                      self.textinput[7].text,
                                                                                      self.textinput[8].text,
                                                                                      self.textinput[9].text,
                                                                                      self.textinput[10].text,
                                                                                      self.textinput[11].text,
                                                                                      self.textinput[12].text,
                                                                                      self.textinput[13].text,
                                                                                      self.textinput[14].text,
                                                                                      self.textinput[15].text,
                                                                                      self.textinput[16].text,
                                                                                      self.textinput[17].text,
                                                                                      self.textinput[18].text
                                                                                      ]))
        self.box.add_widget(self.btn)
        self.add_widget(self.box)
        self.open()

    def b_returner(self, cd, f_ind, textinputlst):
        fd = Food(getattr(cd.usedfoodsincount[f_ind], 'food_name'),
                  getattr(cd.usedfoodsincount[f_ind], 'food_datetime'),
                  float(textinputlst[0]), float(textinputlst[1]), float(textinputlst[2]), float(textinputlst[3]),
                  float(textinputlst[4]), float(textinputlst[5]), float(textinputlst[6]), float(textinputlst[7]),
                  float(textinputlst[8]), float(textinputlst[9]), float(textinputlst[10]), float(textinputlst[11]),
                  float(textinputlst[12]), float(textinputlst[13]), float(textinputlst[14]), float(textinputlst[15]),
                  float(textinputlst[16]), float(textinputlst[17]), float(textinputlst[18]))
        tempfd = cd.usedfoodsincount[f_ind]
        cd.usedfoodsincount[f_ind] = fd
        fileindivid = b_defineifmissingindividuals()
        strtowrite = ""
        i = 0
        while i < 21:
            if i >= 2:
                ii = i - 2
                strtowrite += ('r,' + str(getattr(cd.usedfoodsincount[f_ind], foodattrnamelst[ii])) + ' ')
            elif i == 1:
                strtowrite += ('r,' + str(getattr(cd.usedfoodsincount[f_ind], 'food_datetime')) + ' ')
            else:
                strtowrite += ('r,' + str(getattr(cd.usedfoodsincount[f_ind], 'food_name')) + ' ')
            i += 1
        strtowrite += '\n'
        b_writetolineindividuals(cd, f_ind + 1, '\n')
        b_writetolineindividuals(cd, f_ind + 1, strtowrite)
        foodsubtractpreadd(cd, tempfd, fd)
        dst = b_defineifmissingtype()
        with open(dst, 'r') as dstr:
            f = dstr.readlines()
            dstr.close()
        strforwrite = f[f_ind]
        strforwrite = strforwrite.replace('t', 'r')
        strforwrite = strforwrite.replace('e', 'r')
        b_writetoline(f_ind + 1, '\n')
        b_writetoline(f_ind + 1, strforwrite)
        if b_estimatefromcustomchecker(f_ind, fileindivid, cd):
            with open(dst, 'r') as dstr:
                f = dstr.readlines()
                dstr.close()
            strforwrite = f[f_ind]
            strforwrite = strforwrite.replace('r', 'e')
            b_writetoline(f_ind + 1, '\n')
            b_writetoline(f_ind + 1, strforwrite)

        global user
        global con
        datestring = cd.date_time
        datestring = str(datestring)
        datestring = datestring[0:10]
        curcursor = con.cursor()
        indtext = b_defineifmissingindividuals()
        with open(indtext, 'r') as readind:
            indtextlist = readind.readlines()
            print("indtextlist: ")
            readind.close()
        indtxt = ""
        u = 0
        while u < len(indtextlist):
            indtxt += indtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_business_food_data SET individualstextfile_r = \'" + indtxt + \
            "\' WHERE username = \'" + user + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        typtext = b_defineifmissingtype()
        with open(typtext, 'r') as readtyp:
            typtextlist = readtyp.readlines()
            print("typtextlist: ")
            readtyp.close()
        typtxt = ""
        u = 0
        while u < len(typtextlist):
            typtxt += typtextlist[u]
            u += 1
        str_to_execute = \
            "UPDATE dietfriend_business_food_data SET typefile_tx_ex_rx_ = \'" + typtxt + "\' WHERE username = \'" + \
            user + "\'"
        print(str_to_execute)
        curcursor.execute(str_to_execute)
        curcursor.execute("COMMIT")

        self.remove_widget(self.box)
        self.dismiss()


def appliermispec(fd_name):
    b_partialestref()
    dayfordisplay = business_doprocess()
    fd = None
    fd_index = None
    fd_indx = 0
    while fd_indx < len(dayfordisplay.usedfoodsincount):
        if fd_name == dayfordisplay.usedfoodsincount[fd_indx].food_name:
            fd_index = fd_indx
            fd = dayfordisplay.usedfoodsincount[fd_indx]
            break
        fd_indx += 1
    print("Through for")
    fdr = (fd, fd_index)
    return fdr


class BusinessMISpecFood(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.editrbtn = None

    def tempapply(self):
        global universal_list
        global user
        path = os.path.join(os.path.dirname(__file__), "B_DietFriend_Pictures"+user)
        c = b_defineifmissingmispecsession()
        with open(c, 'r') as d:
            fd_name = d.readlines()[0].strip()
            d.close()
        fdr = appliermispec(fd_name)
        fd = fdr[0]
        fd_index = fdr[1]
        self.editrbtn = MDFillRoundFlatIconButton(text="Edit", icon="flask-empty-plus", font_size=16,
                                                  on_release=lambda x: b_applyindedits(lb=4, duz=fd))
        self.ids.flt.add_widget(self.editrbtn)
        try:
            print(universal_list[0][fd_index])
            self.ids.mispecfoodimage.source = path + "\\" + str(universal_list[0][fd_index])
            print("Image worked!")
            self.ids.fdnme.text = '\n\n\n\n' + str(fd.food_name)
            self.servingmispec.text = str(fd.serving)
            self.caloriesmispec.text = str(fd.calories)
            self.totalfatmispec.text = str(fd.total_fat)
            self.saturatedfatmispec.text = str(fd.saturated_fat)
            self.transfatmispec.text = str(fd.trans_fat)
            self.cholesterolmispec.text = str(fd.cholesterol)
            self.sodiummispec.text = str(fd.sodium)
            self.totalcarbmispec.text = str(fd.total_carb)
            self.fibermispec.text = str(fd.fiber)
            self.totalsugarsmispec.text = str(fd.total_sugars)
            self.addedsugarsmispec.text = str(fd.added_sugars)
            self.proteinmispec.text = str(fd.protein)
            self.calciummispec.text = str(fd.calcium)
            self.ironmispec.text = str(fd.iron)
            self.potassiummispec.text = str(fd.potassium)
            self.vitaminamispec.text = str(fd.vitamin_a)
            self.vitaminbmispec.text = str(fd.vitamin_b)
            self.vitamincmispec.text = str(fd.vitamin_c)
            self.vitamindmispec.text = str(fd.vitamin_d)
            print("Text worked!")
        except:
            pass

    def mispecdestroy(self):
        self.ids.flt.clear_widgets()


class BusinessHome(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.box = {}
        self.lb = {}
        self.gc = {}
        self.b = {}
        self.b2 = {}

    def bfdremover(self):
        p = 0
        while p < len(self.box):
            fdname = self.b2[p].name[0:len(self.b2[p].name)]
            if self.box[p].name[0:len(self.box[p].name)] == fdname:
                self.name.curbfoods.remove_widget(self.box[p])
            p += 1

    def recordx(self, fd_name):
        c = b_defineifmissingflsession()
        with open(c, 'a') as f:
            f.write(fd_name.text + "\n")
            f.close()

    def tempapplyswitcher(self, fd_name, fd):
        kt = b_defineifmissingmispecsession()
        with open(kt, 'w') as hap:
            hap.write(fd_name.text + "\n")
            hap.close()
        self.manager.current = "mispec"

    def onloadbfdl(self):
        lst = []
        b_partialestref()
        global con
        global user
        crs = con.cursor()
        query = sql.SQL(
            "SELECT num_followers FROM business_followers WHERE (business_name = \'" + user + "\')")
        print("Query: ")
        print(query)
        crs.execute(query)
        numfollowers = str(crs.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(', '').replace(
            ')', '').replace(',', '')
        print(numfollowers)
        self.ids.bfllwrs.text = str(numfollowers) + " Followers"
        dayfordisplay = business_doprocess()
        fdfile = b_defineifmissingflsession()
        print("x")
        with open(fdfile, 'r') as fdf:
            fdl = fdf.readlines()
            fdf.close()
        if len(fdl) == 0:
            with open(fdfile, 'w') as p:
                p.close()
        fd = 0
        print("x")
        while fd < len(dayfordisplay.usedfoodsincount):
            if not isinfd(fdl, fd, dayfordisplay.usedfoodsincount):
                ind = fd
                self.box[ind] = (GridLayout(cols=2, rows=1, size_hint=(1, self.size_hint_min_y),
                                            padding=[10, 2, 10, 2]))
                self.lb[ind] = (Label(text=dayfordisplay.usedfoodsincount[ind].food_name, size_hint_y=0.1))
                self.gc[ind] = (GridLayout(cols=1, rows=2))
                """Edit Buttons START"""
                if ind == 0:
                    # if self.lb[0].text != '' and self.lb[0] is not None:
                    #    self.recordx(self.lb[0])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[0], dayfordisplay.usedfoodsincount[0])))
                elif ind == 1:
                    # if self.lb[1].text != '' and self.lb[1] is not None:
                    #    self.recordx(self.lb[1])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[1], dayfordisplay.usedfoodsincount[1])))
                elif ind == 2:
                    # if self.lb[2].text != '' and self.lb[2] is not None:
                    #    self.recordx(self.lb[2])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[2], dayfordisplay.usedfoodsincount[2])))
                elif ind == 3:
                    # if self.lb[3].text != '' and self.lb[3] is not None:
                    #    self.recordx(self.lb[3])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[3], dayfordisplay.usedfoodsincount[3])))
                elif ind == 4:
                    # if self.lb[4].text != '' and self.lb[4] is not None:
                    #     self.recordx(self.lb[4])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[4], dayfordisplay.usedfoodsincount[4])))
                elif ind == 5:
                    # if self.lb[5].text != '' and self.lb[5] is not None:
                    #     self.recordx(self.lb[5])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[5], dayfordisplay.usedfoodsincount[5])))
                elif ind == 6:
                    # if self.lb[6].text != '' and self.lb[6] is not None:
                    #     self.recordx(self.lb[6])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[6], dayfordisplay.usedfoodsincount[6])))
                elif ind == 7:
                    # if self.lb[7].text != '' and self.lb[7] is not None:
                    #     self.recordx(self.lb[7])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[7], dayfordisplay.usedfoodsincount[7])))
                elif ind == 8:
                    # if self.lb[8].text != '' and self.lb[8] is not None:
                    #     self.recordx(self.lb[8])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[8], dayfordisplay.usedfoodsincount[8])))
                elif ind == 9:
                    # if self.lb[9].text != '' and self.lb[9] is not None:
                    #     self.recordx(self.lb[9])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[9], dayfordisplay.usedfoodsincount[9])))
                elif ind == 10:
                    # if self.lb[10].text != '' and self.lb[10] is not None:
                    #     self.recordx(self.lb[10])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[10], dayfordisplay.usedfoodsincount[10])))
                elif ind == 11:
                    # if self.lb[11].text != '' and self.lb[11] is not None:
                    #     self.recordx(self.lb[11])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[11], dayfordisplay.usedfoodsincount[11])))
                elif ind == 12:
                    # if self.lb[12].text != '' and self.lb[12] is not None:
                    #     self.recordx(self.lb[12])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[12], dayfordisplay.usedfoodsincount[12])))
                elif ind == 13:
                    # if self.lb[13].text != '' and self.lb[13] is not None:
                    #     self.recordx(self.lb[13])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[13], dayfordisplay.usedfoodsincount[13])))
                elif ind == 14:
                    # if self.lb[14].text != '' and self.lb[14] is not None:
                    #     self.recordx(self.lb[14])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[14], dayfordisplay.usedfoodsincount[14])))
                elif ind == 15:
                    # if self.lb[15].text != '' and self.lb[15] is not None:
                    #     self.recordx(self.lb[15])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[15], dayfordisplay.usedfoodsincount[15])))
                elif ind == 16:
                    # if self.lb[16].text != '' and self.lb[16] is not None:
                    #     self.recordx(self.lb[16])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[16], dayfordisplay.usedfoodsincount[16])))
                elif ind == 17:
                    # if self.lb[17].text != '' and self.lb[17] is not None:
                    #     self.recordx(self.lb[17])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[17], dayfordisplay.usedfoodsincount[17])))
                elif ind == 18:
                    # if self.lb[18].text != '' and self.lb[18] is not None:
                    #     self.recordx(self.lb[18])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[18], dayfordisplay.usedfoodsincount[18])))
                elif ind == 19:
                    # if self.lb[19].text != '' and self.lb[19] is not None:
                    #     self.recordx(self.lb[19])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[19], dayfordisplay.usedfoodsincount[19])))
                elif ind == 20:
                    # if self.lb[20].text != '' and self.lb[20] is not None:
                    #     self.recordx(self.lb[20])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[20], dayfordisplay.usedfoodsincount[20])))
                elif ind == 21:
                    # if self.lb[21].text != '' and self.lb[21] is not None:
                    #     self.recordx(self.lb[21])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[21], dayfordisplay.usedfoodsincount[21])))
                elif ind == 22:
                    # if self.lb[22].text != '' and self.lb[22] is not None:
                    #     self.recordx(self.lb[22])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[22], dayfordisplay.usedfoodsincount[22])))
                elif ind == 23:
                    # if self.lb[23].text != '' and self.lb[23] is not None:
                    #     self.recordx(self.lb[23])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[23], dayfordisplay.usedfoodsincount[23])))
                elif ind == 24:
                    # if self.lb[24].text != '' and self.lb[24] is not None:
                    #     self.recordx(self.lb[24])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[24], dayfordisplay.usedfoodsincount[24])))
                elif ind == 25:
                    # if self.lb[25].text != '' and self.lb[25] is not None:
                    #     self.recordx(self.lb[25])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[25], dayfordisplay.usedfoodsincount[25])))
                elif ind == 26:
                    # if self.lb[26].text != '' and self.lb[26] is not None:
                    #     self.recordx(self.lb[26])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[26], dayfordisplay.usedfoodsincount[26])))
                elif ind == 27:
                    # if self.lb[27].text != '' and self.lb[27] is not None:
                    #     self.recordx(self.lb[27])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[27], dayfordisplay.usedfoodsincount[27])))
                elif ind == 28:
                    # if self.lb[28].text != '' and self.lb[28] is not None:
                    #     self.recordx(self.lb[28])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[28], dayfordisplay.usedfoodsincount[28])))
                elif ind == 29:
                    # if self.lb[29].text != '' and self.lb[29] is not None:
                    #     self.recordx(self.lb[29])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[29], dayfordisplay.usedfoodsincount[29])))
                elif ind == 30:
                    # if self.lb[30].text != '' and self.lb[30] is not None:
                    #     self.recordx(self.lb[30])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[30], dayfordisplay.usedfoodsincount[30])))
                elif ind == 31:
                    # if self.lb[31].text != '' and self.lb[31] is not None:
                    #     self.recordx(self.lb[31])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[31], dayfordisplay.usedfoodsincount[31])))
                elif ind == 32:
                    # if self.lb[32].text != '' and self.lb[32] is not None:
                    #     self.recordx(self.lb[32])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[32], dayfordisplay.usedfoodsincount[32])))
                elif ind == 33:
                    # if self.lb[33].text != '' and self.lb[33] is not None:
                    #     self.recordx(self.lb[33])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[33], dayfordisplay.usedfoodsincount[33])))
                elif ind == 34:
                    # if self.lb[34].text != '' and self.lb[34] is not None:
                    #     self.recordx(self.lb[34])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[34], dayfordisplay.usedfoodsincount[34])))
                elif ind == 35:
                    # if self.lb[35].text != '' and self.lb[35] is not None:
                    #     self.recordx(self.lb[35])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[35], dayfordisplay.usedfoodsincount[35])))
                elif ind == 36:
                    # if self.lb[36].text != '' and self.lb[36] is not None:
                    #     self.recordx(self.lb[36])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[36], dayfordisplay.usedfoodsincount[36])))
                elif ind == 37:
                    # if self.lb[37].text != '' and self.lb[37] is not None:
                    #     self.recordx(self.lb[37])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[37], dayfordisplay.usedfoodsincount[37])))
                elif ind == 38:
                    # if self.lb[38].text != '' and self.lb[38] is not None:
                    #     self.recordx(self.lb[38])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[38], dayfordisplay.usedfoodsincount[38])))
                elif ind == 39:
                    # if self.lb[39].text != '' and self.lb[39] is not None:
                    #     self.recordx(self.lb[39])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[39], dayfordisplay.usedfoodsincount[39])))
                elif ind == 40:
                    # if self.lb[40].text != '' and self.lb[40] is not None:
                    #     self.recordx(self.lb[40])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[40], dayfordisplay.usedfoodsincount[40])))
                elif ind == 41:
                    # if self.lb[41].text != '' and self.lb[41] is not None:
                    #     self.recordx(self.lb[41])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[41], dayfordisplay.usedfoodsincount[41])))
                elif ind == 42:
                    # if self.lb[42].text != '' and self.lb[42] is not None:
                    #     self.recordx(self.lb[42])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[42], dayfordisplay.usedfoodsincount[42])))
                elif ind == 43:
                    # if self.lb[43].text != '' and self.lb[43] is not None:
                    #     self.recordx(self.lb[43])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[43], dayfordisplay.usedfoodsincount[43])))
                elif ind == 44:
                    # if self.lb[44].text != '' and self.lb[44] is not None:
                    #     self.recordx(self.lb[44])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[44], dayfordisplay.usedfoodsincount[44])))
                elif ind == 45:
                    # if self.lb[45].text != '' and self.lb[45] is not None:
                    #     self.recordx(self.lb[45])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[45], dayfordisplay.usedfoodsincount[45])))
                elif ind == 46:
                    # if self.lb[46].text != '' and self.lb[46] is not None:
                    #     self.recordx(self.lb[46])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[46], dayfordisplay.usedfoodsincount[46])))
                elif ind == 47:
                    # if self.lb[47].text != '' and self.lb[47] is not None:
                    #     self.recordx(self.lb[47])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[47], dayfordisplay.usedfoodsincount[47])))
                elif ind == 48:
                    # if self.lb[48].text != '' and self.lb[48] is not None:
                    #     self.recordx(self.lb[48])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[48], dayfordisplay.usedfoodsincount[48])))
                elif ind == 49:
                    # if self.lb[49].text != '' and self.lb[49] is not None:
                    #     self.recordx(self.lb[49])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[49], dayfordisplay.usedfoodsincount[49])))
                else:
                    # if self.lb[50].text != '' and self.lb[50] is not None:
                    #     self.recordx(self.lb[50])
                    self.b[ind] = (Button(text="View", size_hint_y=0.1,
                                          on_press=lambda x:
                                          self.tempapplyswitcher(self.lb[50], dayfordisplay.usedfoodsincount[50])))
                """Edit Buttons END"""
                """Delete Buttons START"""
                if ind == 0:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[0])))
                elif ind == 1:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[1])))
                elif ind == 2:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[2])))
                elif ind == 3:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[3])))
                elif ind == 4:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[4])))
                elif ind == 5:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[5])))
                elif ind == 6:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[6])))
                elif ind == 7:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[7])))
                elif ind == 8:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[8])))
                elif ind == 9:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[9])))
                elif ind == 10:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[10])))
                elif ind == 11:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[11])))
                elif ind == 12:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[12])))
                elif ind == 13:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[13])))
                elif ind == 14:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[14])))
                elif ind == 15:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[15])))
                elif ind == 16:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[16])))
                elif ind == 17:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[17])))
                elif ind == 18:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[18])))
                elif ind == 19:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[19])))
                elif ind == 20:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[20])))
                elif ind == 21:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[21])))
                elif ind == 22:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[22])))
                elif ind == 23:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[23])))
                elif ind == 24:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[24])))
                elif ind == 25:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[25])))
                elif ind == 26:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[26])))
                elif ind == 27:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[27])))
                elif ind == 28:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[28])))
                elif ind == 29:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[29])))
                elif ind == 30:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[30])))
                elif ind == 31:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[31])))
                elif ind == 32:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[32])))
                elif ind == 33:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[33])))
                elif ind == 34:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[34])))
                elif ind == 35:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[35])))
                elif ind == 36:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[36])))
                elif ind == 37:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[37])))
                elif ind == 38:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[38])))
                elif ind == 39:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[39])))
                elif ind == 40:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[40])))
                elif ind == 41:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[41])))
                elif ind == 42:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[42])))
                elif ind == 43:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[43])))
                elif ind == 44:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[44])))
                elif ind == 45:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[45])))
                elif ind == 46:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[46])))
                elif ind == 47:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[47])))
                elif ind == 48:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[48])))
                elif ind == 49:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[49])))
                else:
                    self.b2[ind] = (Button(text="Delete", size_hint_y=0.1,
                                           on_press=lambda x: self.ids.curbfoods.remove_widget(self.box[50])))
                """Delete Buttons END"""
                self.gc[ind].add_widget(self.b[ind])
                self.gc[ind].add_widget(self.b2[ind])
                self.box[ind].add_widget(self.lb[ind])
                self.box[ind].add_widget(self.gc[ind])
                self.ids.curbfoods.add_widget(self.box[ind])
                lst.append([self.box[ind], self.lb[ind], self.gc[ind], self.b[ind], self.b2[ind]])
                print(lst[ind][0])
                print(self.ids.curbfoods.children)
                addfdnametotext(dayfordisplay.usedfoodsincount[ind].food_name, fdfile)
            fd += 1
        print(lst)


class FollowersPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.box = {}
        self.pic = {}
        self.lb = {}
        self.gc = {}
        self.b = {}
        self.b2 = {}

    def dstry(self):
        self.ids.curfllwrs.clear_widgets()

    def on_load__(self):
        i = 0
        follower = {}
        global user
        global con
        crr = con.cursor()
        query = sql.SQL(
            "SELECT followers FROM business_followers WHERE (business_name = \'" + user + "\')")
        print("Query: ")
        print(query)
        crr.execute(query)
        followerlst = fixalldatabasedonnewline(str(crr.fetchall()))
        u = 0
        while u < len(followerlst):
            follower[u] = followerlst[u]
            u += 1
        if 0 < len(followerlst) < 50:
            p = len(followerlst)
        elif len(followerlst) == 0:
            p = 0
            self.lb[0] = (Label(text="No Followers", size_hint_y=0.1))
            self.ids.curfllwrs.add_widget(self.lb[0])
        else:
            p = 50
        if p > 0:
            while i < p:
                ind = i
                self.box[ind] = (GridLayout(cols=3, rows=1, size_hint=(1, self.size_hint_min_y),
                                            padding=[10, 10, 10, 10]))
                self.pic[ind] = Image(source=self.getpicture(follower[i]))
                self.lb[ind] = (Label(text=follower[i], size_hint_y=0.1))
                self.gc[ind] = (GridLayout(cols=1, rows=2))
                """Edit Buttons START"""
                if ind == 0:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[0])))
                elif ind == 1:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[1])))
                elif ind == 2:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[2])))
                elif ind == 3:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[3])))
                elif ind == 4:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[4])))
                elif ind == 5:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[5])))
                elif ind == 6:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[6])))
                elif ind == 7:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[7])))
                elif ind == 8:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[8])))
                elif ind == 9:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[9])))
                elif ind == 10:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[10])))
                elif ind == 11:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[11])))
                elif ind == 12:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[12])))
                elif ind == 13:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[13])))
                elif ind == 14:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[14])))
                elif ind == 15:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[15])))
                elif ind == 16:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[16])))
                elif ind == 17:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[17])))
                elif ind == 18:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[18])))
                elif ind == 19:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[19])))
                elif ind == 20:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[20])))
                elif ind == 21:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[21])))
                elif ind == 22:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[22])))
                elif ind == 23:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[23])))
                elif ind == 24:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[24])))
                elif ind == 25:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[25])))
                elif ind == 26:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[26])))
                elif ind == 27:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[27])))
                elif ind == 28:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[28])))
                elif ind == 29:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[29])))
                elif ind == 30:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[30])))
                elif ind == 31:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[31])))
                elif ind == 32:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[32])))
                elif ind == 33:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[33])))
                elif ind == 34:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[34])))
                elif ind == 35:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[35])))
                elif ind == 36:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[36])))
                elif ind == 37:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[37])))
                elif ind == 38:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[38])))
                elif ind == 39:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[39])))
                elif ind == 40:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[40])))
                elif ind == 41:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[41])))
                elif ind == 42:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[42])))
                elif ind == 43:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[43])))
                elif ind == 44:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[44])))
                elif ind == 45:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[45])))
                elif ind == 46:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[46])))
                elif ind == 47:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[47])))
                elif ind == 48:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[48])))
                elif ind == 49:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[49])))
                else:
                    self.b[ind] = (Button(text="View Profile", size_hint_y=0.1,
                                          on_press=lambda x: self.viewprofile(follower[50])))
                """Edit Buttons END"""
                """Delete Buttons START"""
                if ind == 0:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[0])))
                elif ind == 1:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[1])))
                elif ind == 2:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[2])))
                elif ind == 3:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[3])))
                elif ind == 4:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[4])))
                elif ind == 5:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[5])))
                elif ind == 6:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[6])))
                elif ind == 7:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[7])))
                elif ind == 8:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[8])))
                elif ind == 9:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[9])))
                elif ind == 10:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[10])))
                elif ind == 11:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[11])))
                elif ind == 12:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[12])))
                elif ind == 13:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[13])))
                elif ind == 14:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[14])))
                elif ind == 15:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[15])))
                elif ind == 16:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[16])))
                elif ind == 17:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[17])))
                elif ind == 18:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[18])))
                elif ind == 19:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[19])))
                elif ind == 20:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[20])))
                elif ind == 21:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[21])))
                elif ind == 22:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[22])))
                elif ind == 23:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[23])))
                elif ind == 24:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[24])))
                elif ind == 25:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[25])))
                elif ind == 26:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[26])))
                elif ind == 27:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[27])))
                elif ind == 28:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[28])))
                elif ind == 29:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[29])))
                elif ind == 30:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[30])))
                elif ind == 31:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[31])))
                elif ind == 32:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[32])))
                elif ind == 33:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[33])))
                elif ind == 34:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[34])))
                elif ind == 35:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[35])))
                elif ind == 36:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[36])))
                elif ind == 37:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[37])))
                elif ind == 38:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[38])))
                elif ind == 39:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[39])))
                elif ind == 40:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[40])))
                elif ind == 41:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[41])))
                elif ind == 42:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[42])))
                elif ind == 43:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[43])))
                elif ind == 44:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[44])))
                elif ind == 45:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[45])))
                elif ind == 46:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[46])))
                elif ind == 47:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[47])))
                elif ind == 48:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[48])))
                elif ind == 49:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[49])))
                else:
                    self.b2[ind] = (Button(text="View on Map", size_hint_y=0.1,
                                           on_press=lambda x: self.mapify(follower[50])))
                """Delete Buttons END"""
                self.gc[ind].add_widget(self.b[ind])
                self.gc[ind].add_widget(self.b2[ind])
                self.box[ind].add_widget(self.pic[ind])
                self.box[ind].add_widget(self.lb[ind])
                self.box[ind].add_widget(self.gc[ind])
                self.ids.curfllwrs.add_widget(self.box[ind])

    def getpicture(self, fllwrname):
        global con
        cursr = con.cursor()
        query = sql.SQL(
            "SELECT profile_picture FROM client_settings WHERE (username = \'" + fllwrname + "\')")
        print("Query: ")
        print(query)
        cursr.execute(query)
        profile_pic_ref = str(cursr.fetchall()).replace(' ', '').replace('[', '').replace(']', '').replace('(',
                                                                                                           '').replace(
            ')', '').replace(',', '').replace('\'', '').replace('\\n', '\n')
        return profile_pic_ref

    def viewprofile(self, fllwrname):
        WindowManager.current = "ipfbv"

    def mapify(self, fllwrname):
        WindowManager.current = "ipmap"


class IndividualProfilePage(Screen):
    def checkpicchange(self):
        Clock.schedule_interval(self.changepicture, 0.5)

    def deschedulepicchange(self):
        Clock.unschedule(self.changepicture)

    def changepicture(self, *args):
        global con
        global user
        cr = con.cursor()
        picturechanger = defineifmissingpic()
        with open(picturechanger, 'r') as pcf:
            try:
                pcline = pcf.readlines()[0].strip()
                here = True
                pcf.close()
            except IndexError:
                here = False
        if here:
            with open(picturechanger, 'w') as pcc:
                pcc.write("")
                pcc.close()
            self.app.profile_image = pcline
            str_to_execute = \
                "UPDATE client_settings SET profile_picture = \'" + pcline + "\n" + "\' WHERE username = \'" + \
                user + "\'"
            print(str_to_execute)
            cr.execute(str_to_execute)
            cr.execute("COMMIT")

    def getpicture(self):
        global con
        global user
        cursr = con.cursor()
        query = sql.SQL(
            "SELECT profile_picture FROM client_settings WHERE (username = \'" + user + "\')")
        print("Query: ")
        print(query)
        cursr.execute(query)
        profile_pic_ref = str(cursr.fetchall()).replace(' ', '').replace('[', '').replace(']', '') \
            .replace('(', '').replace(')', '').replace(',', '').replace('\'', '').replace('\\n', '\n')
        if profile_pic_ref.find('None') != -1:
            self.app.profile_image = profile_pic_ref

    def getprnam(self):
        global user
        self.ids.profile_name.text = user

    def get_client_only_description(self):
        global con
        global user
        cursr = con.cursor()
        query = sql.SQL(
            "SELECT client_only_description FROM client_settings WHERE (username = \'" + user + "\')")
        print("Query: ")
        print(query)
        cursr.execute(query)
        codescr = str(cursr.fetchall()).replace(' ', '').replace('[', '').replace(']', '') \
            .replace('(', '').replace(')', '').replace(',', '').replace('\'', '').replace('\\n', '\n')
        if codescr.find('None') != -1:
            self.ids.i_profile_description.text = newlineadder(codescr, 0.7)

    def getfavoritedesig(self):
        global con
        global user
        cursr = con.cursor()
        query = sql.SQL(
            "SELECT client_only_favorites FROM client_settings WHERE (username = \'" + user + "\')")
        print("Query: ")
        print(query)
        cursr.execute(query)
        s = str(cursr.fetchall())
        print("s: s: s: ")
        print(s)
        if s.find('None') == -1:
            desigs = fixalldatabasedonnewline(s)
            for desig in desigs:
                ilw = IconLeftWidget(icon=desig)
                query = sql.SQL(
                    "SELECT icon_hint FROM icon_colors WHERE (icon_nme = \'" + desig + "\')")
                print("Query: ")
                print(query)
                cursr.execute(query)
                iconhint = str(cursr.fetchall()).replace('[', '').replace(']', '').replace('(', '').replace(')',
                                                                                                            '').replace(
                    ',', '').replace('\'', '').replace('\\n', '\n')
                oili = OneLineIconListItem(text=iconhint, bg_color=find_color(cursr, desig))
                oili.add_widget(ilw)
                self.ids.list_of_favorites.add_widget(oili)

    def setfavoritedesig(self):
        global con
        global user
        cr = con.cursor()
        picturechanger = defineifmissingclientfavorites()
        with open(picturechanger, 'r') as pcf:
            try:
                pcline = pcf.readlines()[0].strip()
                here = True
                pcf.close()
            except IndexError:
                here = False
        if here:
            with open(picturechanger, 'w') as pcc:
                pcc.write("")
                pcc.close()
        # TO-DO

    def set_client_only_description(self):
        global con
        global user
        cr = con.cursor()
        picturechanger = defineifmissingcod()
        with open(picturechanger, 'r') as pcf:
            try:
                pcline = pcf.readlines()[0].strip()
                here = True
                pcf.close()
            except IndexError:
                here = False
        if here:
            with open(picturechanger, 'w') as pcc:
                pcc.write("")
                pcc.close()
        # TO-DO

    def entre(self):
        self.getprnam()
        self.get_client_only_description()
        self.getfavoritedesig()


class LightVsDarkPage(Screen):
    def change_theme(self, newtheme):
        global con
        global user
        global theme
        theme = newtheme
        cur = con.cursor()
        str_to_execute = \
            "UPDATE client_settings SET bg_theme = \'" + theme + "\' WHERE username = \'" + \
            user + "\'"
        print(str_to_execute)
        cur.execute(str_to_execute)
        cur.execute("COMMIT")


class ColorThemePage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dd_item_dict = {}
        self.items_to_add = ['Teal', 'Red', 'Pink', 'Indigo', 'Blue', 'LightBlue', 'Lime', 'Yellow', 'Orange', 'Amber']
        self.menu = []

    def dropdown_(self):
        i = 0
        while i < len(self.items_to_add):
            if len(self.dd_item_dict) <= 10:
                self.dd_item_dict[i] = self.items_to_add[i]
            print(self.dd_item_dict[i])
            if i == 0:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[0],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[0]),
                })
            elif i == 1:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[1],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[1]),
                })
            elif i == 2:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[2],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[2]),
                })
            elif i == 3:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[3],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[3]),
                })
            elif i == 4:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[4],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[4]),
                })
            elif i == 5:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[5],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[5]),
                })
            elif i == 6:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[6],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[6]),
                })
            elif i == 7:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[7],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[7]),
                })
            elif i == 8:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[8],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[8]),
                })
            else:
                self.menu.append({
                    "viewclass": "OneLineListItem",
                    "text": self.dd_item_dict[9],
                    "on_release": lambda: self.menu_callback(instance=self.dd_item_dict[9]),
                })
            i += 1
        self.dropdown = MDDropdownMenu(caller=self.ids.dropdown_button, items=self.menu, width_mult=4)
        print("self.dropdown.items: ")
        print(self.dropdown.items)
        self.dropdown.open()

    def menu_callback(self, instance):
        print(instance)
        self.change_primary_p(new_primary_p=instance)
        self.ids.restart_to_see_changes_color.text = "Restart app to see changes."

    def change_primary_p(self, new_primary_p):
        global con
        global user
        global primary_p
        primary_p = new_primary_p
        cur = con.cursor()
        str_to_execute = \
            "UPDATE client_settings SET primary_p = \'" + new_primary_p + "\' WHERE username = \'" + \
            user + "\'"
        print(str_to_execute)
        cur.execute(str_to_execute)
        cur.execute("COMMIT")


class LocationSettingsIndividual(Screen):
    pass


class PersonalGoalsIndividual(Screen):
    pass


class PersonalInfoIndividual(Screen):
    pass


class BusinessProfilePage(Screen):
    pass


class BusinessSettings(Screen):
    pass


class BusinessPrivacyPolicy(Screen):
    pass


class WindowManager(ScreenManager):
    pass


class BEditFood(Screen):
    pass


class WeekSummary(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global loggedin
        global loggedoutb4
        loggedoutb4 = False
        loggedin = False
        Clock.schedule_interval(self.sched, 1.0)
        Clock.schedule_interval(self.reschedule, 5.0)

    def sched(self, *args):
        print("Check")
        global loggedin
        if loggedin:
            print("boolcheck")
            Clock.schedule_once(self._finish_init)
            Clock.unschedule(self.sched)

    def reschedule(self, *args):
        global loggedin
        global loggedoutb4
        if loggedin and loggedoutb4:
            Clock.schedule_once(self.changefeature('calories'))
            Clock.schedule_once(self._finish_init())
            loggedoutb4 = False

    def _finish_init(self, *args):
        cd = doprocess(getimgs())
        featr = getgraphsubject()
        self.samples = 7
        print("AHAH")
        self.ymaxcal = maxcal(cd, featr)
        print(self.ymaxcal)
        print("AHAH2222")
        self.graph = Graph(xmin=0, xmax=self.samples, ymin=0, ymax=self.ymaxcal, border_color=[0, 1, 1, 1],
                           tick_color=[0, 1, 1, 0.7], x_grid=True, y_grid=True, draw_border=True, x_grid_label=True,
                           y_grid_label=True, x_ticks_major=1, y_ticks_major=self.ymaxcal / 15)
        self.ids.graph.add_widget(self.graph)
        print("one")
        self.plot_x = np.linspace(0, 1, self.samples)
        self.plot_y = [getxda(cd, 6, featr), getxda(cd, 5, featr), getxda(cd, 4, featr), getxda(cd, 3, featr),
                       getxda(cd, 2, featr), getxda(cd, 1, featr), getxda(cd, 0, featr)]
        print("aone")
        self.plot = LinePlot(color=[1, 1, 0, 1], line_width=1.5)
        self.plot.points = [(x, self.plot_y[x]) for x in range(self.samples)]
        self.graph.add_plot(self.plot)
        filecleanser(running_id, cd)
        print("Done")

    # def update_plot(self):
    #     cd = doprocess(getimgs())
    #     featr = getgraphsubject()
    #     self.plot_y = [getxda(cd, 6, featr), getxda(cd, 5, featr), getxda(cd, 4, featr), getxda(cd, 3, featr),
    #                    getxda(cd, 2, featr), getxda(cd, 1, featr), getxda(cd, 0, featr)]
    #     self.plot.points = [(x, self.plot_y[x]) for x in range(self.samples)]

    def changefeature(self, featr):
        b = defineifmissinggraphsubject()
        with open(b, 'w') as wrt:
            wrt.write(featr)
            wrt.close()
        self.ids.graph.clear_widgets(children=None)


class dietfriendv1(MDApp):
    def __init__(self, **kwargs):
        self.individual_data = {
            'Theme': 'yin-yang',
            'Color': 'palette',
            'Location Settings': 'compass-rose',
            'Personal Goals': 'karate',
            'Personal Info': 'scale-balance',
            'Financial Info': 'credit-card-check-outline',
        }
        self.profile_image = 'blank-account.png'
        super().__init__(**kwargs)
        Clock.schedule_once(self._finish_init)

    def _finish_init(self, *args):
        global theme
        global primary_p

        self.theme_cls.theme_style = theme
        self.theme_cls.accent_palette = "Blue"
        self.theme_cls.primary_palette = primary_p

    def callback(self, instance):
        print(instance.icon)
        if instance.icon == 'yin-yang':
            self.root.current = "themeldpage"
        elif instance.icon == 'palette':
            self.root.current = "colorpage"
        else:
            pass

    # def build(self):
    #     return kv


def filecleanser(idx, cr):
    path = os.path.dirname(__file__)
    date = cr.date_time
    fdate = str(date)
    print("DATE")
    print(date)
    year = fdate[0:fdate.find("-")]
    oneyearago = int(year) - 1
    twoyearsago = int(year) - 2
    threeyearsago = int(year) - 3
    fouryearsago = int(year) - 4
    fiveyearsago = int(year) - 5
    oneyearago = str(oneyearago)
    twoyearsago = str(twoyearsago)
    threeyearsago = str(threeyearsago)
    fouryearsago = str(fouryearsago)
    fiveyearsago = str(fiveyearsago)
    files = [f for f in os.listdir(path) if (os.path.basename(f).find("_fl") != -1 or
                                             os.path.basename(f).find("_bfl") != -1 or
                                             os.path.basename(f).find("_hint") != -1 or
                                             os.path.basename(f).find("_cod") != -1 or
                                             os.path.basename(f).find("_client_favorites") != -1 or
                                             os.path.basename(f).find("_popup_desig") != -1 or
                                             os.path.basename(f).find("_profile_pic") != -1 or
                                             os.path.basename(f).find("_gs") != -1 or
                                             os.path.basename(f).find("_mi") != -1 or
                                             os.path.basename(f).find("_bmispec") != -1 or
                                             os.path.basename(f).find(getdate(cr, 8)) != -1 or
                                             os.path.basename(f).find(oneyearago) != -1 or
                                             os.path.basename(f).find(twoyearsago) != -1 or
                                             os.path.basename(f).find(threeyearsago) != -1 or
                                             os.path.basename(f).find(fouryearsago) != -1 or
                                             os.path.basename(f).find(fiveyearsago) != -1) and f.endswith(".txt")]
    for f in files:
        s = os.path.basename(f)
        print(s)
        x = os.path.join(path, s)
        print(x)
        os.remove(x)


def filecleanser_dayexcluded():
    path = os.path.dirname(__file__)
    files = [f for f in os.listdir(path) if (os.path.basename(f).find("_fl") != -1 or
                                             os.path.basename(f).find("_bfl") != -1 or
                                             os.path.basename(f).find("_hint") != -1 or
                                             os.path.basename(f).find("_cod") != -1 or
                                             os.path.basename(f).find("_client_favorites") != -1 or
                                             os.path.basename(f).find("_popup_desig") != -1 or
                                             os.path.basename(f).find("_profile_pic") != -1 or
                                             os.path.basename(f).find("_gs") != -1 or
                                             os.path.basename(f).find("_bmispec") != -1 or
                                             os.path.basename(f).find("_mi") != -1
                                             ) and f.endswith(".txt")]
    for f in files:
        s = os.path.basename(f)
        print(s)
        x = os.path.join(path, s)
        print(x)
        os.remove(x)


def themer_init():
    global con
    global theme
    global primary_p
    extcursor = con.cursor()
    previous_platorm_settings = defineifmissing_prev_insecure_settings()
    pastuser = previous_platorm_settings[1]
    if pastuser != '':
        if row_exists_theme(extcursor, pastuser):
            query = sql.SQL(
                "SELECT bg_theme FROM client_settings WHERE (username = \'" + pastuser + "\')")
            print("Query: ")
            print(query)
            extcursor.execute(query)
            theme_setting = str(extcursor.fetchall())
            print("theme_setting:")
            print(theme_setting)
            if theme_setting.find('Light') != -1:
                theme_setting = 'Light'
            elif theme_setting.find('Dark') != -1:
                theme_setting = 'Dark'
            else:
                theme_setting = 'Light'
            theme = theme_setting
        else:
            theme = "Light"
    else:
        theme = "Light"
    if pastuser != '':
        if row_exists_primary_p(extcursor, pastuser):
            query = sql.SQL(
                "SELECT primary_p FROM client_settings WHERE (username = \'" + pastuser + "\')")
            print("Query: ")
            print(query)
            extcursor.execute(query)
            primary_p_setting = str(extcursor.fetchall()).replace('[', '').replace(']', '').replace('(', '').replace(
                ')', '').replace(',', '').replace('\'', '')
            print("primary_p_setting:")
            print(primary_p_setting)
            primary_p = primary_p_setting
        else:
            primary_p = "Teal"
    else:
        primary_p = "Teal"


def close_past_connections():
    try:
        global con
        con.close()
    except:
        pass


def connection_init():
    global con
    con = psycopg2.connect(database="dietfriendcab", user='postgres',
                           password='dietfriendx$', host='127.0.0.1', port='5432')
    close_past_connections()
    con = psycopg2.connect(database="dietfriendcab", user='postgres',
                           password='dietfriendx$', host='127.0.0.1', port='5432')


# connect_to_database()
print("Check Path:")
print(os.path.dirname(__file__))
print(":End Check")
# print(os.path.dirname('DietFriend'))
# print(Path('DietFriend').cwd())
# print()
# print()

global theme
global primary_p
global con
global loggedin
global loggedoutb4
global user
global universal_list
global est_serving_size
global indforfdlstpopup
global cleared
# [num_params, table, col_name, set_to, condition_1_name, cond_1, condition_2_name, cond_2]
listforupdatedb = [[2, "dietfriend_client_food_data", "fulltextfile_t", "", "username", "", "date", ""],
                   [2, "dietfriend_client_food_data", "typefile_tx_ex_rx_", "", "username", "", "date", ""],
                   [2, "dietfriend_client_food_data", "esttextfile_e", "", "username", "", "date", ""],
                   [2, "dietfriend_client_food_data", "individualstextfile_r", "", "username", "", "date", ""],
                   [0, "", "", "", "", "", "", ""],
                   [2, "client_moreinfo_value_storage", "_im_twocalories", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twototal_fat", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twosaturated_fat", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twotrans_fat", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twocholesterol", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twosodium", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twototal_carb", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twofiber", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twototal_sugars", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twoadded_sugars", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twoprotein", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twocalcium", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twoiron", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twopotassium", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twovitamin_a", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twovitamin_b", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twovitamin_c", "", "username", "", "datetime", ""],
                   [2, "client_moreinfo_value_storage", "_im_twovitamin_d", "", "username", "", "datetime", ""]]
filecleanser_dayexcluded()
connection_init()
themer_init()
foodattrnamelst = ['serving', 'calories', 'total_fat', 'saturated_fat', 'trans_fat', 'cholesterol', 'sodium',
                   'total_carb', 'fiber', 'total_sugars', 'added_sugars', 'protein', 'calcium', 'iron', 'potassium',
                   'vitamin_a', 'vitamin_b', 'vitamin_c', 'vitamin_d']
running_id = getrandomid()
Config.set('kivy', 'keyboard_mode', 'systemandmulti')
kivy.require('1.9.1')
os.environ["KIVY_NO_CONSOLELOG"] = "1"
# kv = Builder.load_file('dietfriendv1.kv')
if __name__ == "__main__":
    dietfriendv1().run()

# To Fix:
# recognize '.' only when surrounded by a number on each side
# Assume calcium, iron, potassium, vitamin_a, vitamin_b, and vitamin_c in mg if not any unit;
# vitamin_d in micrograms if no unit
