import argparse 
import time
import torch 
import numpy as np
import json
import sys

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

import os
a = os.listdir("flowers/train")
a.sort()

def process_img(img):
    im = Image.open(img)
    wd, ht = im.size
    picture_coords = [wd, ht]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if (max_element == 0):
        min_element = 1
    else:
        min_element = 0
    aspect_ratio=picture_coords[max_element]/picture_coords[min_element]
    new_picture_coords = [0,0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)   
    wd, ht = new_picture_coords
    left = (wd - 244)/2
    top = (ht - 244)/2
    right = (wd + 244)/2
    bottom = (ht + 244)/2
    im = im.crop((left, top, right, bottom))
    np_img = np.array(im)
    np_img = np_img.astype('float64')
    np_img = np_img / [255,255,255]
    np_img = (np_img - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_img = np_img.transpose((2, 0, 1))
    return np_img

def load_mdl():
    mdl_info = torch.load(args.mdl_checkpoint)
    mdl = mdl_info['model']
    mdl.classifier = mdl_info['classifier']
    mdl.load_state_dict(mdl_info['state_dict'])
    return mdl

def read_categories():
    if (args.category_names is not None):
        cat_file = args.category_names 
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None

def classify_img(img_path, topk=5):
    topk=int(topk)
    with torch.no_grad():
        img = process_img(img_path)
        img = torch.from_numpy(img)
        img.unsqueeze_(0)
        img = img.float()
        mdl = load_mdl()
        if (args.gpu):
           img = img.cuda()
           mdl = mdl.cuda()
        else:
            img = img.cpu()
            mdl = mdl.cpu()
        outputs = mdl(img)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        results = zip(probs,classes)
        return results
        
def display_prediction(results):
    cat_file = read_categories()
    i = 0
    for p, c in results:
        i = i + 1
        p = str(round(p,4) * 100.) + '%'
        if (cat_file):
            c = cat_file.get(str(a[c-1]),'None')
        else:
            c = ' class {}'.format(str(a[c-1]))
        print("{}.{} ({})".format(i, c,p))
    return None
    
def parse():
    parser = argparse.ArgumentParser(description='use a neural network to classify an image!')
    parser.add_argument('img_input', help='image file to classifiy (required)')
    parser.add_argument('mdl_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse() 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
    img_path = args.img_input
    prediction = classify_img(img_path,top_k)
    display_prediction(prediction)
    return prediction

main()