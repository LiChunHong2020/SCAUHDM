import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.efficientdet import EfficientDetBackbone
from utils.utils import (decodebox, efficientdet_correct_boxes,
                         letterbox_image, non_max_suppression)

image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

def preprocess_input(image):
    image /= 255
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image

class EfficientDet(object):
    _defaults = {
        "model_path"    : 'model_data/Epoch92-Total_Loss0.0917-Val_Loss0.0749.pth',
        "classes_path"  : 'model_data/helmet_detection.txt',
        "phi"           : 2,
        "confidence"    : 0.01,
        "iou"           : 0.5,
        "cuda"          : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()
        

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    

    def generate(self):

        self.net = EfficientDetBackbone(len(self.class_names),self.phi).eval()

        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))


        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image):

        image       = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])

        crop_img    = np.array(letterbox_image(image, (image_sizes[self.phi], image_sizes[self.phi])))
        photo       = np.array(crop_img, dtype = np.float32)
        photo       = np.transpose(preprocess_input(photo), (2, 0, 1))

        with torch.no_grad():
            images  = torch.from_numpy(np.asarray([photo]))
            if self.cuda:
                images = images.cuda()

            _, regression, classification, anchors = self.net(images)
            
            regression = decodebox(regression, anchors, images)
            detection = torch.cat([regression, classification],axis=-1)
            batch_detections = non_max_suppression(detection, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)

            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image
                

            top_index = batch_detections[:,4] > self.confidence
            top_conf = batch_detections[top_index,4]
            top_label = np.array(batch_detections[top_index,-1], np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)


            boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // image_sizes[self.phi], 1)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image


    def get_FPS(self, image, test_interval):

        image = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])

        crop_img    = np.array(letterbox_image(image, (image_sizes[self.phi], image_sizes[self.phi])))
        photo       = np.array(crop_img, dtype = np.float32)
        photo       = np.transpose(preprocess_input(photo), (2, 0, 1))

        with torch.no_grad():
            images = torch.from_numpy(np.asarray([photo]))
            if self.cuda:
                images = images.cuda()
                
            _, regression, classification, anchors = self.net(images)
            
            regression  = decodebox(regression, anchors, images)
            detection   = torch.cat([regression,classification], axis=-1)
            batch_detections = non_max_suppression(detection, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:,4] > self.confidence
                top_conf = batch_detections[top_index,4]
                top_label = np.array(batch_detections[top_index,-1], np.int32)
                top_bboxes = np.array(batch_detections[top_index,:4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

                boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)
            except:
                pass 
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                _, regression, classification, anchors = self.net(images)
                
                regression = decodebox(regression, anchors, images)
                detection = torch.cat([regression,classification],axis=-1)
                batch_detections = non_max_suppression(detection, len(self.class_names),
                                                        conf_thres=self.confidence,
                                                        nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:,4] > self.confidence
                    top_conf = batch_detections[top_index,4]
                    top_label = np.array(batch_detections[top_index,-1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index,:4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

                    boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)
                except:
                    pass 

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
