import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = '../model/sesr_collapse_m5_x4.onnx'
RKNN_MODEL = '../model/sesr_collapse_m5_x4.rknn'
IMG_PATH = '../data/lr_img.png'
DATASET = './coco_subset_20.txt'

QUANTIZE_ON = True






if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
 

    # Inference
    print('--> Running model')
    img2 = np.expand_dims(img, 0)
    outputs = rknn.inference(inputs=[img2], data_format=['nhwc']) #nhwc

    rres_img = outputs[0]
    print(rres_img.shape)
    rres_img=rres_img.transpose(0, 2, 3, 1)
    print(rres_img.shape)
    
    # print(outputs)
    # print(outputs[0][0][0])
    print('rres_img[0]',len(rres_img[0]))
    print('rres_img[0][0]',len(rres_img[0][0]))
    print('rres_img[0][1]',len(rres_img[0][1]))
    print('rres_img[0][2]',len(rres_img[0][2]))
    print('done')

    res_img = rres_img[0]
    res_img = res_img * 255
    res_img = np.clip(res_img, 0, 255)
    
    print('res_img', res_img)
    print('res_img len:',len(res_img))
    print('res_img[0] len:',len(res_img[0]))
    print('res_img[1] len:',len(res_img[1]))
    print('res_img[2] len:',len(res_img[2]))

    hr_img = np.uint8(res_img)
    cv2.imshow('hr_img', hr_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

 

    rknn.release()
