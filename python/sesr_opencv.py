import cv2  
import numpy as np

IMG_PATH = '../data/lr_img.png' 
ONNX_MODEL = '../model/sesr_collapse_m5_x4.onnx'

if __name__ == '__main__':

    print('load model')
    sesr_net = cv2.dnn.readNetFromONNX(ONNX_MODEL)
    
    print('load image')
    img = cv2.imread(IMG_PATH)

    cv2.imshow('img', img)

    blob = cv2.dnn.blobFromImage(img)
    print(blob.shape)
    sesr_net.setInput(blob)
    result = sesr_net.forward()

    ress_img = cv2.dnn.imagesFromBlob(result)
    # print('ress_img shape:', ress_img)

    res_img = ress_img[0]

    res_img = np.clip(res_img, 0, 255)

    hr_img = np.uint8(res_img)
    cv2.imshow('hr_img', hr_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

