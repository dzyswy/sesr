#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "rknn_api.h"




int main(int argc, char* argv[])
{
    if (argc != 3) {
        printf("usage: %s *.png *.onnx\n", argv[0]);
        printf("usage: %s ../../data/lr_img.png ../../model/sesr_collapse_m5_x4.rknn\n", argv[0]);
        return -1;
    }

    std::string img_name = argv[1];
    std::string rknn_model = argv[2];

    rknn_context ctx = 0;

    return 0;
}






