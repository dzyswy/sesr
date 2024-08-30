#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
 
using namespace cv;
using namespace cv::dnn;
using namespace std;
 
 
int main( int argc, char * *argv)
{

    if (argc != 3) {
        printf("usage: %s *.png *.onnx\n", argv[0]);
        printf("usage: %s ../../data/lr_img.png ../../model/sesr_collapse_m5_x4.onnx\n", argv[0]);
        return -1;
    }

    std::string img_name = argv[1];
    std::string onnx_model = argv[2];
 
    float scale = 1.0;
    //cv::Scalar mean {103.939, 116.779, 123.68 };
    cv::Scalar mean {0, 0, 0};
    bool swapRB = false;
    bool crop = false;
    bool useOpenCL = false;
 
    cv::Mat img = cv::imread(img_name);
    if (img.empty()) {
        cout << "Can't read image from file: " << img_name << endl;
        return 2;
    }

    cv::Mat lr_img;
    cv::resize(img, lr_img, cv::Size(64, 64), 0, 0, cv::INTER_CUBIC);
 
    // Load model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_model);
    if (useOpenCL)
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
 
    // Create a 4D blob from a frame.
    cv::Mat inputBlob = cv::dnn::blobFromImage(lr_img, scale, lr_img.size(), mean, swapRB, crop); 
 
    // forward netword

    net.setInput(inputBlob);
    cv::Mat output = net.forward();
 
    // process output
    // Mat(output.size[ 2], output.size[ 3], CV_32F, output.ptr < float >( 0, 0)) += 103. 939;
    // Mat(output.size[ 2], output.size[ 3], CV_32F, output.ptr < float >( 0, 1)) += 116. 779;
    // Mat(output.size[ 2], output.size[ 3], CV_32F, output.ptr < float >( 0, 2)) += 123. 68;
 
    std::vector<cv::Mat> ress;
    cv::dnn::imagesFromBlob(output, ress);
 
    // show res
    cv::Mat res;
    //cv::normalize(ress[0], res, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    ress[0].convertTo(res, CV_8UC3);
    cv::imshow("reslut", res);
    cv::imwrite("reslut.png", res);
 
    cv::imshow("lr_img", lr_img);
    cv::imwrite("lr_img.png", lr_img);
 
    cv::waitKey();
    return 0;
}