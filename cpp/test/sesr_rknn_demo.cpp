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
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"


#define MAX_TEXT_LINE_LENGTH 1024



typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} rknn_app_context_t;



static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


int init_rknn_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    int ret;
    int model_len = 0;
    char* model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_rknn_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}


// int inference_rknn_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, resnet_result* out_result, int topk)
// {
//     int ret;
//     image_buffer_t img;
//     rknn_input inputs[1];
//     rknn_output outputs[1];

//     //defualt initialized
//     memset(&img, 0, sizeof(image_buffer_t));
//     memset(inputs, 0, sizeof(inputs));
//     memset(outputs, 0, sizeof(outputs));

//     // Pre Process
//     img.width = app_ctx->model_width;
//     img.height = app_ctx->model_height;
//     img.format = IMAGE_FORMAT_RGB888;
//     img.size = img.width * img.height * 3;
//     img.virt_addr = (unsigned char*)malloc(img.size);
//     if (img.virt_addr == NULL) {
//         printf("malloc buffer size:%d fail!\n", img.size);
//         return -1;
//     }

//     //caution: might have bug!!
//     ret = convert_image(src_img, &img, NULL, NULL, 0);
//     if (ret < 0) {
//         printf("convert_image fail! ret=%d\n", ret);
//         return -1;
//     }

//     // Set Input Data
//     inputs[0].index = 0;
//     inputs[0].type  = RKNN_TENSOR_UINT8;
//     inputs[0].fmt   = RKNN_TENSOR_NHWC;
//     inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
//     inputs[0].buf   = img.virt_addr;

//     ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
//     if (ret < 0) {
//         printf("rknn_input_set fail! ret=%d\n", ret);
//         return -1;
//     }

//     // Run
//     printf("rknn_run\n");
//     ret = rknn_run(app_ctx->rknn_ctx, nullptr);
//     if (ret < 0) {
//         printf("rknn_run fail! ret=%d\n", ret);
//         return -1;
//     }

//     // Get Output
//     outputs[0].want_float = 1;
//     ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
//     if (ret < 0) {
//         printf("rknn_outputs_get fail! ret=%d\n", ret);
//         goto out;
//     }

//     // Post Process
//     softmax((float*)outputs[0].buf, app_ctx->output_attrs[0].n_elems);

//     get_topk_with_indices((float*)outputs[0].buf, app_ctx->output_attrs[0].n_elems, topk, out_result);

//     // Remeber to release rknn output
//     rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);

// out:
//     if (img.virt_addr != NULL) {
//         free(img.virt_addr);
//     }

//     return ret;
// }





int main(int argc, char* argv[])
{
    int ret;
    if (argc != 3) {
        printf("usage: %s *.png *.onnx\n", argv[0]);
        printf("usage: %s ../../data/lr_img.png ../../model/sesr_collapse_m5_x4.rknn\n", argv[0]);
        return -1;
    }

    std::string img_name = argv[1];
    std::string rknn_model = argv[2];

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(img_name.c_str(), &src_image);
    if (ret != 0) {
        printf("read image fail! ret=%d image_path=%s\n", ret, img_name.c_str());
        return -1;
    }


    
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_rknn_model(rknn_model.c_str(), &rknn_app_ctx);
    if (ret != 0) {
        printf("init_mobilenet_model fail! ret=%d model_path=%s\n", ret, rknn_model.c_str());
        return -1;
    }

    return 0;
}






