// 该代码适合模型中已经包含了后处理操作，代码中只负责nms后处理算法

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

#include "retinaface.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "rknn_box_priors.h"

#include "opencv2/opencv.hpp"

#define NMS_THRESHOLD 0.5
#define CONF_THRESHOLD 0.5
#define VIS_THRESHOLD 0.1

static int clamp(int x, int min, int max) {
    if (x > max) return max;
    if (x < min) return min;
    return x;
}

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1) * (ymax0 - ymin0 + 1) + (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, float *outputLocations, int order[], float threshold) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1) {
                continue;
            }
            
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 3];
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            // printf("iou:%f\n", iou);
            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(float *input, int left, int right, int *indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}


static int filterValidResult(float *scores, int filter_indice[], float *props, 
                            float threshold, const int num_results,const int num_class) {
    int validCount = 0;
    for (int i = 0; i < num_results; ++i) {
        // float face_score = scores[i * 7];
        // printf("%d num_class \n", num_class);
        float face_score = scores[i * num_class];
        if (face_score > threshold) {
            filter_indice[validCount] = i;
            props[validCount] = face_score;
            ++validCount;
        }
    }

    return validCount;
}

static int post_process_retinanet(rknn_app_context_t *app_ctx, rknn_output outputs[], retinaface_result *result, letterbox_four *letter_box, const int num_class) 
{
    float *scores = (float *)outputs[0].buf; // [1, 46440, 7, 0]
    float *location = (float *)outputs[1].buf; // [1, 46440, 4, 0]

    // int scores_size = outputs[0].size / sizeof(float); // 46440*7
    // int location_size = outputs[1].size / sizeof(float); // 46440*4
    int num_priors = 46440; // 5层FPN的H*W,决定了anchor数量
    int filter_indices[num_priors];
    float props[num_priors];

    memset(filter_indices, 0, sizeof(int)*num_priors);
    memset(props, 0, sizeof(float)*num_priors);

    // filter valid result
    int validCount = filterValidResult(scores, filter_indices, props, CONF_THRESHOLD, num_priors, num_class);
    printf("%d valid \n", validCount);
    printf("filterValidResult Done:");
    for (int i=0; i<validCount; i++){
        int n = filter_indices[i];
        float x1 = location[n * 4 + 0];
        float y1 = location[n * 4 + 1];
        float x2 = location[n * 4 + 2];
        float y2 = location[n * 4 + 3];
        printf("before nms idx: %d, score: %f:, box: %f, %f, %f, %f \n ", i, scores[n*num_class] ,x1, y1, x2, y2);
    }
    quick_sort_indice_inverse(props, 0, validCount - 1, filter_indices);
    nms(validCount, location, filter_indices, NMS_THRESHOLD);

    int last_count = 0;
    result->count = 0;
    for (int i = 0; i < validCount; ++i) {
        if (last_count >= 128) {
            printf("Warning: detected more than 128 faces, can not handle that");
            break;
        }
        if (filter_indices[i] == -1 || props[i] < VIS_THRESHOLD) {
            continue;
        }

        int n = filter_indices[i];
        float x1 = location[n * 4 + 0] - letter_box->left_pad;
        float y1 = location[n * 4 + 1] - letter_box->top_pad;
        float x2 = location[n * 4 + 2] - letter_box->left_pad;
        float y2 = location[n * 4 + 3] - letter_box->top_pad;
        int model_in_w = app_ctx->model_width;
        int model_in_h = app_ctx->model_height;
        result->object[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->x_scale);
        result->object[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->y_scale);
        result->object[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->x_scale);
        result->object[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->y_scale);
        result->object[last_count].score = props[i]; // Confidence
        last_count++;
    }
    result->count = last_count;

    return 0;
}


static int pre_process_retinanet(rknn_app_context_t *app_ctx, const cv::Mat& orig_img, cv::Mat& dist_img, letterbox_four *letter_box, bool scale_image)
{
    // 获取原始图像的宽度和高度
    int originalWidth = orig_img.cols;
    int originalHeight = orig_img.rows;
    // 获取目标图像的宽度和高度
    int targetWidth = app_ctx->model_width;
    int targetHeight =  app_ctx->model_height;
    // 计算缩放比例
    float widthScale = (float)targetWidth / originalWidth;
    float heightScale = (float)targetHeight / originalHeight;
    if (scale_image)
    {
        float scale = (widthScale < heightScale) ? widthScale : heightScale;

        // 计算新的宽度和高度
        int newWidth = (int)(originalWidth * scale);
        int newHeight = (int)(originalHeight * scale);
        
        cv::Mat resize_img;
        cv::resize(orig_img, resize_img, cv::Size(newWidth, newHeight));
        // 计算需要填充的像素
        int left_pad = (targetWidth - newWidth) / 2;  // 居中贴图
        int right_pad = targetWidth - newWidth - left_pad;
        int top_pad = (targetHeight - newHeight) / 2;
        int bottom_pad = targetHeight - newHeight - top_pad;

        // 填充图像
        cv::copyMakeBorder(resize_img, dist_img, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        
        cv::imwrite("input01.png", dist_img);

        letter_box->x_scale = scale;
        letter_box->y_scale = scale;
        letter_box->left_pad = left_pad;
        letter_box->right_pad = right_pad;
        letter_box->top_pad = top_pad;
        letter_box->bottom_pad = bottom_pad;
        
        printf("scale: %f, left_pad: %d, right_pad: %d, top_pad: %d, bottom_pad: %d\n",
         scale, left_pad, right_pad, top_pad, bottom_pad);
    }else
    {
        cv::resize(orig_img, dist_img, cv::Size(targetWidth, targetHeight));
        letter_box->x_scale = widthScale;
        letter_box->y_scale = heightScale;
        letter_box->left_pad = 0;
        letter_box->right_pad = 0;
        letter_box->top_pad = 0;
        letter_box->bottom_pad = 0;
        cv::imwrite("input02.png", dist_img);
        printf("x_scale: %f, y_scale: %f, pad: %d\n",
         widthScale, heightScale, letter_box->left_pad);
    }
    
    return 0;
}


int init_retinanet_model(const char *model_path, rknn_app_context_t *app_ctx) {
    int ret;
    int model_len = 0;
    char *model;
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
    // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

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
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
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

int release_retinanet_model(rknn_app_context_t *app_ctx) {
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

int inference_retinanet_model(rknn_app_context_t *app_ctx, cv::Mat src_img, retinaface_result *out_result, const int num_class) {
    int ret;
    letterbox_four letter_box;  //     int x_pad;int y_pad;float scale; image resize params
    rknn_input inputs[1];
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(rknn_output));
    memset(&letter_box, 0, sizeof(letterbox_t));

    // Pre Process
    bool scale_image = false; // resize image to model input size, if true, scale image based on letterbox
    cv::Mat dist_img;
    ret = pre_process_retinanet(app_ctx, src_img, dist_img, &letter_box, scale_image);
    if (ret < 0) {
        printf("pre_process_retinanet fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = dist_img.cols * dist_img.rows * dist_img.channels() * sizeof(uint8_t);
    inputs[0].buf   = dist_img.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ret;
    }

    ret = post_process_retinanet(app_ctx, outputs, out_result, &letter_box, num_class);
    if (ret < 0) {
        printf("post_process_retinaface fail! ret=%d\n", ret);
        return -1;
    }
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    return 0;
}
