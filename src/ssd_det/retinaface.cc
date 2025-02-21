#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "rknn_box_priors.h"

#include "inter_model/retinanet.hpp"

#define NMS_THRESHOLD 0.4
#define CONF_THRESHOLD 0.2
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

static int filterValidResult(float *scores, float *loc, float *landms, const float boxPriors[][4], int model_in_h, int model_in_w,
                             int filter_indice[], float *props, float threshold, const int num_results) {
    int validCount = 0;
    const float VARIANCES[2] = {0.1, 0.2};
    // Scale them back to the input size.
    for (int i = 0; i < num_results; ++i) {
        float face_score = scores[i * 2 + 1];
        if (face_score > threshold) {
            filter_indice[validCount] = i;
            props[validCount] = face_score;
            // printf("%d %f box:(%f %f %f %f)\n", i, face_score, loc[i * 4 + 0], loc[i * 4 + 1], loc[i * 4 + 2], loc[i * 4 + 3]);
            //decode location to origin position
            float xcenter = loc[i * 4 + 0] * VARIANCES[0] * boxPriors[i][2] + boxPriors[i][0];
            float ycenter = loc[i * 4 + 1] * VARIANCES[0] * boxPriors[i][3] + boxPriors[i][1];
            float w = (float) expf(loc[i * 4 + 2] * VARIANCES[1] ) * boxPriors[i][2];
            float h = (float) expf(loc[i * 4 + 3] * VARIANCES[1]) * boxPriors[i][3];

            float xmin = xcenter - w * 0.5f;
            float ymin = ycenter - h * 0.5f;
            float xmax = xmin + w;
            float ymax = ymin + h;

            loc[i * 4 + 0] = xmin ;
            loc[i * 4 + 1] = ymin ;
            loc[i * 4 + 2] = xmax ;
            loc[i * 4 + 3] = ymax ;
            for (int j = 0; j < 5; ++j) {
                landms[i * 10 + 2 * j] = landms[i * 10 + 2 * j] * VARIANCES[0] * boxPriors[i][2] + boxPriors[i][0];
                landms[i * 10 + 2 * j + 1] = landms[i * 10 + 2 * j + 1] * VARIANCES[0] * boxPriors[i][3] + boxPriors[i][1];
            }
            ++validCount;
        }
    }

    return validCount;
}

static int post_process_retinaface(rknn_app_context_t *app_ctx, rknn_output outputs[], retinaface_result *result, letterbox_t *letter_box) {
    float *location = (float *)outputs[0].buf;
    float *scores = (float *)outputs[1].buf;
    float *landms = (float *)outputs[2].buf;
    const float (*prior_ptr)[4];
    int num_priors = 0;
    if (app_ctx->model_height == 320) {
        num_priors = 4200;//anchors box number
        prior_ptr = BOX_PRIORS_320;
    } else if(app_ctx->model_height == 640){
        num_priors = 16800;//anchors box number
        prior_ptr = BOX_PRIORS_640;
    }
    else
    {
        printf("model_shape error!!!\n");
        return -1;

    }

    int filter_indices[num_priors];
    float props[num_priors];

    memset(filter_indices, 0, sizeof(int)*num_priors);
    memset(props, 0, sizeof(float)*num_priors);

    int validCount = filterValidResult(scores, location, landms, prior_ptr, app_ctx->model_height, app_ctx->model_width,
                                       filter_indices, props, CONF_THRESHOLD, num_priors);

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
        float x1 = location[n * 4 + 0] * app_ctx->model_width - letter_box->x_pad;
        float y1 = location[n * 4 + 1] * app_ctx->model_height - letter_box->y_pad;
        float x2 = location[n * 4 + 2] * app_ctx->model_width - letter_box->x_pad;
        float y2 = location[n * 4 + 3] * app_ctx->model_height - letter_box->y_pad;
        int model_in_w = app_ctx->model_width;
        int model_in_h = app_ctx->model_height;
        result->object[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / letter_box->scale); // Face box
        result->object[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        result->object[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        result->object[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        result->object[last_count].score = props[i]; // Confidence
        for (int j = 0; j < 5; ++j) { // Facial feature points
            float ponit_x = landms[n * 10 + 2 * j] * app_ctx->model_width - letter_box->x_pad;
            float ponit_y = landms[n * 10 + 2 * j + 1] * app_ctx->model_height - letter_box->y_pad;
            result->object[last_count].ponit[j].x = (int)(clamp(ponit_x, 0, model_in_w) / letter_box->scale);
            result->object[last_count].ponit[j].y = (int)(clamp(ponit_y, 0, model_in_h) / letter_box->scale);
        }
        last_count++;
    }

    result->count = last_count;

    return 0;
}

int init_retinaface_model(const char *model_path, rknn_app_context_t *app_ctx) {
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
    // printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    // printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(output_attrs[i]));
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

int release_retinaface_model(rknn_app_context_t *app_ctx) {
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

static int pre_process_retinaface(rknn_app_context_t *app_ctx, const cv::Mat& orig_img, cv::Mat& dist_img, letterbox_t *letter_box)
{
    // 获取原始图像的宽度和高度
    int originalWidth = orig_img.cols;
    int originalHeight = orig_img.rows;
    // 获取目标图像的宽度和高度---正方形
    int targetWidth = app_ctx->model_width;
    int targetHeight =  app_ctx->model_height;
    // 计算缩放比例
    float widthScale = (float)targetWidth / originalWidth;
    float heightScale = (float)targetHeight / originalHeight;
    float scale = std::min(widthScale, heightScale);
    int newWidth = (int)(originalWidth * scale);
    int newHeight = (int)(originalHeight * scale);
    cv::Mat resize_img;
    cv::resize(orig_img, resize_img, cv::Size(newWidth, newHeight));
    // 计算需要填充的像素,左上角开始贴图
    int left_pad = 0;
    int right_pad = targetWidth - newWidth - left_pad;
    int top_pad = 0;
    int bottom_pad = targetHeight - newHeight - top_pad;
    cv::copyMakeBorder(resize_img, dist_img, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    letter_box->scale = scale;
    letter_box->x_pad = left_pad;
    letter_box->y_pad = top_pad;
    return 0;
}

int inference_retinaface_model(rknn_app_context_t *app_ctx, cv::Mat src_img, retinaface_result *out_result) {
    int ret;
    image_buffer_t img;
    letterbox_t letter_box;
    rknn_input inputs[1];
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(rknn_output) * 3);
    memset(&letter_box, 0, sizeof(letterbox_t));

    cv::Mat dist_img;
    ret = pre_process_retinaface(app_ctx, src_img, dist_img, &letter_box);
    if (ret < 0) {
        printf("pre_process_retinaface fail! ret=%d\n", ret);
        return -1;
    }
    // cv::imwrite("tmp.png", dist_img);
    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = dist_img.cols * dist_img.rows * dist_img.channels() * sizeof(uint8_t);
    inputs[0].buf   = dist_img.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
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
        outputs[i].is_prealloc = 0;
        outputs[i].buf = (void*)malloc(outputs[i].size);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 3, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    ret = post_process_retinaface(app_ctx, outputs, out_result, &letter_box);
    if (ret < 0) {
        printf("post_process_retinaface fail! ret=%d\n", ret);
        return -1;
    }
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, 3, outputs);

out:
    if (img.virt_addr != NULL) {
        free(img.virt_addr);
    }

    return ret;
}
