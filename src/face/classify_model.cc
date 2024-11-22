#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "classify_model.h"

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int post_process_classify(rknn_app_context_t *app_ctx, rknn_output outputs[], classify_result *result) {
    result->size = app_ctx->io_num.n_output;
    // 定义每个属性的输出长度
    int attr_num[result->size];
    for (int i = 0; i < result->size; i++) {
        attr_num[i] = 2;  // 每个属性都是二分类
    }
    float scoreArray[result->size];
    int clsArray[result->size];
    for (int i = 0; i < result->size; i++) {
        float *output_data = (float *)outputs[i].buf;
        int max_index = -1;
        float max_value = -0.1;
        for (int j = 0; j < attr_num[i]; j++) {
        // 判断output_data[j]中的j是否越界
            // printf("output_data[%d]: %f\n", j, output_data[j]);
            if (output_data[j] > max_value) {
                max_value = output_data[j];
                max_index = j;
            }
        }
        scoreArray[i] = max_value;
        clsArray[i] = max_index;
        printf("attr_id: %d, clsArray: %d scoreArray: %f\n", i, clsArray[i], scoreArray[i]);
    }
    result->score = scoreArray;
    result->cls = clsArray;
    return 0;
}

int init_classify_model(const char *model_path, rknn_app_context_t *app_ctx) {
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

int release_classify_model(rknn_app_context_t *app_ctx) {
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

int inference_classify_model(rknn_app_context_t *app_ctx, image_buffer_t *src_img, classify_result *out_result) {
    int ret;
    image_buffer_t img;
    letterbox_t letter_box;
    rknn_input inputs[1];
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(&img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));
    memset(&letter_box, 0, sizeof(letterbox_t));
    int bg_color = 114;//letterbox background pixel

    // Pre Process
    img.width = app_ctx->model_width;
    img.height = app_ctx->model_height;
    // img.format = IMAGE_FORMAT_RGB888;
    img.format = src_img->format;
    img.size = get_image_size(&img);
    img.virt_addr = (unsigned char *)malloc(img.size);

    if (img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", img.size);
        return -1;
    }
    // pre process
    ret = convert_image_with_letterbox(src_img, &img, &letter_box, bg_color);
    if (ret < 0) {
        printf("convert_image fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf   = img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
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
        // goto out;
        if (img.virt_addr != NULL) {
            free(img.virt_addr);
        }
        return ret;
    }
    // post process
    ret = post_process_classify(app_ctx, outputs, out_result);
    if (ret < 0) {
        printf("post_process_classify fail! ret=%d\n", ret);
        return -1;
    }
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, 3, outputs);

    if (img.virt_addr != NULL) {
        free(img.virt_addr);
    }
    return ret;
}



