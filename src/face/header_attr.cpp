// backbone:repvgg, 3.2M params, person attribute,include: hat/hellem, glass, mask


/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>

// using namespace std;
// using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int            model_len = ftell(fp);
  unsigned char* model     = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp) {
    fclose(fp);
  }
  return model;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int inference_header_attr_model(cv::Mat orig_img, int* result)
{
  const int MODEL_IN_WIDTH    = 64;
  const int MODEL_IN_HEIGHT   = 64;
  const int MODEL_IN_CHANNELS = 3;

  rknn_context ctx = 0;
  int            ret;
  int            model_len = 0;
  unsigned char* model;

  const char* model_path = "model/FaceAttr.rknn";

  // Load image
  // cv::Mat orig_img = cv::imread(img_path, cv::IMREAD_COLOR);
  // 检查图像是否成功读取
  // if (orig_img.empty()) {
  //     std::cerr << "Failed to read image: " << img_path << std::endl;

  //     // 尝试获取更详细的错误信息
  //     try {
  //         // 重新尝试读取图像，捕获可能的异常
  //         orig_img = cv::imread(img_path, cv::IMREAD_COLOR);
  //     } catch (const cv::Exception& e) {
  //         std::cerr << "OpenCV Exception: " << e.what() << std::endl;
  //     } catch (const std::exception& e) {
  //         std::cerr << "Standard Exception: " << e.what() << std::endl;
  //     } catch (...) {
  //         std::cerr << "Unknown Exception" << std::endl;
  //     }
  // }

  // cv::Mat orig_img_rgb;
  // cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);

  cv::Mat img;
  if (orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
    // printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
    cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
  }
  // std::string image_path = "face_tmp.png";
  // bool success = cv::imwrite(image_path.c_str(), img);
  // std::cout << "保存截图" << image_path.c_str() << "成功: " << success << std::endl;

  // Load RKNN Model
  model = load_model(model_path, &model_len);
  ret   = rknn_init(&ctx, model, model_len, 0, NULL);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

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

  // Set Input Data
  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type  = RKNN_TENSOR_UINT8;
  inputs[0].size  = img.cols * img.rows * img.channels() * sizeof(uint8_t);
  inputs[0].fmt   = RKNN_TENSOR_NHWC;
  inputs[0].buf   = img.data;

  ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return -1;
  }

  // Run
  // printf("rknn_run\n");
  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  // Get Output + Post Process
  rknn_output outputs[io_num.n_output];  // 获取三个输出
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++) {
    outputs[i].index = i;
    outputs[i].want_float = 1; // 设置输出偏好为浮点数
  }
  // 定义每个属性的输出长度
  int attr_num[io_num.n_output];
  // for (int i = 0; i < io_num.n_output; i++) {
  //   attr_num[i] = 2;  // 每个属性都是二分类
  // }
  attr_num[0] = 3;
  attr_num[1] = 2;
  attr_num[2] = 2;
  
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
    return -1;
  }
    
  for (int i = 0; i < io_num.n_output; i++) {
    float *output_data = (float *)outputs[i].buf;
    int max_index = 0;
    float max_value = 0.0;
    for (int j = 0; j < attr_num[i]; j++) {
      // printf("output[%d][%d]: %f\n", i, j, output_data[j]);
      if (output_data[j] > max_value) {
        max_value = output_data[j];
        max_index = j;
      }
      // printf("max_value: %f max_index: %d\n", max_value, max_index);
    }
    // std::cout << "Output[" << i << "]: " << max_index << " " << max_value << std::endl;
    // 保存max_index和max_value到 一个全局变量中，并返回
    result[i] = max_index;
  }
  // Release rknn_outputs
  rknn_outputs_release(ctx, 1, outputs);

  // Release
  if (ctx > 0)
  {
    rknn_destroy(ctx);
  }
  if (model) {
    free(model);
  }
  return 0;
}

