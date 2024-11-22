
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "rknn_api.h"

#include "retinaface.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"

/*-------------------------------------------
                  Functions
-------------------------------------------*/
int inference_header_det_model(cv::Mat orig_img, retinaface_result* result)
{
  // cv::Mat orig_img_rgb;
  // cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);

  rknn_context ctx = 0;
  int            ret;
  int            model_len = 0;
  unsigned char* model;
  
  const int num_class = 1;
  float header_det_threshold = 0.5;
  const char* model_path = "model/HeaderDet.rknn";

  rknn_app_context_t rknn_app_ctx;
  memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

  ret = init_retinanet_model(model_path, &rknn_app_ctx);
  if (ret != 0) {
    printf("init_retinanet_model fail! ret=%d model_path=%s\n", ret, model_path);
    return -1;
  }

  bool modelSwitch = true;
  ret = inference_retinanet_model(&rknn_app_ctx, orig_img, result, num_class);
  if (ret != 0) {
      printf("inference_retinanet_model fail! ret=%d\n", ret);
      modelSwitch = false;
  }
  cv::Mat orig_img_clone = orig_img.clone();
  if (modelSwitch) {
      bool enable_draw_image = true; //画图
      for (int i = 0; i < result->count; ++i) {
          if (enable_draw_image) {
              int rx = result->object[i].box.left;
              int ry = result->object[i].box.top;
              int rw = result->object[i].box.right - result->object[i].box.left;
              int rh = result->object[i].box.bottom - result->object[i].box.top;

              cv::Rect box(rx, ry, rw, rh);
              std::string text = "Header";
              std::string score_str = std::to_string(result->object[i].score);
              text += " " + score_str;
              cv::Scalar color(0, 255, 0);  // 绿色
              cv::rectangle(orig_img_clone, box, color, 2);
              cv::Point textOrg(box.x, box.y - 10);
              cv::putText(orig_img_clone, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
              
          }
          if (result->object[i].score < header_det_threshold) {
              continue;
          }
          printf("header @(%d %d %d %d) score=%f\n", result->object[i].box.left, result->object[i].box.top,
              result->object[i].box.right, result->object[i].box.bottom, result->object[i].score);
      }
      if (enable_draw_image) {
          cv::imwrite("result.png", orig_img_clone);
          std::cout << "Draw result on result.png is finished." << std::endl;
      }
    }
    ret = release_retinanet_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_retinanet_model fail! ret=%d\n", ret);
    }
    return 0;
}

