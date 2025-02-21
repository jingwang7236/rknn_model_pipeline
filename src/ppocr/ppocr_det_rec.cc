// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppocr_system.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"
#include "outer_model/model_func.hpp"

#define INDENT "    "
#define THRESHOLD 0.3                                       // pixel score threshold
#define BOX_THRESHOLD 0.6                            // box score threshold
#define USE_DILATION false                               // whether to do dilation, true or false
#define DB_SCORE_MODE "slow"                        // slow or fast. slow for polygon mask; fast for rectangle mask
#define DB_BOX_TYPE "poly"                                // poly or quad. poly for returning polygon box; quad for returning rectangle box
#define DB_UNCLIP_RATIO 1.5                          // unclip ratio for poly type

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif
ppocr_text_recog_array_result_t inference_ppocr_det_rec_model(ppocr_system_app_context *rknn_app_ctx, det_model_input input_data, bool enable_logger=false)
{

    ppocr_text_recog_array_result_t results;
    int ret;
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = input_data.width;
    src_image.height = input_data.height;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = input_data.width * input_data.height * input_data.channel;
    src_image.virt_addr = (unsigned char *)malloc(src_image.size);
    if (src_image.virt_addr == NULL) {
      printf("malloc buffer size:%d fail!\n", src_image.size);
      results.count = 0;
      return results;
    }
    memcpy(src_image.virt_addr, input_data.data, src_image.size);

    ppocr_det_postprocess_params params;
    params.threshold = THRESHOLD;
    params.box_threshold = BOX_THRESHOLD;
    params.use_dilate = USE_DILATION;
    params.db_score_mode = DB_SCORE_MODE;
    params.db_box_type = DB_BOX_TYPE;
    params.db_unclip_ratio = DB_UNCLIP_RATIO;
    const unsigned char blue[] = {0, 0, 255};

    // ret = inference_ppocr_system_model(rknn_app_ctx, &src_image, &params, &results);
    ret = inference_ppocr_system_model_new(rknn_app_ctx, &src_image, &params, &results);
    if (ret != 0) {
        printf("inference_ppocr_system_model fail! ret=%d\n", ret);
        if (src_image.virt_addr != NULL) {
            free(src_image.virt_addr);
        }
        results.count = 0;
        return results;
    }
    if (enable_logger){
        for (int i = 0; i < results.count; i++)
        {
            printf("[%d] @ [(%d, %d), (%d, %d), (%d, %d), (%d, %d)]\n", i,
                results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 
                results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y);
            printf("regconize result: %s, score=%f\n", results.text_result[i].text.str, results.text_result[i].text.score);
        }
    }
    bool enable_draw_image = false; //画图,本地测试
    if (enable_draw_image) {
        // Draw Objects
        printf("DRAWING OBJECT\n");
        for (int i = 0; i < results.count; i++)
        {
            printf("[%d] @ [(%d, %d), (%d, %d), (%d, %d), (%d, %d)]\n", i,
                results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 
                results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y);
            //draw Quadrangle box
            draw_line(&src_image, results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 255, 2);
            draw_line(&src_image, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, 255, 2);
            draw_line(&src_image, results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y, 255, 2);
            draw_line(&src_image, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y, results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, 255, 2);
            printf("regconize result: %s, score=%f\n", results.text_result[i].text.str, results.text_result[i].text.score);
        }
        printf("    SAVE TO ./out.jpg\n");
        write_image("./out.jpg", &src_image);
    }

    if (src_image.virt_addr != NULL) {
        free(src_image.virt_addr);
    }

    return results;
}

#ifdef __cplusplus
}
#endif