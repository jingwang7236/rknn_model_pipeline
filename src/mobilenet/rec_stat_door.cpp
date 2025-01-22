// mobilenet recognition

/*-------------------------------------------
				Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "image_utils.h"
#include "file_utils.h"
#include "opencv2/opencv.hpp"
#include "mobilenet.h"
#include "outer_model/model_func.hpp"
#include "stb_image_resize2.h"
#define READ_IMAGE_TYPE STBIR_RGB


extern std::map<int, std::string> cls_stat_door_category_map_mobilenet = { {0,"closed"},{1,"open"},{2,"other"} };

/*-------------------------------------------
				  Main Function
-------------------------------------------*/

mobilenet_result inference_rec_stat_door_mobilenetv3_model(rknn_app_context_t* app_ctx, det_model_input input_data, cls_model_inference_params cls_stat_door, bool enable_logger)
{
	mobilenet_result od_results;
	memset(&od_results, 0, sizeof(resnet_result));

	// 分配内存用于存储调整大小后的图像
	unsigned char* resized_data = (unsigned char*)malloc(cls_stat_door.img_width * cls_stat_door.img_height * input_data.channel);

	if (!resized_data) {
		printf("Failed to allocate memory for resized image\n");
		od_results.cls = -3;
		od_results.score = -3;
		return od_results;
	}

	// 调整图像大小
	if (!stbir_resize_uint8_linear(input_data.data, input_data.width, input_data.height, 0,
		resized_data, cls_stat_door.img_width, cls_stat_door.img_height, 0, READ_IMAGE_TYPE)) {
		printf("Failed to resize image\n");
		free(resized_data);
		od_results.cls = -2;
		od_results.score = -2;
		return od_results;
	}

	image_buffer_t src_image;
	memset(&src_image, 0, sizeof(image_buffer_t));

	src_image.width = cls_stat_door.img_width;
	src_image.height = cls_stat_door.img_height;
	src_image.format = IMAGE_FORMAT_RGB888;
	src_image.size = cls_stat_door.img_height * cls_stat_door.img_width * input_data.channel;
	// src_image.virt_addr = resized_img.data;
	src_image.virt_addr = resized_data;

	int ret = inference_mobilenet_model(app_ctx, &src_image, &od_results, cls_stat_door.top_k);
	if (ret != 0)
	{
		od_results.cls = -1;
		od_results.score = -1;
		printf("init_rec_ren_resnet_model fail! ret=%d\n", ret);
		free(resized_data);
		return od_results;
	}

	// Clean up resources
	free(resized_data);

	return od_results;
}
