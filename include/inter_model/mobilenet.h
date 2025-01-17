#ifndef _RKNN_DEMO_MOBILENET_H_
#define _RKNN_DEMO_MOBILENET_H_

#include "outer_model/model_params.hpp"
#include "common.h"


int init_mobilenet_model(const char* model_path, rknn_app_context_t* app_ctx);

int release_mobilenet_model(rknn_app_context_t* app_ctx);


#endif //_RKNN_DEMO_MOBILENET_H_