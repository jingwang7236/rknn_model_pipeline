#ifndef _RKNN_DET_CLS_FUNC_H_
#define _RKNN_DET_CLS_FUNC_H_

#include <map>

#include "model_params.hpp"
#include "common.h"
#include "ppocr_system.h"

int init_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_model(rknn_app_context_t *app_ctx);

ssd_det_result inference_phone_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

ssd_det_result inference_header_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

retinaface_result inference_face_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

cls_model_result inference_face_attr_model(rknn_app_context_t *app_ctx, det_model_input input_data, box_rect header_box, bool enable_logger);

object_detect_result_list inference_person_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

ppocr_text_recog_array_result_t inference_ppocr_det_rec_model(ppocr_system_app_context *rknn_app_ctx, det_model_input input_data, bool enable_logger);

object_detect_result_list inference_det_knife_model(rknn_app_context_t* app_ctx, det_model_input input_data, bool enable_logger);

// resnet rec ren
resnet_result inference_rec_person_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

// det hand
object_detect_result_list inference_det_hand_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

// 模型管理类
class ClsModelManager{
    public:
        void addModel(const std::string& modelName, const std::string& modelPath, ClsInferenceFunction inferenceFunc) {
            models[modelName] = {modelName, modelPath, inferenceFunc};
        }

        ClsModelInfo getModel(const std::string& modelName) {
            if (models.find(modelName) != models.end()) {
                return models[modelName];
            } else {
                throw std::runtime_error("Model not found");
            }
        }

    private:
        std::map<std::string, ClsModelInfo> models;
};

class DetModelManager{
    public:
        void addModel(const std::string& modelName, const std::string& modelPath, DetInferenceFunction inferenceFunc) {
            models[modelName] = {modelName, modelPath, inferenceFunc};
        }

        DetModelInfo getModel(const std::string& modelName) {
            if (models.find(modelName) != models.end()) {
                return models[modelName];
            } else {
                throw std::runtime_error("Model not found");
            }
        }

    private:
        std::map<std::string, DetModelInfo> models;
};

int ClsModelAccuracyCalculator(ClsModelManager& modelManager, const std::string& modelName, const char *testset_file);
int DetModelMapCalculator(DetModelManager& modelManager, const std::string& modelName, const char *testset_file, std::map<std::string, int> label_name_map, float CONF_THRESHOLD, float NMS_THRESHOLD);

#endif // _RKNN_DET_CLS_FUNC_H_
