#ifndef _RKNN_DET_CLS_FUNC_H_
#define _RKNN_DET_CLS_FUNC_H_

#include <map>

#include "model_params.hpp"
#include "common.h"
#include "ppocr_system.h"

int init_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_model(rknn_app_context_t *app_ctx);

object_detect_result_list inference_phone_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

object_detect_result_list inference_header_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

retinaface_result inference_face_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

cls_model_result inference_face_attr_model(rknn_app_context_t *app_ctx, det_model_input input_data, box_rect header_box, bool enable_logger);

object_detect_result_list inference_coco_person_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

object_detect_result_list inference_person_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

ppocr_text_recog_array_result_t inference_ppocr_det_rec_model(ppocr_system_app_context *rknn_app_ctx, det_model_input input_data, bool enable_logger);

object_detect_result_list inference_det_knife_model(rknn_app_context_t* app_ctx, det_model_input input_data, const char* label_txt_path,bool enable_logger);
object_detect_result_list inference_det_knife_model(rknn_app_context_t* app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger);

/* det gun rga */
object_detect_result_list inference_det_gun_model(rknn_app_context_t* app_ctx, det_model_input input_data, const char* label_txt_path,bool enable_logger);

/* det gun opencv&params */
object_detect_result_list inference_det_gun_model(rknn_app_context_t* app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger);

/* det stat door rga */
object_detect_result_list inference_det_stat_door_model(rknn_app_context_t* app_ctx, det_model_input input_data, const char* label_txt_path, bool enable_logger);

/* det stat door  opencv&params*/
object_detect_result_list inference_det_stat_door_model(rknn_app_context_t* app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger);

// resnet rec ren
resnet_result inference_rec_person_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

// det hand
object_detect_result_list inference_det_hand_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool det_by_square = true, bool enable_logger = true);

// det_kx
object_detect_result_list inference_det_kx_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = true); 

// rec_hand
resnet_result inference_rec_hand_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = false);

// pose ren
object_detect_pose_result_list inference_pose_ren_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = true);

// rec kx orient
resnet_result inference_rec_kx_orient_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = true);

// pose kx_hp
object_detect_pose_result_list inference_pose_kx_hp_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = true);

// pose kx_sz
object_detect_pose_result_list inference_pose_kx_sz_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = true);

/* obb stick opencv&params */
object_detect_obb_result_list inference_obb_stick_model(rknn_app_context_t* app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger);

/* rec status door */
resnet_result inference_rec_stat_door_resnet18_model(rknn_app_context_t* app_ctx, det_model_input input_data, bool enable_logger);

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
