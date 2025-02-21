#include <iostream>
#include <string.h>
#include <dlfcn.h>

#include "model_func.hpp"
#include "model_params.hpp"

#include "stb_image.h"
#include "yaml-cpp/yaml.h"

#include <chrono>  // 计算耗时
using namespace std::chrono; 

/*
在链接动态库时可能会遇到找不到库的问题，解决方法如下：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${shared_library_abs_directory}

*/

// ================================ 输入输出相同的调用接口 ==============================
// ssd_det 模型: header_det/phone_det/face_det/coco_person_det/person_det
typedef object_detect_result_list (*InferenceSSDDetModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
// 人脸属性分类函数:face_attr
typedef cls_model_result (*InferenceFaceClsModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, box_rect header_box, bool enable_logger);
// ppocr 模型:ppocr
typedef ppocr_text_recog_array_result_t (*InferencePPOCRModelFunc)(ppocr_system_app_context *app_ctx, det_model_input input_data, bool enable_logger);
// yolo_det 模型: det_kx
typedef object_detect_result_list (*InferenceYOLODetModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
// yolo2_det 模型, 4个输入参数: hand_det
typedef object_detect_result_list (*InferenceYOLO2DetModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, bool det_by_square, bool enable_logger);
// yolo3_det 模型, 5个输入参数: det_knife
typedef object_detect_result_list (*InferenceYOLO3DetModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger);
// res_ren 分类模型:rec_ren/rec_ren_mobilenet/rec_kx_orient/rec_stat_door
typedef resnet_result (*InferenceResRecModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
// pose 模型: pose_ren/pose_kx_hp/pose_kx_sz
typedef object_detect_pose_result_list (*InferencePoseModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
//detect_obb 模型：obb_stick
typedef object_detect_obb_result_list (*InferenceOBBDetModelFunc)(rknn_app_context_t *app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger);


// 加载动态so库
static int loadSo(const char* soPath, void*& handle) {
    handle = dlopen(soPath, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        return 1;
    }
    dlerror();  // 清除之前的错误
    return 0;
}

// 加载图片
static det_model_input LoadImage(const char* image_path) {
    // Load image
    int width, height, channel;
    unsigned char* data = stbi_load(image_path, &width, &height, &channel, 3);
    if (data == NULL) {
        printf("Failed to load image from path: %s\n", image_path);
        exit;
    }

    // init input data
    det_model_input input_data;
    input_data.data = data;
    input_data.width = width;
    input_data.height = height;
    input_data.channel = channel;

    return input_data;
}

// 读取yaml文件
static YAML::Node ReadYamlFile(const char* yaml_path) {
    // Load YAML configuration
    YAML::Node config = YAML::LoadFile(yaml_path);
    YAML::Node model_node = config["models"];

    if (!model_node) {
        std::cerr << "Unknown model_name: models " <<  std::endl;
        throw std::invalid_argument("Unknown model_name: models");
    }
    return model_node;
}

static int InitModel(const char* model_path, rknn_app_context_t* rknn_app_ctx) {
    // const char* model_path = "model/HeaderDet.rknn";
    int ret = 0;
    memset(rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    ret = init_model(model_path, rknn_app_ctx);  // 初始化
    if (ret < 0) {
        printf("init_model failed!\n");
        return ret;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_name> <image_path>\n", argv[0]);
        return -1;
    }
    int ret = 0;
    const char *model_name = argv[1];
    const char *image_path = argv[2];  // single image path

    // 打开动态库
    dlerror();
    void* handle = NULL;
    const char *so_path = "/userdata/jingwang/rknn_model_pipeline/build/src/librknn_model_pipeline.so";
    ret = loadSo(so_path, handle);

    // 加载图片
    det_model_input input_data = LoadImage(image_path);

    // 读取yaml文件
    YAML::Node config = ReadYamlFile("models.yaml");
    YAML::Node model_node = config[model_name];
    if (!model_node) {
        std::cerr << "Unknown model_name: " << model_name << std::endl;
        throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
    }
    
    // 获取函数名和函数类型
    const std::string func_name_str = model_node["function_name"].as<std::string>();
    const char* func_name = func_name_str.c_str();
    std::cout << "func_name: " << func_name << std::endl;
    const std::string function_type_str = model_node["function_type"].as<std::string>();
    
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    if (function_type_str == "ppocr"){ // 多个模型
        const std::string enable_log_str = model_node["enable_log"].as<std::string>();
        bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
        const std::vector<std::string> model_path_vec = model_node["path"].as<std::vector<std::string>>();
        const char* det_model_path = model_path_vec[0].c_str();
        const char* rec_model_path = model_path_vec[1].c_str();
        ppocr_system_app_context rknn_app_ctx;
        ret = InitModel(det_model_path, &rknn_app_ctx.det_context);
        ret = InitModel(rec_model_path, &rknn_app_ctx.rec_context);
        // 函数指针类型，指向 inference函数
        InferencePPOCRModelFunc inference_model = (InferencePPOCRModelFunc)dlsym(handle, func_name);
        ppocr_text_recog_array_result_t results = inference_model(&rknn_app_ctx, input_data, enable_log);

    } else { // 单个模型
        const std::string model_path_str = model_node["path"].as<std::string>();
        const char* model_path = model_path_str.c_str();
        std::cout << "model_path: " << model_path << std::endl;
        rknn_app_context_t rknn_app_ctx;
        ret = InitModel(model_path, &rknn_app_ctx);
        if (ret < 0) {
            std::cerr << "InitModel failed" << std::endl;
            return ret;
        }
        if (function_type_str == "ssd_det") {
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            InferenceSSDDetModelFunc inference_model = (InferenceSSDDetModelFunc)dlsym(handle, func_name);
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            object_detect_result_list result = inference_model(&rknn_app_ctx, input_data, enable_log);
        }else if (function_type_str == "face_attr"){   // 头部检测+人脸属性
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            const std::string det_model_path_str = config["header_det"]["path"].as<std::string>();
            const char* det_model_path = det_model_path_str.c_str();
            rknn_app_context_t det_rknn_app_ctx;
            ret = InitModel(det_model_path, &det_rknn_app_ctx);
            InferenceSSDDetModelFunc det_inference_model = (InferenceSSDDetModelFunc)dlsym(handle, "inference_header_det_model");
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            object_detect_result_list det_result = det_inference_model(&det_rknn_app_ctx, input_data, enable_log);
            std::cout << "det_result.count:" << det_result.count << std::endl;
            InferenceFaceClsModelFunc inference_model = (InferenceFaceClsModelFunc)dlsym(handle, func_name);
            for (int i = 0; i < det_result.count; i++) {
                box_rect header_box;  // header的box
                header_box.left = det_result.results[i].box.left;
                header_box.top = det_result.results[i].box.top;
                header_box.right = det_result.results[i].box.right;
                header_box.bottom = det_result.results[i].box.bottom;
                // 人脸属性模型
                std::cout << "header_box.left:" << header_box.left << std::endl;
                std::cout << "header_box.top:" << header_box.top << std::endl;
                std::cout << "header_box.right:" << header_box.right << std::endl;
                std::cout << "header_box.bottom:" << header_box.bottom << std::endl;
                cls_model_result cls_result = inference_model(&rknn_app_ctx, input_data, header_box, enable_log);
            }
            ret = release_model(&det_rknn_app_ctx);
        }else if (function_type_str == "yolo_det"){
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            InferenceYOLODetModelFunc inference_model = (InferenceYOLODetModelFunc)dlsym(handle, func_name);
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            object_detect_result_list result = inference_model(&rknn_app_ctx, input_data, enable_log);
        }
        else if (function_type_str == "yolo2_det"){
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            const std::string is_quant_str = model_node["is_quant"].as<std::string>();
            bool is_quant = (is_quant_str == "true");
            rknn_app_ctx.is_quant = is_quant;
            InferenceYOLO2DetModelFunc inference_model = (InferenceYOLO2DetModelFunc)dlsym(handle, func_name);
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            object_detect_result_list result = inference_model(&rknn_app_ctx, input_data, false, enable_log);
        } else if (function_type_str == "yolo3_det"){
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            const std::string is_quant_str = model_node["is_quant"].as<std::string>();
            bool is_quant = (is_quant_str == "true" || is_quant_str == "True");
            rknn_app_ctx.is_quant = is_quant;
            model_inference_params params;
            params.input_height = model_node["params"]["input_height"].as<int>();
            params.input_width = model_node["params"]["input_width"].as<int>();
            params.nms_threshold = model_node["params"]["nms_threshold"].as<float>();
            params.box_threshold = model_node["params"]["box_threshold"].as<float>();
            InferenceYOLO3DetModelFunc inference_model = (InferenceYOLO3DetModelFunc)dlsym(handle, func_name);
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            object_detect_result_list result = inference_model(&rknn_app_ctx, input_data, params, false, enable_log);
        } 
        else if (function_type_str == "res_rec"){
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            InferenceResRecModelFunc inference_model = (InferenceResRecModelFunc)dlsym(handle, func_name);
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            resnet_result result = inference_model(&rknn_app_ctx, input_data, enable_log);
        } else if (function_type_str == "pose"){
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            InferencePoseModelFunc inference_model = (InferencePoseModelFunc)dlsym(handle, func_name);
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            object_detect_pose_result_list result = inference_model(&rknn_app_ctx, input_data, enable_log);
        } else if (function_type_str == "obb_det"){
            const std::string enable_log_str = model_node["enable_log"].as<std::string>();
            bool enable_log = (enable_log_str == "true" || enable_log_str == "True");
            model_inference_params params;
            params.input_height = model_node["params"]["input_height"].as<int>();
            params.input_width = model_node["params"]["input_width"].as<int>();
            params.nms_threshold = model_node["params"]["nms_threshold"].as<float>();
            params.box_threshold = model_node["params"]["box_threshold"].as<float>();
            InferenceOBBDetModelFunc inference_model = (InferenceOBBDetModelFunc)dlsym(handle, func_name);
            const char* dlsym_error = dlerror();
            if (dlsym_error) {
                std::cerr << "Cannot load symbol '" << func_name << "': " << dlsym_error << std::endl;
                dlclose(handle);
                return 1;
            }
            object_detect_obb_result_list result = inference_model(&rknn_app_ctx, input_data, params, false, enable_log);
        } else {
            std::cerr << "Unknown model_name: " << model_name << std::endl;
            throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
        }
    }
    
    ret = release_model(&rknn_app_ctx);  //释放
    if (ret != 0) {
        printf("release_retinanet_model fail! ret=%d\n", ret);
        return -1;
    }
    if (input_data.data) {
        stbi_image_free(input_data.data);
        input_data.data = nullptr;  // 确保指针置为 nullptr
    }
    // 关闭动态库
    if (handle != nullptr) {
        dlclose(handle);
    }
    return 0;
}

