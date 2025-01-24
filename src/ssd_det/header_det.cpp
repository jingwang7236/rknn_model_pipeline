
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <iostream>

#include "outer_model/model_func.hpp"
#include "inter_model/retinanet.hpp"

/*-------------------------------------------
                  Functions
-------------------------------------------*/
/* 根据集成的需求修改接口
 * @param app_ctx: 上下文对象，函数外部负责初始化和释放资源
 * @param orig_img:  输入图片数据是rgb24，转换为cv格式
 * @param gpu_id:  显卡号，方便盒子端调度资源 ---未实现
 * @param logger:  是否打印日志
 * @param result:  返回结果

*/

object_detect_result_list inference_header_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger=false)
{
    std::string label_name = "header";
    std::string model_name = "header_det";
    // rknn_context ctx = 0;
    int            ret;
    // int            model_len = 0;
    // unsigned char* model;
    object_detect_result_list result;

    const int num_class = 1;
    float det_threshold = 0.5;
    // const char* model_path = "model/HeaderDet.rknn";

    unsigned char* data = input_data.data;
    int width = input_data.width;
    int height = input_data.height;
    int channel = input_data.channel;

    if (enable_logger){
        printf("Image size: %d x %d x %d\n", width, height, channel);
    }

    cv::Mat cv_img(height, width, CV_8UC3, data);
    // cv::Mat orig_img(height, width, CV_MAKETYPE(CV_8U, channels), input_data);
    if (cv_img.empty()) {
        std::cerr << "Image is empty or invalid." << std::endl;
    }
    // orig_img的通道顺序为cv图片的默认顺序bgr
    cv::Mat orig_img;
    cv::cvtColor(cv_img, orig_img, cv::COLOR_RGB2BGR); 

    ret = inference_retinanet_model(app_ctx, orig_img, &result, num_class, model_name.c_str());
    if (ret != 0) {
        if (enable_logger){
            printf("inference_retinanet_model fail! ret=%d\n", ret);
        }
        result.count = -1;
        return result;
    }
    if (enable_logger) {
        printf("detect result num: %d\n", result.count);
        for (int i = 0; i < result.count; ++i) {
            printf("%s @(%d %d %d %d) score=%f\n", label_name.c_str(), result.results[i].box.left, result.results[i].box.top,
                result.results[i].box.right, result.results[i].box.bottom, result.results[i].prop);
        }
    }
    bool enable_draw_image = false; //画图,本地测试
    if (enable_draw_image) {
        cv::Mat orig_img_clone = orig_img.clone();
        // 保存输入图片
        // cv::imwrite("input.png", orig_img_clone);
        for (int i = 0; i < result.count; ++i) {
            if (enable_draw_image) {
                int rx = result.results[i].box.left;
                int ry = result.results[i].box.top;
                int rw = result.results[i].box.right - result.results[i].box.left;
                int rh = result.results[i].box.bottom - result.results[i].box.top;

                cv::Rect box(rx, ry, rw, rh);
                std::string text = label_name;
                std::string score_str = std::to_string(result.results[i].prop);
                text += " " + score_str;
                cv::Scalar color(0, 255, 0);  // 绿色
                cv::rectangle(orig_img_clone, box, color, 2);
                cv::Point textOrg(box.x, box.y - 10);
                cv::putText(orig_img_clone, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
                
            }
            if (result.results[i].prop < det_threshold) {
                continue;
            }
            printf("%s @(%d %d %d %d) score=%f\n", label_name.c_str(), result.results[i].box.left, result.results[i].box.top,
                result.results[i].box.right, result.results[i].box.bottom, result.results[i].prop);
        }
        std::string image_path_str = label_name + "_det_result.png";
        const char* image_path = image_path_str.c_str();
        cv::imwrite(image_path, orig_img_clone);
        std::cout << "Draw result on" << image_path << " is finished." << std::endl;
        }
    return result;
}
