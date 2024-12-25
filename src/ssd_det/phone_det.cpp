
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
ssd_det_result inference_phone_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger=false)
{
    // rknn_context ctx = 0;
    int            ret;
    // int            model_len = 0;
    // unsigned char* model;
    ssd_det_result result;

    const int num_class = 7;
    float det_threshold = 0.2;
    // const char* model_path = "model/HeaderDet.rknn";

    unsigned char* data = input_data.data;
    int width = input_data.width;
    int height = input_data.height;
    int channel = input_data.channel;

    cv::Mat cv_img(height, width, CV_8UC3, data);
    // cv::Mat orig_img(height, width, CV_MAKETYPE(CV_8U, channels), input_data);
    if (cv_img.empty()) {
        std::cerr << "Image is empty or invalid." << std::endl;
    }
    // orig_img的通道顺序为cv图片的默认顺序bgr
    cv::Mat orig_img;
    cv::cvtColor(cv_img, orig_img, cv::COLOR_RGB2BGR); 

    float scale_x = (float)width / (float)app_ctx->model_width;  // 2550/960=2.6
    // 如果scale_x 大于2，则将原图分成patch，每张patch的尺寸为原图尺寸的1/2，再进行检测
    if (scale_x > 2) {
        ret = -2; 
        int patchHeight = height / 2;
        int patchWidth = width / 2;
        ssd_det_result all_patch_result;
         // 遍历四个角的patch并进行推理
        int count = 0;
        for (int i = 0; i < 2; ++i) { // 遍历上下两行
            for (int j = 0; j < 2; ++j) { // 遍历左右两列
                ssd_det_result patch_result;
                cv::Rect roi(j * patchWidth, i * patchHeight, patchWidth, patchHeight);
                cv::Mat patch = orig_img(roi);
                ret = inference_retinanet_model(app_ctx, patch, &patch_result, num_class);
                // 检测结果合并
                // printf("patch result num: %d\n", patch_result.count);
                for(int k = 0; k < patch_result.count; ++k){
                    // std::cout << patch_result.object->score << std::endl;
                    // std::cout << "01:patch result: " << patch_result.object[k].box.left << "," << patch_result.object[k].box.top << ","
                    // << patch_result.object[k].box.right << "," << patch_result.object[k].box.bottom << std::endl;
                    patch_result.object[k].box.left += j * patchWidth;
                    patch_result.object[k].box.top += i * patchHeight;
                    patch_result.object[k].box.right += j * patchWidth;
                    patch_result.object[k].box.bottom += i * patchHeight;
                    // std::cout << "02:patch result: " << patch_result.object[k].box.left << "," << patch_result.object[k].box.top << ","
                    // << patch_result.object[k].box.right << "," << patch_result.object[k].box.bottom << std::endl;
                    all_patch_result.object[count] = patch_result.object[k];
                    count ++;
                }
            }
        }
        // TODO: 中心区域进行推理
        ssd_det_result patch_result;
        float ctr_k = 0.5;
        cv::Rect roi(ctr_k * patchWidth, ctr_k * patchHeight, patchWidth, patchHeight);
        cv::Mat patch = orig_img(roi);
        ret = inference_retinanet_model(app_ctx, patch, &patch_result, num_class);
        // 检测结果合并
        // printf("patch result num: %d\n", patch_result.count);
        for(int k = 0; k < patch_result.count; ++k){
            // std::cout << patch_result.object->score << std::endl;
            // std::cout << "01:patch result: " << patch_result.object[k].box.left << "," << patch_result.object[k].box.top << ","
            // << patch_result.object[k].box.right << "," << patch_result.object[k].box.bottom << std::endl;
            patch_result.object[k].box.left += ctr_k * patchWidth;
            patch_result.object[k].box.top += ctr_k * patchHeight;
            patch_result.object[k].box.right += ctr_k * patchWidth;
            patch_result.object[k].box.bottom += ctr_k * patchHeight;
            // std::cout << "02:patch result: " << patch_result.object[k].box.left << "," << patch_result.object[k].box.top << ","
            // << patch_result.object[k].box.right << "," << patch_result.object[k].box.bottom << std::endl;
            all_patch_result.object[count] = patch_result.object[k];
            count ++;
        }
        all_patch_result.count = count;
        // all_patch_result检测结果NMS过滤，保存在result中
        float props[all_patch_result.count];
        int filter_indices[all_patch_result.count];
        float location[all_patch_result.count*4];
        for (int i = 0; i < all_patch_result.count; i++){
            props[i] = all_patch_result.object[i].score;
            filter_indices[i] = i;
            location[i*4] = all_patch_result.object[i].box.left;
            location[i*4+1] = all_patch_result.object[i].box.top;
            location[i*4+2] = all_patch_result.object[i].box.right;
            location[i*4+3] = all_patch_result.object[i].box.bottom;
        }
        quick_sort_indice_inverse(props, 0, all_patch_result.count - 1, filter_indices);
        // printf("filter_indices: %d %d\n", filter_indices[0], filter_indices[1]);
        nms(all_patch_result.count, location, filter_indices, 0.1, width, height);
        int last_count = 0;
        for (int i = 0; i < all_patch_result.count; ++i) {
            if (filter_indices[i] == -1 || props[i] < det_threshold) {
                // printf("filter_indices: %d %f\n", filter_indices[i], props[i]);
                continue;
            }
            int n = filter_indices[i];
            result.object[last_count] = all_patch_result.object[n];
            last_count++;
        }
        result.count = last_count;
        ret = 0;
    }else {
        // input image is raw image
        ret = inference_retinanet_model(app_ctx, orig_img, &result, num_class);
    }
    // ret = inference_retinanet_model(app_ctx, orig_img, &result, num_class);
    if (ret != 0) {
        if (enable_logger){
            printf("inference_retinanet_model fail! ret=%d\n", ret);
        }
        result.count = -1;
        return result;
    }
    if (enable_logger) {
        printf("Image size: %d x %d x %d\n", width, height, channel);
        if (scale_x > 2) {
            printf("phone detect model's input is patch\n");
        }else {
            printf("phone detect model's input is raw image\n");
        }
        printf("detect result num: %d\n", result.count);
        for (int i = 0; i < result.count; ++i) {
            printf("phone @(%d %d %d %d) score=%f\n", result.object[i].box.left, result.object[i].box.top,
                result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
        }
    }
    bool enable_draw_image = false; //画图,本地测试
    if (enable_draw_image) {
        cv::Mat orig_img_clone = orig_img.clone();
        // 保存输入图片
        // cv::imwrite("input.png", orig_img_clone);
        for (int i = 0; i < result.count; ++i) {
            if (enable_draw_image) {
                int rx = result.object[i].box.left;
                int ry = result.object[i].box.top;
                int rw = result.object[i].box.right - result.object[i].box.left;
                int rh = result.object[i].box.bottom - result.object[i].box.top;

                cv::Rect box(rx, ry, rw, rh);
                std::string text = "header";
                std::string score_str = std::to_string(result.object[i].score);
                text += " " + score_str;
                cv::Scalar color(0, 255, 0);  // 绿色
                cv::rectangle(orig_img_clone, box, color, 2);
                cv::Point textOrg(box.x, box.y - 10);
                cv::putText(orig_img_clone, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
                
            }
            if (result.object[i].score < det_threshold) {
                continue;
            }
            printf("header @(%d %d %d %d) score=%f\n", result.object[i].box.left, result.object[i].box.top,
                result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
        }
        const char* image_path = "header_det_result.png";
        cv::imwrite(image_path, orig_img_clone);
        std::cout << "Draw result on" << image_path << " is finished." << std::endl;
        }
    return result;
}
