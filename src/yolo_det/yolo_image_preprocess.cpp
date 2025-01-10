#include "yolo_image_preprocess.h"


// 传入目标边长的 pad resize
ImagePreProcess::ImagePreProcess(int width, int height, int target_size): target_width_(target_size), target_height_(target_size) {
    scale_ = static_cast<double>(target_size) / std::max(height, width);
    padding_x_ = target_size - static_cast<int>(width * scale_);
    padding_y_ = target_size - static_cast<int>(height * scale_);
    new_size_ = cv::Size(static_cast<int>(width * scale_), static_cast<int>(height * scale_));
    letterbox_.scale = scale_;
    letterbox_.x_pad = padding_x_ / 2;
    letterbox_.y_pad = padding_y_ / 2;
}

// 传入目标宽高的 pad resize
ImagePreProcess::ImagePreProcess(int width, int height, int target_width, int target_height): target_width_(target_width), target_height_(target_height) {
    // 计算宽度和高度的缩放比例，并取较小的缩放比例
    double width_scale = static_cast<double>(target_width) / width;
    double height_scale = static_cast<double>(target_height) / height;
    scale_ = std::min(width_scale, height_scale);
    // 根据缩放比例计算新的宽高
    new_size_ = cv::Size(static_cast<int>(width * scale_), static_cast<int>(height * scale_));
    // 计算填充大小
    padding_x_ = target_width - new_size_.width;
    padding_y_ = target_height - new_size_.height;
    // 更新 letterbox 信息
    letterbox_.scale = scale_;
    letterbox_.x_pad = padding_x_ / 2;
    letterbox_.y_pad = padding_y_ / 2;
}

// 进行 pad resize 转化
std::unique_ptr<cv::Mat> ImagePreProcess::Convert(const cv::Mat &src) {
    // 检查输入图像是否为空，如果为空，则返回空指针
    if (&src == nullptr) {
        return nullptr;
    }
    // 创建一个中间变量 resize_img 来存储缩放后的图像
    cv::Mat resize_img;
    // 使用 OpenCV 的 resize 函数，将输入图像 src 按照 new_size_ 缩放
    cv::resize(src, resize_img, new_size_);
    // 创建一个目标图像，大小为 target_height_ x target_width，图像类型与 src 相同，默认填充颜色为灰色（114, 114, 114）
    // auto pad_img = std::make_unique<cv::Mat>(target_height_, target_width_, src.type(), cv::Scalar(114, 114, 114));  // C++14后可用
    std::unique_ptr<cv::Mat> pad_img(new cv::Mat(target_height_, target_width_, src.type(), cv::Scalar(114, 114, 114)));

    // 计算缩放后图像在目标正方形图像中的起始位置（左上角）
    cv::Point position(padding_x_ / 2, padding_y_ / 2);
    // 将缩放后的图像 resize_img 拷贝到目标图像 pad_img 的中心区域（根据 padding_x_ 和 padding_y_ 填充）
    resize_img.copyTo((*pad_img)(cv::Rect(position.x, position.y, resize_img.cols, resize_img.rows)));
    // 返回填充后的图像 pad_img 的指针
    return std::move(pad_img);
}

// const char* 转 cv::mat
cv::Mat convertDataToCvMat(const det_model_input& input) {
    // 检查输入数据是否有效
    if (input.data == nullptr || input.width <= 0 || input.height <= 0 || input.channel <= 0) {
        throw std::invalid_argument("Invalid input data for det_model_input.");
    }

    // 根据通道数创建 cv::Mat
    int type = (input.channel == 1) ? CV_8UC1 : (input.channel == 3 ? CV_8UC3 : CV_8UC4);
    cv::Mat img(input.height, input.width, type, input.data);

    // 如果是 4 通道（RGBA），转换为 3 通道（RGB）
    if (input.channel == 4) {
        cv::Mat rgb_img;
        cv::cvtColor(img, rgb_img, cv::COLOR_RGBA2RGB);  // 将 RGBA 转换为 RGB
        return rgb_img;
    }

    // 如果是 1 或 3 通道，直接返回
    return img;
}


image_rect_t convertObbToAabb(const image_obb_box_t& obb) {
    // 计算角度的弧度值
    float angle_rad = obb.angle * M_PI / 180.0f;

    // 计算旋转矩形的四个角点
    float cos_angle = cos(angle_rad);
    float sin_angle = sin(angle_rad);

    // 计算四个角点的位置
    float half_w = obb.w / 2.0f;
    float half_h = obb.h / 2.0f;

    // 角点坐标，按照顺时针方向计算
    std::vector<std::pair<float, float>> corners = {
        {obb.x + cos_angle * half_w - sin_angle * half_h, obb.y + sin_angle * half_w + cos_angle * half_h},
        {obb.x - cos_angle * half_w - sin_angle * half_h, obb.y - sin_angle * half_w + cos_angle * half_h},
        {obb.x - cos_angle * half_w + sin_angle * half_h, obb.y - sin_angle * half_w - cos_angle * half_h},
        {obb.x + cos_angle * half_w + sin_angle * half_h, obb.y + sin_angle * half_w - cos_angle * half_h}
    };

    // 获取 AABB（最小矩形）
    float min_x = std::min({ corners[0].first, corners[1].first, corners[2].first, corners[3].first });
    float max_x = std::max({ corners[0].first, corners[1].first, corners[2].first, corners[3].first });
    float min_y = std::min({ corners[0].second, corners[1].second, corners[2].second, corners[3].second });
    float max_y = std::max({ corners[0].second, corners[1].second, corners[2].second, corners[3].second });

    // 返回 AABB
    return image_rect_t{ static_cast<int>(min_x), static_cast<int>(min_y), static_cast<int>(max_x), static_cast<int>(max_y) };
}


// 将图片裁剪成一组正方形块
std::vector<CropInfo> cropImage(const cv::Mat &image) {
    int height = image.rows;
    int width = image.cols;

    // 获取最小边和最大边
    int min_side = std::min(height, width);
    int max_side = std::max(height, width);

    // 计算裁剪块数和步长
    int num_blocks = std::ceil(static_cast<float>(max_side) / min_side);
    float overlap = (num_blocks > 1) ? (num_blocks * min_side - max_side) / (num_blocks - 1) : 0;
    float step = min_side - overlap;

    std::vector<CropInfo> cropped_images;

    for (int block_id = 0; block_id < num_blocks; ++block_id) {
        int x_start = 0, y_start = 0;

        if (width > height) { // 宽度更长
            x_start = static_cast<int>(block_id * step);
            int x_end = std::min(x_start + min_side, width);
            CropInfo crop_info = {image(cv::Rect(x_start, 0, x_end - x_start, height)), x_start, 0};
            cropped_images.push_back(crop_info);

            // 保存当前裁剪的图片
            // std::string filename = "crop_" + std::to_string(block_id) + ".png";
            // cv::imwrite(filename, crop_info.cropped_img);

        } else { // 高度更长
            y_start = static_cast<int>(block_id * step);
            int y_end = std::min(y_start + min_side, height);
            CropInfo crop_info = {image(cv::Rect(0, y_start, width, y_end - y_start)), 0, y_start};
            cropped_images.push_back(crop_info);

            // 保存当前裁剪的图片
            // std::string filename = "crop_" + std::to_string(block_id) + ".png";
            // cv::imwrite(filename, crop_info.cropped_img);
        }
    }

    return cropped_images;
}

// 偏移检测框坐标
std::vector<cv::Rect> offsetBboxes(const std::vector<cv::Rect> &bboxes, int x_offset, int y_offset) {
    std::vector<cv::Rect> offset_bboxes;
    for (const auto &bbox : bboxes) {
        offset_bboxes.emplace_back(bbox.x + x_offset, bbox.y + y_offset, bbox.width, bbox.height);
    }
    return offset_bboxes;
}

// 计算两个检测框的 IoU
float calculateIoU(const image_rect_t &box1, const image_rect_t &box2) {
    float interLeft = std::max(box1.left, box2.left);
    float interTop = std::max(box1.top, box2.top);
    float interRight = std::min(box1.right, box2.right);
    float interBottom = std::min(box1.bottom, box2.bottom);

    float interWidth = std::max(0.0f, interRight - interLeft);
    float interHeight = std::max(0.0f, interBottom - interTop);
    float interArea = interWidth * interHeight;

    float box1Area = (box1.right - box1.left) * (box1.bottom - box1.top);
    float box2Area = (box2.right - box2.left) * (box2.bottom - box2.top);

    float unionArea = box1Area + box2Area - interArea;

    return (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
}

// NMS 函数
std::vector<int> crop_nms_with_classes(const std::vector<image_rect_t> &bboxes, 
                                       const std::vector<float> &confidences, 
                                       const std::vector<int> &classes, 
                                       float iou_threshold) {
    // 检查输入有效性
    if (bboxes.empty() || confidences.empty() || classes.empty() || 
        bboxes.size() != confidences.size() || bboxes.size() != classes.size()) {
        return {};
    }

    // 保存最终保留的索引
    std::vector<int> keep;

    // 获取所有类别的集合
    std::set<int> unique_classes(classes.begin(), classes.end());

    // 对每个类别分别进行 NMS
    for (int cls : unique_classes) {
        // 保存属于该类别的索引
        std::vector<int> cls_indices;
        for (size_t i = 0; i < classes.size(); ++i) {
            if (classes[i] == cls) {
                cls_indices.push_back(i);
            }
        }

        // 按照置信度对属于该类别的框排序
        std::sort(cls_indices.begin(), cls_indices.end(), [&confidences](int i, int j) {
            return confidences[i] > confidences[j];
        });

        // 执行 NMS
        while (!cls_indices.empty()) {
            int current = cls_indices.front(); // 取出置信度最高的框
            keep.push_back(current);          // 保留该框
            cls_indices.erase(cls_indices.begin()); // 从索引列表中移除

            // 筛选与当前框 IoU 小于阈值的框
            cls_indices.erase(
                std::remove_if(cls_indices.begin(), cls_indices.end(), [&](int idx) {
                    float iou = calculateIoU(bboxes[current], bboxes[idx]);
                    return iou > iou_threshold; // 移除 IoU 大于阈值的框
                }),
                cls_indices.end());
        }
    }

    return keep; // 返回所有保留的框的索引
}


std::vector<int> crop_nms_with_classes(const std::vector<image_obb_box_t>& bboxes,
    const std::vector<float>& confidences,
    const std::vector<int>& classes,
    float iou_threshold) {
    // 检查输入有效性
    if (bboxes.empty() || confidences.empty() || classes.empty() ||
        bboxes.size() != confidences.size() || bboxes.size() != classes.size()) {
        return {};
    }

    // 保存最终保留的索引
    std::vector<int> keep;

    // 获取所有类别的集合
    std::set<int> unique_classes(classes.begin(), classes.end());

    // 对每个类别分别进行 NMS
    for (int cls : unique_classes) {
        // 保存属于该类别的索引
        std::vector<int> cls_indices;
        for (size_t i = 0; i < classes.size(); ++i) {
            if (classes[i] == cls) {
                cls_indices.push_back(i);
            }
        }

        // 按照置信度对属于该类别的框排序
        std::sort(cls_indices.begin(), cls_indices.end(), [&confidences](int i, int j) {
            return confidences[i] > confidences[j];
        });

        // 执行 NMS
        while (!cls_indices.empty()) {
            int current = cls_indices.front(); // 取出置信度最高的框
            keep.push_back(current);          // 保留该框
            cls_indices.erase(cls_indices.begin()); // 从索引列表中移除

            // 将旋转矩形 (OBB) 转换为 AABB
            image_rect_t aabb_current = convertObbToAabb(bboxes[current]);

            // 筛选与当前框 IoU 小于阈值的框
            cls_indices.erase(
                std::remove_if(cls_indices.begin(), cls_indices.end(), [&](int idx) {
                // 将旋转矩形 (OBB) 转换为 AABB
                image_rect_t aabb_idx = convertObbToAabb(bboxes[idx]);

                // 计算 AABB 的 IoU
                float iou = calculateIoU(aabb_current, aabb_idx);
                return iou > iou_threshold; // 移除 IoU 大于阈值的框
            }),
                cls_indices.end());
        }
    }

    return keep; // 返回所有保留的框的索引
}


int processDetBySquare(rknn_app_context_t *app_ctx, object_detect_result_list *od_results, det_model_input input_data,
                        int input_width, int input_height, float nms_thresh, float box_thresh, bool enable_logger) {
    // det_model_input 转 cv::Mat
    cv::Mat image = convertDataToCvMat(input_data);
    
    // 裁剪图片
    std::vector<CropInfo> cropped_images = cropImage(image);

    // 用于保存结果
    std::vector<image_rect_t> all_bboxes;
    std::vector<float> all_confidences;
    std::vector<int> all_classes;

    // 实例化填充对象，实际上只需resize，不需要填充
    ImagePreProcess det_image_preprocess(cropped_images[0].cropped_img.cols, cropped_images[0].cropped_img.rows, input_width, input_height);

    for (const auto &crop_info : cropped_images) {
        // 推理前处理
        auto convert_img = det_image_preprocess.Convert(crop_info.cropped_img);     // 使用智能指针解引用 convert_img->ptr<unsigned char>()
        // cv::Mat img_rgb = cv::Mat::zeros(input_width, input_height, convert_img->type());
        // convert_img->copyTo(img_rgb);    // img_rgb->ptr()

        // 执行推理
        object_detect_result_list temp_results;
        int ret = inference_yolov8_model(app_ctx, convert_img->ptr<unsigned char>(), &temp_results, det_image_preprocess.get_letter_box(), nms_thresh, box_thresh, enable_logger);
        if (ret != 0) {
            printf("ERROR: Inference failed for one block, ret=%d\n", ret);
            return -1;
        }

        // 偏移检测框并整合结果
        for (int i = 0; i < temp_results.count; ++i) {
            auto &tmp_res = temp_results.results[i];
            tmp_res.box.left += crop_info.x_start;
            tmp_res.box.top += crop_info.y_start;
            tmp_res.box.right += crop_info.x_start;
            tmp_res.box.bottom += crop_info.y_start;
            all_bboxes.emplace_back(tmp_res.box);
            all_confidences.push_back(tmp_res.prop);
            all_classes.push_back(tmp_res.cls_id);
        }
    }

    memset(od_results, 0, sizeof(object_detect_result_list));

    // 应用 NMS 过滤重叠框
    std::vector<int> final_indices = crop_nms_with_classes(all_bboxes, all_confidences, all_classes, box_thresh);
    
    // 将保留的框添加到最终结果中
    for (int idx : final_indices) {
        if (od_results->count >= OBJ_NUMB_MAX_SIZE) {
            break; // 防止超出结果数组大小
        }

        auto &result = od_results->results[od_results->count++];
        result.box = all_bboxes[idx];
        result.prop = all_confidences[idx];
        result.cls_id = all_classes[idx];
    }

    od_results->id = 1; // 设置检测任务 ID

    return 0;
}

int processDetBySquareObb(rknn_app_context_t* app_ctx, object_detect_obb_result_list* od_results, det_model_input input_data,
    int input_width, int input_height, float nms_thresh, float box_thresh, bool enable_logger) {
    // det_model_input 转 cv::Mat
    cv::Mat image = convertDataToCvMat(input_data);

    // 裁剪图片
    std::vector<CropInfo> cropped_images = cropImage(image);

    // 用于保存结果
    std::vector<image_obb_box_t> all_bboxes;
    std::vector<float> all_confidences;
    std::vector<int> all_classes;

    // 实例化填充对象，实际上只需resize，不需要填充
    ImagePreProcess det_image_preprocess(cropped_images[0].cropped_img.cols, cropped_images[0].cropped_img.rows, input_width, input_height);

    for (const auto& crop_info : cropped_images) {
        // 推理前处理
        auto convert_img = det_image_preprocess.Convert(crop_info.cropped_img);     // 使用智能指针解引用 convert_img->ptr<unsigned char>()
        // cv::Mat img_rgb = cv::Mat::zeros(input_width, input_height, convert_img->type());
        // convert_img->copyTo(img_rgb);    // img_rgb->ptr()

        // 执行推理
        object_detect_obb_result_list temp_results;
        int ret = inference_yolov8_obb_model(app_ctx, convert_img->ptr<unsigned char>(), &temp_results, det_image_preprocess.get_letter_box(),1, nms_thresh, box_thresh, enable_logger);
        if (ret != 0) {
            printf("ERROR: Inference failed for one block, ret=%d\n", ret);
            return -1;
        }

        // 偏移检测框并整合结果
        for (int i = 0; i < temp_results.count; ++i) {
            auto& tmp_res = temp_results.results[i];
            
            // 处理旋转框的偏移
            image_obb_box_t obb = tmp_res.box;
            obb.x += crop_info.x_start;  // 偏移中心点 x
            obb.y += crop_info.y_start;  // 偏移中心点 y

            // 将调整后的 OBB 存储到 all_bboxes 中
            all_bboxes.push_back(obb);
            all_confidences.push_back(tmp_res.prop);
            all_classes.push_back(tmp_res.cls_id);
        }
    }

    memset(od_results, 0, sizeof(object_detect_obb_result_list));

    // 应用 NMS 过滤重叠框
    std::vector<int> final_indices = crop_nms_with_classes(all_bboxes, all_confidences, all_classes, box_thresh);

    // 将保留的框添加到最终结果中
    for (int idx : final_indices) {
        if (od_results->count >= OBJ_NUMB_MAX_SIZE) {
            break; // 防止超出结果数组大小
        }

        auto& result = od_results->results[od_results->count++];
        result.box = all_bboxes[idx];
        result.prop = all_confidences[idx];
        result.cls_id = all_classes[idx];
    }

    od_results->id = 1; // 设置检测任务 ID

    return 0;
}
