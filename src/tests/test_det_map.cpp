/*
检测模型计算map；
以行人检测模型为例，计算map
输入：测试数据集（img_path json_path）、模型名、模型路径、推理函数
输出：map/PR
*/
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>
#include <map>

#include "stb_image.h"
#include "model_func.hpp"
#include "json.hpp"  // 包含 nlohmann/json 库

using json = nlohmann::json;

struct DetectionResult {
    int class_id;
    float score;
    std::vector<float> bbox;
    bool is_true_positive;
};

// 计算某个类别的 Precision-Recall 曲线
std::vector<std::pair<float, float>> compute_precision_recall_curve(const std::vector<DetectionResult>& results, int num_gt_boxes) {
    std::vector<DetectionResult> sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end(), [](const DetectionResult& a, const DetectionResult& b) {
        return a.score > b.score;
    });

    int true_positives = 0;
    int false_positives = 0;
    std::vector<std::pair<float, float>> pr_curve;

    for (const auto& result : sorted_results) {
        if (result.is_true_positive) {
            true_positives++;
        } else {
            false_positives++;
        }
        float precision = static_cast<float>(true_positives) / (true_positives + false_positives);
        float recall = static_cast<float>(true_positives) / num_gt_boxes;
        pr_curve.push_back({recall, precision});
    }

    return pr_curve;
}

// 计算 AP 使用 11 点插值法
float compute_average_precision(const std::vector<std::pair<float, float>>& pr_curve) {
    float ap = 0.0f;
    for (float recall = 0.0f; recall <= 1.0f; recall += 0.1f) {
        float max_precision = 0.0f;
        for (const auto& point : pr_curve) {
            if (point.first >= recall) {
                max_precision = std::max(max_precision, point.second);
            }
        }
        ap += max_precision;
    }
    return ap / 11.0f;
}


// 计算指定阈值下的 Precision 和 Recall
std::pair<float, float> compute_precision_recall_at_threshold(const std::vector<DetectionResult>& results, int num_gt_boxes, float threshold) {
    int true_positives = 0;
    int false_positives = 0;

    for (const auto& result : results) {
        if (result.score >= threshold) {
            if (result.is_true_positive) {
                true_positives++;
            } else {
                false_positives++;
            }
        }
    }

    float precision = (true_positives + false_positives) > 0 ? static_cast<float>(true_positives) / (true_positives + false_positives) : 0.0f;
    float recall = num_gt_boxes > 0 ? static_cast<float>(true_positives) / num_gt_boxes : 0.0f;

    return {precision, recall};
}

static det_model_input read_image(const char *image_path){
    // Load image
    int width, height, channel;
    unsigned char* data = stbi_load(image_path, &width, &height, &channel, 3);
    if (data == NULL) {
        printf("Failed to load image from path: %s\n", image_path);
        exit(1);
    }
    // init input data
    det_model_input input_data;
    input_data.data = data;
    input_data.width = width;
    input_data.height = height;
    input_data.channel = channel;
    return input_data;
}


// 读取 JSON 文件并提取标注类别和标注框
std::vector<std::vector<float>> read_json_labels(const std::string& json_path, std::map<std::string, int> label_name_map) {
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        std::cerr << "Failed to open JSON file: " << json_path << std::endl;
        return {};
    }

    json data;
    json_file >> data;

    std::vector<std::vector<float>> gt_box_list;
    for (const auto& obj : data["shapes"]) {
        std::string label_name = obj["label"].get<std::string>();
        // 如果label_name不在label_name_map中，则continue
        if (label_name_map.find(label_name) == label_name_map.end()) {
            continue;
        }
        int label_id = label_name_map[label_name];
        float x1 = obj["points"][0][0].get<float>();
        float y1 = obj["points"][0][1].get<float>();
        float x2 = obj["points"][1][0].get<float>();
        float y2 = obj["points"][1][1].get<float>();
        gt_box_list.push_back({static_cast<float>(label_id), x1, y1, x2, y2});
    }
    return gt_box_list;
}

// 计算两个边界框之间的 IoU
float calculate_iou(const std::vector<float>& box1, const std::vector<float>& box2) {
    float x1_min = std::max(box1[0], box2[0]);
    float y1_min = std::max(box1[1], box2[1]);
    float x2_max = std::min(box1[2], box2[2]);
    float y2_max = std::min(box1[3], box2[3]);

    float intersection_area = std::max(0.0f, x2_max - x1_min) * std::max(0.0f, y2_max - y1_min);
    float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = box1_area + box2_area - intersection_area;

    return intersection_area / union_area;
}

// 匹配预测框和真实框
std::vector<bool> match_boxes(const std::vector<std::vector<float>>& pred_boxes, const std::vector<std::vector<float>>& gt_boxes, float nms_threshold) {
    std::vector<bool> pred_matched(gt_boxes.size(), false); // 记录预测框是否与真实框匹配
    std::vector<bool> gt_matched(gt_boxes.size(), false);  // 记录每个 gt_box 是否已经被匹配过

    for (size_t i = 0; i < pred_boxes.size(); ++i) {
        float max_iou = 0.0f;
        size_t max_index = 0;
        bool found = false;
        for (size_t j = 0; j < gt_boxes.size(); ++j) {
            if (gt_matched[j]) {  // 如果 gt_box 已经被匹配过，跳过
                continue;
            }
            // 检查类别标签是否相同
            if (pred_boxes[i][0] != gt_boxes[j][0]) {
                continue;
            }
            // 创建一个从索引1开始的子向量,不包含类别id
            std::vector<float> box1(pred_boxes[i].begin() + 1, pred_boxes[i].end());
            std::vector<float> box2(gt_boxes[j].begin() + 1, gt_boxes[j].end());
            float iou = calculate_iou(box1, box2);
            if (iou > max_iou) {
                max_iou = iou;
                max_index = j;
                found = true;
            }
        }
        if (found && max_iou > nms_threshold) {
            pred_matched[i] = true;
            gt_matched[max_index] = true;// 标记 gt_box 已被匹配
        }
    }
    return pred_matched;
}

int DetModelMapCalculator(DetModelManager& modelManager, const std::string& modelName, const char *testset_file, std::map<std::string, int> label_name_map, float CONF_THRESHOLD, float NMS_THRESHOLD){

    // 获取模型信息
    DetModelInfo modelInfo;
    try {
        modelInfo = modelManager.getModel(modelName);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    // 模型初始化
    int ret = 0;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    const char* cls_model_path =  modelInfo.modelPath.c_str();
    ret = init_model(cls_model_path, &rknn_app_ctx);

    // 读取文件中的每一行的图片路径和json标注文件：image_path json_path
    std::ifstream infile(testset_file);
    if (!infile) {
        std::cout << "Failed to open file: " << testset_file << std::endl;
        return -1;
    }
    std::string line;
    int total_cnt = 0;
    int num_class = rknn_app_ctx.io_num.n_output;
    std::vector<std::vector<DetectionResult>> all_results(num_class);
    std::map<int, int> gt_box_counts;  // 存储每个类别的真实框数量

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string image_path;
        std::string json_path;
        if (!(iss >> image_path >> json_path)) {
            std::cerr << "Error reading image path from line: " << line << std::endl;
            continue;
        }
        total_cnt++;
        // 读取json标注文件得到box列表
        std::vector<std::vector<float>> gt_boxes = read_json_labels(json_path, label_name_map);
        // 更新每个类别的真实框数量
        for (const auto& gt_box : gt_boxes) {
            int class_id = static_cast<int>(gt_box[0]);
            gt_box_counts[class_id]++;
        }
        // 模型推理图片获取结果
        det_model_input input_data = read_image(image_path.c_str());
        box_rect header_box;
        header_box.left = 0;
        header_box.top = 0;
        header_box.right = input_data.width;
        header_box.bottom = input_data.height;
        object_detect_result_list result = modelInfo.inferenceFunc(&rknn_app_ctx, input_data, false);
        std::vector<std::vector<float>> pred_boxes;
        std::vector<float> pred_scores;
        for (int i = 0; i < result.count; i++) {
            float score = result.results[i].prop;
            if (score < 0.1) {
                continue;
            }
            float x1 = result.results[i].box.left;
            float y1 = result.results[i].box.top;
            float x2 = result.results[i].box.right;
            float y2 = result.results[i].box.bottom;
            int label_id = result.results[i].cls_id;
            pred_boxes.push_back({static_cast<float>(label_id), x1, y1, x2, y2});
            pred_scores.push_back(score);
        }
         // 匹配预测框和真实框
        std::vector<bool> matched = match_boxes(pred_boxes, gt_boxes, NMS_THRESHOLD);
        // 存储所有类别的已经匹配到gt_box的预测结果
        for (size_t i = 0; i < pred_boxes.size(); i++) {
            int class_id = static_cast<int>(pred_boxes[i][0]); // 类别id
            DetectionResult det_result;
            det_result.class_id = class_id;
            det_result.score = pred_scores[i];
            det_result.bbox = {pred_boxes[i][1], pred_boxes[i][2], pred_boxes[i][3], pred_boxes[i][4]};
            det_result.is_true_positive = matched[i];
            all_results[class_id].push_back(det_result);
        }
        if (input_data.data != nullptr){
            free(input_data.data);
        }
        printf("Person Count Result: %zu\n", all_results[0].size());
        printf("Processed image: %d\n", total_cnt);
    }
    printf("Total images: %d\n", total_cnt);
    // 计算每个类别的 AP
    std::vector<float> aps(num_class, 0.0f);
    for (int class_id = 0; class_id < num_class; class_id++) {
        if (!all_results[class_id].empty()) {
            int num_gt_boxes = gt_box_counts[class_id];  // 使用存储的真实框数量
            std::vector<std::pair<float, float>> pr_curve = compute_precision_recall_curve(all_results[class_id], num_gt_boxes);
            aps[class_id] = compute_average_precision(pr_curve);
            std::cout << "Class " << class_id << ": AP = " << aps[class_id] << std::endl;
            // 计算指定阈值下的 Precision 和 Recall
            for (float thres = CONF_THRESHOLD; thres <= 1.0f; thres += 0.1f){
                std::pair<float, float> pr_at_threshold = compute_precision_recall_at_threshold(all_results[class_id], num_gt_boxes, thres);
                std::cout << "Class " << class_id << ": Precision at threshold " << thres << " = " << pr_at_threshold.first << std::endl;
                std::cout << "Class " << class_id << ": Recall at threshold " << thres << " = " << pr_at_threshold.second << std::endl;
            }
            
        }
    }

    // 计算 mAP
    float mAP = 0.0f;
    for (float ap : aps) {
        mAP += ap;
    }
    mAP /= num_class;
    std::cout << "mAP: " << mAP << std::endl;

    ret = release_model(&rknn_app_ctx);
    infile.close();
    return 0;
}
