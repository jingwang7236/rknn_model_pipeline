#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <math.h>
#include <set>
#include <vector>
#include <algorithm>

#include "Float16.h"
#include "yolo_postprocess.h"
#include <sys/time.h>

inline static int clamp_hw(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static char *readLine_hw(FILE *fp, char *buffer, int *len){
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines_hw(const char *fileName, char *lines[], int max_line){
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine_hw(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName_hw(const char *locationFilename, char *label[], int obj_class_num){
    // printf("load lable %s\n", locationFilename);
    readLines_hw(locationFilename, label, obj_class_num);
    return 0;
}

static float CalculateOverlap_hw(float xmin0, float ymin0, float xmax0, float ymax0, 
                                 float xmin1, float ymin1, float xmax1, float ymax1){
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

// box转化为角点
std::vector<float> rbbox_to_corners_hw(const std::vector<float>& rbbox) {
    // generate clockwise corners and rotate it clockwise
    // 顺时针方向返回角点位置
    float cx = rbbox[0] + rbbox[2] / 2;
    float cy = rbbox[1] + rbbox[3] / 2;
    float x_d = rbbox[2];
    float y_d = rbbox[3];
    float angle = rbbox[4];
    float a_cos = std::cos(angle);
    float a_sin = std::sin(angle);
    std::vector<float> corners(8, 0.0);
    float corners_x[4] = { -x_d / 2, -x_d / 2, x_d / 2, x_d / 2 };
    float corners_y[4] = { -y_d / 2, y_d / 2, y_d / 2, -y_d / 2 };
    for (int i = 0; i < 4; ++i) {
        corners[2 * i] = a_cos * corners_x[i] - a_sin * corners_y[i] + cx;
        corners[2 * i + 1] = a_sin * corners_x[i] + a_cos * corners_y[i] + cy;
    }
    return corners;
}

// 检测点是否在四边形内的函数
bool point_in_quadrilateral_hw(float pt_x, float pt_y, const std::vector<float>& corners) {
    float ab0 = corners[2] - corners[0];
    float ab1 = corners[3] - corners[1];

    float ad0 = corners[6] - corners[0];
    float ad1 = corners[7] - corners[1];

    float ap0 = pt_x - corners[0];
    float ap1 = pt_y - corners[1];

    float abab = ab0 * ab0 + ab1 * ab1;
    float abap = ab0 * ap0 + ab1 * ap1;
    float adad = ad0 * ad0 + ad1 * ad1;
    float adap = ad0 * ap0 + ad1 * ap1;

    return abab >= abap && abap >= 0 && adad >= adap && adap >= 0;
}

// 检测线段相交并计算交点的函数
int line_segment_intersection_hw(const std::vector<float>& pts1, const std::vector<float>& pts2,
                                 int i, int j, bool& ret1, float& point_x, float& point_y) {
    // pts1, pts2 分别为 corners
    // i j 分别表示第几个交点，取其和其后一个点构成的线段
    // 返回 tuple(bool, pts) bool=true pts为交点
    std::vector<float> A(2), B(2), C(2), D(2), ret(2);
    A[0] = pts1[2 * i];
    A[1] = pts1[2 * i + 1];

    B[0] = pts1[2 * ((i + 1) % 4)];
    B[1] = pts1[2 * ((i + 1) % 4) + 1];

    C[0] = pts2[2 * j];
    C[1] = pts2[2 * j + 1];

    D[0] = pts2[2 * ((j + 1) % 4)];
    D[1] = pts2[2 * ((j + 1) % 4) + 1];

    float BA0 = B[0] - A[0];
    float BA1 = B[1] - A[1];
    float DA0 = D[0] - A[0];
    float CA0 = C[0] - A[0];
    float DA1 = D[1] - A[1];
    float CA1 = C[1] - A[1];

    // 叉乘判断方向
    bool acd = DA1 * CA0 > CA1 * DA0;
    bool bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0]);

    if (acd != bcd) {
        bool abc = CA1 * BA0 > BA1 * CA0;
        bool abd = DA1 * BA0 > BA1 * DA0;

        // 判断方向
        if (abc != abd) {
            float DC0 = D[0] - C[0];
            float DC1 = D[1] - C[1];
            float ABBA = A[0] * B[1] - B[0] * A[1];
            float CDDC = C[0] * D[1] - D[0] * C[1];
            float DH = BA1 * DC0 - BA0 * DC1;
            float Dx = ABBA * DC0 - BA0 * CDDC;
            float Dy = ABBA * DC1 - BA1 * CDDC;
            ret[0] = Dx / DH;
            ret[1] = Dy / DH;
            ret1 = true;
            point_x = ret[0];
            point_y = ret[1];
            return 0;
        }
    }
    ret1 = false;
    point_x = ret[0];
    point_y = ret[1];
    return 0;
}

// 比较函数，用于排序
bool compare_points_hw(const std::vector<float>& pt1, const std::vector<float>& pt2, const std::vector<float>& center) {
    float vx1 = pt1[0] - center[0];
    float vy1 = pt1[1] - center[1];
    float vx2 = pt2[0] - center[0];
    float vy2 = pt2[1] - center[1];
    float d1 = std::sqrt(vx1 * vx1 + vy1 * vy1);
    float d2 = std::sqrt(vx2 * vx2 + vy2 * vy2);
    vx1 /= d1;
    vy1 /= d1;
    vx2 /= d2;
    vy2 /= d2;
    if (vy1 < 0) {
        vx1 = -2 - vx1;
    }
    if (vy2 < 0) {
        vx2 = -2 - vx2;
    }
    return vx1 < vx2;
}

// 对凸多边形的顶点进行排序
void sort_vertex_in_convex_polygon_hw(std::vector<std::vector<float>>& int_pts, int num_of_inter) {
    if (num_of_inter > 0) {
        std::vector<float> center(2, 0);
        for (int i = 0; i < num_of_inter; ++i) {
            center[0] += int_pts[i][0];
            center[1] += int_pts[i][1];
        }
        center[0] /= num_of_inter;
        center[1] /= num_of_inter;
        std::sort(int_pts.begin(), int_pts.end(), [&center](const std::vector<float>& pt1, const std::vector<float>& pt2) {
            return compare_points_hw(pt1, pt2, center);
        });
    }
}

// 计算三角形的面积
float triangle_area_hw(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
    return std::abs((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0;
}

// 计算多边形转化为多个三角形面积之和
float polygon_area_hw(const std::vector<std::vector<float>>& int_pts, int num_of_inter) {
    float area_val = 0.0;
    for (int i = 1; i < num_of_inter - 1; ++i) {
        area_val += triangle_area_hw(int_pts[0], int_pts[i], int_pts[i + 1]);
    }
    return area_val;
}

float Cal_IOU_hw(float x1, float y1, float w1, float h1, float angle1, float x2, float y2, float w2, float h2, float angle2) {
    // 定义两个box的数据
    std::vector<float> rbbox1 = { x1, y1, w1, h1, angle1 };
    std::vector<float> rbbox2 = { x2, y2, w2, h2, angle2 };

    // 调用函数得到角点数据
    std::vector<float> corners1 = rbbox_to_corners_hw(rbbox1);
    std::vector<float> corners2 = rbbox_to_corners_hw(rbbox2);

    std::vector<std::vector<float>> pts;
    int num_pts = 0;
    // 检测角点是否在对方的四边形内
    for (int i = 0; i < 4; ++i) {
        float point_x = corners1[2 * i];
        float point_y = corners1[2 * i + 1];
        if (point_in_quadrilateral_hw(point_x, point_y, corners2)) {
            num_pts++;
            pts.push_back({ point_x, point_y });
        }
    }
    for (int i = 0; i < 4; ++i) {
        float point_x = corners2[2 * i];
        float point_y = corners2[2 * i + 1];
        if (point_in_quadrilateral_hw(point_x, point_y, corners1)) {
            num_pts++;
            pts.push_back({ point_x, point_y });
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float point_x, point_y;
            bool ret;
            line_segment_intersection_hw(corners1, corners2, i, j, ret, point_x, point_y);
            if (ret) {
                num_pts++;
                pts.push_back({ point_x, point_y });
            }
        }
    }

    sort_vertex_in_convex_polygon_hw(pts, num_pts);

    float polygon_area_val = polygon_area_hw(pts, num_pts);

    // 计算 area_union
    float area_union = rbbox1[2] * rbbox1[3] + rbbox2[2] * rbbox2[3] - polygon_area_val;
    return polygon_area_val / area_union;
}

void softmax_hw(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += expf(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val) / sum_exp;
    }
}

static int nms_hw(int validCount, std::vector<float> &outputLocations, std::vector<float> objProbs, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold){
    // 优化点1：双指针+滑动窗口性质+YOLO检测结果性质，简化IoU计算
    // int invalid_count = 0;
    int i = 0, j = 1;
    while (i < validCount && j < validCount){
        while (i < validCount && (order[i] == -1 || classIds[order[i]] != filterId)) i++;  // 找到最小索引满足
        while (j < validCount && (order[j] == -1 || classIds[order[j]] != filterId)) j++;  // 找到移动索引满足
        if (j >= validCount) break;
        
        int n = order[i];

        while (j < validCount && order[j] != -1 && classIds[order[j]] == filterId){
            int m = order[j];

            // 计算两个框的坐标差值，则选择置信度较低的框置为 -1
            if (
                fabs(outputLocations[n * 4 + 0] - outputLocations[m * 4 + 0]) < 1.5 &&  // x
                fabs(outputLocations[n * 4 + 1] - outputLocations[m * 4 + 1]) < 1.5 &&  // y
                fabs(outputLocations[n * 4 + 2] - outputLocations[m * 4 + 2]) < 2.0 &&  // w
                fabs(outputLocations[n * 4 + 3] - outputLocations[m * 4 + 3]) < 2.0     // h
            ) {
                if (objProbs[i] >= objProbs[j]) {
                    order[j] = -1;  // 置信度较低的框置为无效
                    // invalid_count++;
                    j++;
                } else {
                    order[i] = -1;
                    //  invalid_count++;
                    i = j;  // 当前最小置信度低导致无效，修改为移动索引
                    j = i + 1;
                    break;
                }
            }else{
                i = j;  // 当前最小和下一个不同导致无效，修改为移动索引
                j = i + 1;
                break;
            }
        }
    }
    
    // printf("total valid count: %d, exclude invalid count: %d\n", validCount, invalid_count);

    for (int i = 0; i < validCount; ++i){
        int n = order[i];
        if (n == -1 || classIds[n] != filterId){
            continue;
        }
        for (int j = i + 1; j < validCount; ++j){
            int m = order[j];
            if (m == -1 || classIds[m] != filterId){
                continue;
            }

            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            // 优化点2：两个框没有交集，直接跳过
            if (xmin0 > xmax1 || xmax0 < xmin1 || ymin0 > ymax1 || ymax0 < ymin1) {
                continue;
            }

            float iou = CalculateOverlap_hw(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            // 优化点3：引入置信度，提早判断无效，并选择高置信度结果
            if (iou > threshold){
                if (objProbs[i] >= objProbs[j]){
                    order[j] = -1;
                }else{
                    order[i] = -1;
                    break;
                }
            }
        }
    }
    return 0;
}

static int nms_obb_hw(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
    int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 5 + 0];
            float ymin0 = outputLocations[n * 5 + 1];
            float w0 = outputLocations[n * 5 + 2];
            float h0 = outputLocations[n * 5 + 3];
            float angle0 = outputLocations[n * 5 + 4];

            float xmin1 = outputLocations[m * 5 + 0];
            float ymin1 = outputLocations[m * 5 + 1];
            float w1 = outputLocations[m * 5 + 2];
            float h1 = outputLocations[m * 5 + 3];
            float angle1 = outputLocations[n * 5 + 4];

            float iou = Cal_IOU_hw(xmin0, ymin0, w0, h0, angle0, xmin1, ymin1, w1, h1, angle1);
            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int nms_pose_hw(int validCount, std::vector<float>& outputLocations, std::vector<float> objProbs, std::vector<int> classIds, std::vector<int>& order,
    int filterId, float threshold){
    // 优化点1：双指针+滑动窗口性质+YOLO检测结果性质，简化IoU计算
    // int invalid_count = 0;
    int i = 0, j = 1;
    while (i < validCount && j < validCount){
        while (i < validCount && (order[i] == -1 || classIds[order[i]] != filterId)) i++;  // 找到最小索引满足
        while (j < validCount && (order[j] == -1 || classIds[order[j]] != filterId)) j++;  // 找到移动索引满足
        if (j >= validCount) break;
        
        int n = order[i];

        while (j < validCount && order[j] != -1 && classIds[order[j]] == filterId){
            int m = order[j];

            // 计算两个框的坐标差值，则选择置信度较低的框置为 -1
            if (
                fabs(outputLocations[n * 5 + 0] - outputLocations[m * 5 + 0]) < 1.5 &&  // x
                fabs(outputLocations[n * 5 + 1] - outputLocations[m * 5 + 1]) < 1.5 &&  // y
                fabs(outputLocations[n * 5 + 2] - outputLocations[m * 5 + 2]) < 2.0 &&  // w
                fabs(outputLocations[n * 5 + 3] - outputLocations[m * 5 + 3]) < 2.0     // h
            ) {
                if (objProbs[i] >= objProbs[j]) {
                    order[j] = -1;  // 置信度较低的框置为无效
                    // invalid_count++;
                    j++;
                } else {
                    order[i] = -1;
                    // invalid_count++;
                    i = j;  // 当前最小置信度低导致无效，修改为移动索引
                    j = i + 1;
                    break;
                }
            }else{
                i = j;  // 当前最小和下一个不同导致无效，修改为移动索引
                j = i + 1;
                break;
            }
        }
    }
    // printf("total valid count: %d, exclude invalid count: %d\n", validCount, invalid_count);

    for (int i = 0; i < validCount; ++i){
        int n = order[i];
        if (n == -1 || classIds[n] != filterId){
            continue;
        }
        for (int j = i + 1; j < validCount; ++j){
            int m = order[j];
            if (m == -1 || classIds[m] != filterId){
                continue;
            }
            float xmin0 = outputLocations[n * 5 + 0];
            float ymin0 = outputLocations[n * 5 + 1];
            float xmax0 = outputLocations[n * 5 + 0] + outputLocations[n * 5 + 2];
            float ymax0 = outputLocations[n * 5 + 1] + outputLocations[n * 5 + 3];

            float xmin1 = outputLocations[m * 5 + 0];
            float ymin1 = outputLocations[m * 5 + 1];
            float xmax1 = outputLocations[m * 5 + 0] + outputLocations[m * 5 + 2];
            float ymax1 = outputLocations[m * 5 + 1] + outputLocations[m * 5 + 3];

            // 优化点2：两个框没有交集，直接跳过
            if (xmin0 > xmax1 || xmax0 < xmin1 || ymin0 > ymax1 || ymax0 < ymin1) {
                continue;
            }

            float iou = CalculateOverlap_hw(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            // 优化点3：引入置信度，提早判断无效，并选择高置信度结果
            if (iou > threshold){
                if (objProbs[i] >= objProbs[j]){
                    order[j] = -1;
                }else{
                    order[i] = -1;
                    break;
                }
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse_hw(std::vector<float> &input, int left, int right, std::vector<int> &indices){
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right){
        key_index = indices[left];
        key = input[left];
        while (low < high){
            while (low < high && input[high] <= key){
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key){
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse_hw(input, left, low - 1, indices);
        quick_sort_indice_inverse_hw(input, low + 1, right, indices);
    }
    return low;
}

void sort_with_indices(std::vector<float>& input, std::vector<int>& indices) {
    // 初始化 indices 数组
    indices.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
        indices[i] = i; // 填充索引为 0, 1, 2, ...
    }

    // 使用 std::sort 对 input 和 indices 同时排序
    std::sort(indices.begin(), indices.end(), [&input](int i1, int i2) {
        return input[i1] > input[i2]; // 按照降序排列
    });

    // 根据排序后的 indices 调整 input 的顺序
    std::vector<float> sorted_input(input.size());
    for (int i = 0; i < input.size(); ++i) {
        sorted_input[i] = input[indices[i]];
    }
    input = std::move(sorted_input); // 更新 input
}


static float sigmoid_hw(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid_hw(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip_hw(float val, float min, float max){
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine_hw(float f32, int32_t zp, float scale){
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip_hw(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8_hw(float f32, int32_t zp, float scale){
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip_hw(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32_hw(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float deqnt_affine_u8_to_f32_hw(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static void compute_dfl_hw(float* tensor, int dfl_len, float* box){
    for (int b = 0; b < 4; b++){
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i=0; i < dfl_len; i++){
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i] / exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int process_u8_hw(uint8_t *box_tensor, int32_t box_zp, float box_scale,
                        uint8_t *score_tensor, int32_t score_zp, float score_scale,
                        uint8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold, int obj_class_num=1){
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    uint8_t score_thres_u8 = qnt_f32_to_affine_u8_hw(threshold, score_zp, score_scale);
    uint8_t score_sum_thres_u8 = qnt_f32_to_affine_u8_hw(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++){
        for (int j = 0; j < grid_w; j++){
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // Use score sum to quickly filter
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_u8){
                    continue;
                }
            }

            uint8_t max_score = -score_zp;
            for (int c = 0; c < obj_class_num; c++){
                int score_offset = i* grid_w + j + c*grid_len;
                if (score_tensor[score_offset] > score_thres_u8){
                    max_score = score_tensor[score_offset];
                    max_class_id = c;
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    for (int k = 0; k < dfl_len * 4; k++){
                        before_dfl[k] = deqnt_affine_u8_to_f32_hw(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    compute_dfl_hw(before_dfl, dfl_len, box);

                    float x1, y1, x2, y2, w, h;
                    x1 = (-box[0] + j + 0.5) * stride;
                    y1 = (-box[1] + i + 0.5) * stride;
                    x2 = (box[2] + j + 0.5) * stride;
                    y2 = (box[3] + i + 0.5) * stride;
                    w = x2 - x1;
                    h = y2 - y1;
                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);

                    objProbs.push_back(deqnt_affine_u8_to_f32_hw(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_u8_obb_hw(uint8_t* input, uint8_t* angle_feature, int grid_h, int grid_w, int stride,
    std::vector<float>& boxes, std::vector<float>& boxScores, std::vector<int>& classId, float threshold,
    int32_t zp, float scale, int32_t angle_feature_zp, float angle_feature_scale, int index, int obj_class_num=1) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + obj_class_num;
    int validCount = 0;

    uint8_t thres_i8 = qnt_f32_to_affine_u8_hw(unsigmoid_hw(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < obj_class_num; a++) {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid_hw(deqnt_affine_u8_to_f32_hw(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w],
                        zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_u8_to_f32_hw(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax_hw(&loc[i * 16], 16);
                    }
                    float xywh_[4] = { 0, 0, 0, 0 };
                    float xywh[4] = { 0, 0, 0, 0 };
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }

                    float xywh_add[2], xywh_sub[2];
                    xywh_add[0] = xywh_[0] + xywh_[2];
                    xywh_add[1] = xywh_[1] + xywh_[3];
                    xywh_sub[0] = (xywh_[2] - xywh_[0]) / 2;
                    xywh_sub[1] = (xywh_[3] - xywh_[1]) / 2;
                    float angle_feature_ = deqnt_affine_u8_to_f32_hw(angle_feature[index + (h * grid_w) + w], angle_feature_zp, angle_feature_scale);
                    angle_feature_ = (angle_feature_ - 0.25) * 3.1415927410125732;
                    float angle_feature_cos = cos(angle_feature_);
                    float angle_feature_sin = sin(angle_feature_);
                    float xy_mul1 = xywh_sub[0] * angle_feature_cos;
                    float xy_mul2 = xywh_sub[1] * angle_feature_sin;
                    float xy_mul3 = xywh_sub[0] * angle_feature_sin;
                    float xy_mul4 = xywh_sub[1] * angle_feature_cos;
                    xywh_[0] = ((xy_mul1 - xy_mul2) + w + 0.5) * stride;
                    xywh_[1] = ((xy_mul3 + xy_mul4) + h + 0.5) * stride;
                    xywh_[2] = xywh_add[0] * stride;
                    xywh_[3] = xywh_add[1] * stride;
                    xywh[0] = (xywh_[0] - xywh_[2] / 2);
                    xywh[1] = (xywh_[1] - xywh_[3] / 2);
                    xywh[2] = xywh_[2];
                    xywh[3] = xywh_[3];
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(angle_feature_);//angle
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_u8_pose_hw(uint8_t* input, int grid_h, int grid_w, int stride,
    std::vector<float>& boxes, std::vector<float>& boxScores, std::vector<int>& classId, float threshold,
    int32_t zp, float scale, int index, int obj_class_num=1) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + obj_class_num;
    int validCount = 0;

    uint8_t thres_i8 = qnt_f32_to_affine_u8_hw(unsigmoid_hw(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < obj_class_num; a++) {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid_hw(deqnt_affine_u8_to_f32_hw(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w], zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_u8_to_f32_hw(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax_hw(&loc[i * 16], 16);
                    }
                    float xywh_[4] = { 0, 0, 0, 0 };
                    float xywh[4] = { 0, 0, 0, 0 };
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0] = (w + 0.5) - xywh_[0];
                    xywh_[1] = (h + 0.5) - xywh_[1];
                    xywh_[2] = (w + 0.5) + xywh_[2];
                    xywh_[3] = (h + 0.5) + xywh_[3];
                    xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                    xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                    xywh[0] = xywh[0] - xywh[2] / 2;
                    xywh[1] = xywh[1] - xywh[3] / 2;
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_i8_hw(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold, int obj_class_num=1){
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine_hw(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine_hw(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++){
        for (int j = 0; j < grid_w; j++){
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c= 0; c< obj_class_num; c++){
                int score_offset = i* grid_w + j + c*grid_len;
                if (score_tensor[score_offset] > score_thres_i8){
                    max_score = score_tensor[score_offset];
                    max_class_id = c;
                    float box[4];
                    float before_dfl[dfl_len*4];
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = deqnt_affine_to_f32_hw(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    compute_dfl_hw(before_dfl, dfl_len, box);

                    float x1,y1,x2,y2,w,h;
                    x1 = (-box[0] + j + 0.5)*stride;
                    y1 = (-box[1] + i + 0.5)*stride;
                    x2 = (box[2] + j + 0.5)*stride;
                    y2 = (box[3] + i + 0.5)*stride;
                    w = x2 - x1;
                    h = y2 - y1;
                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);

                    objProbs.push_back(deqnt_affine_to_f32_hw(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount ++;
                }
            }
        }
    }
    return validCount;
}

static int process_i8_obb_hw(int8_t* input, int8_t* angle_feature, int grid_h, int grid_w, int stride,
    std::vector<float>& boxes, std::vector<float>& boxScores, std::vector<int>& classId, float threshold,
    int32_t zp, float scale, int32_t angle_feature_zp, float angle_feature_scale, int index, int obj_class_num=1) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + obj_class_num;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine_hw(unsigmoid_hw(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < obj_class_num; a++) {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid_hw(deqnt_affine_to_f32_hw(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w],
                        zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32_hw(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax_hw(&loc[i * 16], 16);
                    }
                    float xywh_[4] = { 0, 0, 0, 0 };
                    float xywh[4] = { 0, 0, 0, 0 };
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }

                    float xywh_add[2], xywh_sub[2];
                    xywh_add[0] = xywh_[0] + xywh_[2];
                    xywh_add[1] = xywh_[1] + xywh_[3];
                    xywh_sub[0] = (xywh_[2] - xywh_[0]) / 2;
                    xywh_sub[1] = (xywh_[3] - xywh_[1]) / 2;
                    float angle_feature_ = deqnt_affine_to_f32_hw(angle_feature[index + (h * grid_w) + w], angle_feature_zp, angle_feature_scale);
                    angle_feature_ = (angle_feature_ - 0.25) * 3.1415927410125732;
                    float angle_feature_cos = cos(angle_feature_);
                    float angle_feature_sin = sin(angle_feature_);
                    float xy_mul1 = xywh_sub[0] * angle_feature_cos;
                    float xy_mul2 = xywh_sub[1] * angle_feature_sin;
                    float xy_mul3 = xywh_sub[0] * angle_feature_sin;
                    float xy_mul4 = xywh_sub[1] * angle_feature_cos;
                    xywh_[0] = ((xy_mul1 - xy_mul2) + w + 0.5) * stride;
                    xywh_[1] = ((xy_mul3 + xy_mul4) + h + 0.5) * stride;
                    xywh_[2] = xywh_add[0] * stride;
                    xywh_[3] = xywh_add[1] * stride;
                    xywh[0] = (xywh_[0] - xywh_[2] / 2);
                    xywh[1] = (xywh_[1] - xywh_[3] / 2);
                    xywh[2] = xywh_[2];
                    xywh[3] = xywh_[3];
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(angle_feature_);//angle
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_i8_pose_hw(int8_t* input, int grid_h, int grid_w, int stride,
    std::vector<float>& boxes, std::vector<float>& boxScores, std::vector<int>& classId, float threshold,
    int32_t zp, float scale, int index, int obj_class_num) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + obj_class_num;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine_hw(unsigmoid_hw(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < obj_class_num; a++) {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_i8) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid_hw(deqnt_affine_to_f32_hw(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w], zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32_hw(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax_hw(&loc[i * 16], 16);
                    }
                    float xywh_[4] = { 0, 0, 0, 0 };
                    float xywh[4] = { 0, 0, 0, 0 };
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0] = (w + 0.5) - xywh_[0];
                    xywh_[1] = (h + 0.5) - xywh_[1];
                    xywh_[2] = (w + 0.5) + xywh_[2];
                    xywh_[3] = (h + 0.5) + xywh_[3];
                    xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                    xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                    xywh[0] = xywh[0] - xywh[2] / 2;
                    xywh[1] = xywh[1] - xywh[3] / 2;
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_fp32_hw(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        std::vector<int> &classId, 
                        float threshold, int obj_class_num=1){
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++){
        for (int j = 0; j < grid_w; j++){
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c < obj_class_num; c++){
                int score_offset = i* grid_w + j + c*grid_len;
                if (score_tensor[score_offset] > threshold){
                    max_score = score_tensor[score_offset];
                    max_class_id = c;
                    float box[4];
                    float before_dfl[dfl_len*4];
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = box_tensor[offset];
                        offset += grid_len;
                    }
                    compute_dfl_hw(before_dfl, dfl_len, box);

                    float x1,y1,x2,y2,w,h;
                    x1 = (-box[0] + j + 0.5)*stride;
                    y1 = (-box[1] + i + 0.5)*stride;
                    x2 = (box[2] + j + 0.5)*stride;
                    y2 = (box[3] + i + 0.5)*stride;
                    w = x2 - x1;
                    h = y2 - y1;
                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);

                    objProbs.push_back(max_score);
                    classId.push_back(max_class_id);
                    validCount ++;
                }
            }
        }
    }
    return validCount;
}


static int process_fp32_obb_hw(float* input, float* angle_feature, int grid_h, int grid_w, int stride,
    std::vector<float>& boxes, std::vector<float>& boxScores, std::vector<int>& classId, float threshold,
    int32_t zp, float scale, int32_t angle_feature_zp, float angle_feature_scale, int index, int obj_class_num=1) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + obj_class_num;
    int validCount = 0;

    float thres_fp32 = unsigmoid_hw(threshold);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < obj_class_num; a++) {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_fp32) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid_hw(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w]);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax_hw(&loc[i * 16], 16);
                    }
                    float xywh_[4] = { 0, 0, 0, 0 };
                    float xywh[4] = { 0, 0, 0, 0 };
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }

                    float xywh_add[2], xywh_sub[2];
                    xywh_add[0] = xywh_[0] + xywh_[2];
                    xywh_add[1] = xywh_[1] + xywh_[3];
                    xywh_sub[0] = (xywh_[2] - xywh_[0]) / 2;
                    xywh_sub[1] = (xywh_[3] - xywh_[1]) / 2;
                    float angle_feature_ = angle_feature[index + (h * grid_w) + w];
                    angle_feature_ = (angle_feature_ - 0.25) * 3.1415927410125732;
                    float angle_feature_cos = cos(angle_feature_);
                    float angle_feature_sin = sin(angle_feature_);
                    float xy_mul1 = xywh_sub[0] * angle_feature_cos;
                    float xy_mul2 = xywh_sub[1] * angle_feature_sin;
                    float xy_mul3 = xywh_sub[0] * angle_feature_sin;
                    float xy_mul4 = xywh_sub[1] * angle_feature_cos;
                    xywh_[0] = ((xy_mul1 - xy_mul2) + w + 0.5) * stride;
                    xywh_[1] = ((xy_mul3 + xy_mul4) + h + 0.5) * stride;
                    xywh_[2] = xywh_add[0] * stride;
                    xywh_[3] = xywh_add[1] * stride;
                    xywh[0] = (xywh_[0] - xywh_[2] / 2);
                    xywh[1] = (xywh_[1] - xywh_[3] / 2);
                    xywh[2] = xywh_[2];
                    xywh[3] = xywh_[3];
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(angle_feature_);//angle
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_fp32_pose_hw(float* input, int grid_h, int grid_w, int stride,
    std::vector<float>& boxes, std::vector<float>& boxScores, std::vector<int>& classId, float threshold,
    int32_t zp, float scale, int index, int obj_class_num=1) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + obj_class_num;
    int validCount = 0;
    float thres_fp = unsigmoid_hw(threshold);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < obj_class_num; a++) {
                if (input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w] >= thres_fp) { //[1,tensor_len,grid_h,grid_w]
                    float box_conf_f32 = sigmoid_hw(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w]);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax_hw(&loc[i * 16], 16);
                    }
                    float xywh_[4] = { 0, 0, 0, 0 };
                    float xywh[4] = { 0, 0, 0, 0 };
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0] = (w + 0.5) - xywh_[0];
                    xywh_[1] = (h + 0.5) - xywh_[1];
                    xywh_[2] = (w + 0.5) + xywh_[2];
                    xywh_[3] = (h + 0.5) + xywh_[3];
                    xywh[0] = ((xywh_[0] + xywh_[2]) / 2) * stride;
                    xywh[1] = ((xywh_[1] + xywh_[3]) / 2) * stride;
                    xywh[2] = (xywh_[2] - xywh_[0]) * stride;
                    xywh[3] = (xywh_[3] - xywh_[1]) * stride;
                    xywh[0] = xywh[0] - xywh[2] / 2;
                    xywh[1] = xywh[1] - xywh[3] / 2;
                    boxes.push_back(xywh[0]);//x
                    boxes.push_back(xywh[1]);//y
                    boxes.push_back(xywh[2]);//w
                    boxes.push_back(xywh[3]);//h
                    boxes.push_back(float(index + (h * grid_w) + w));//keypoints index
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int process_u8_v8_hw(uint8_t* box_tensor, int32_t box_zp, float box_scale,
    uint8_t* score_tensor, int32_t score_zp, float score_scale,
    uint8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& objProbs,
    std::vector<int>& classId,
    float threshold, int obj_class_num){
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    uint8_t score_thres_u8 = qnt_f32_to_affine_u8_hw(threshold, score_zp, score_scale);
    uint8_t score_sum_thres_u8 = qnt_f32_to_affine_u8_hw(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++){
        for (int j = 0; j < grid_w; j++){
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // Use score sum to quickly filter
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_u8){
                    continue;
                }
            }

            uint8_t max_score = -score_zp;
            for (int c = 0; c < obj_class_num; c++){
                if ((score_tensor[offset] > score_thres_u8) && (score_tensor[offset] > max_score)){
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > score_thres_u8){
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++){
                    before_dfl[k] = deqnt_affine_u8_to_f32_hw(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl_hw(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_u8_to_f32_hw(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

static int process_i8_v8_hw(int8_t* box_tensor, int32_t box_zp, float box_scale,
    int8_t* score_tensor, int32_t score_zp, float score_scale,
    int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& objProbs,
    std::vector<int>& classId,
    float threshold, int obj_class_num=1){
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine_hw(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine_hw(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++){
        for (int j = 0; j < grid_w; j++){
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c = 0; c < obj_class_num; c++) {
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score)){
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > score_thres_i8) {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = deqnt_affine_to_f32_hw(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl_hw(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32_hw(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

static int process_fp32_v8_hw(float* box_tensor, float* score_tensor, float* score_sum_tensor,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& objProbs,
    std::vector<int>& classId,
    float threshold, int obj_class_num=1){
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++){
        for (int j = 0; j < grid_w; j++){
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < threshold) {
                    continue;
                }
            }

            float max_score = 0;
            for (int c = 0; c < obj_class_num; c++) {
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > threshold) {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl_hw(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

int post_process_hw(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, 
    object_detect_result_list *od_results, int obj_class_num){
    rknn_output *_outputs = (rknn_output *)outputs;
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    // default 3 branch
    int dfl_len = app_ctx->output_attrs[0].dims[1] /4;
    int output_per_branch = app_ctx->io_num.n_output / 3;
    for (int i = 0; i < 3; i++){
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = app_ctx->output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;

        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];
        stride = model_in_h / grid_h;

        if (app_ctx->is_quant){
            validCount += process_i8_hw((int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                     (int8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len, 
                                     filterBoxes, objProbs, classId, conf_threshold, obj_class_num);
        }
        else{
            validCount += process_fp32_hw((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, 
                                       filterBoxes, objProbs, classId, conf_threshold, obj_class_num);
        }
    }

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i){
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse_hw(objProbs, 0, validCount - 1, indexArray);

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i){
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE){
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp_hw(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp_hw(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp_hw(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp_hw(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int post_process_det_hw(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, 
    object_detect_result_list* od_results, int obj_class_num){
    rknn_output* _outputs = (rknn_output*)outputs;

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    // default 3 branch
    int dfl_len = app_ctx->output_attrs[0].dims[1] / 4;
    int output_per_branch = app_ctx->io_num.n_output / 3;

    for (int i = 0; i < 3; i++){
        void* score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3) {
            score_sum = _outputs[i * output_per_branch + 2].buf;
            score_sum_zp = app_ctx->output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i * output_per_branch + 2].scale;
        }
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;

        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];

        stride = model_in_h / grid_h;

        if (app_ctx->is_quant){
            validCount += process_i8_v8_hw(
                (int8_t*)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                (int8_t*)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                (int8_t*)score_sum, score_sum_zp, score_sum_scale,
                grid_h, grid_w, stride, dfl_len,
                filterBoxes, objProbs, classId, conf_threshold, obj_class_num);
        }
        else{
            validCount += process_fp32_v8_hw(
                (float*)_outputs[box_idx].buf, 
                (float*)_outputs[score_idx].buf, 
                (float*)score_sum,
                grid_h, grid_w, stride, dfl_len, 
                filterBoxes, objProbs, classId, conf_threshold, obj_class_num);
        }
    }

    // no object detect
    if (validCount <= 0){
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i){
        indexArray.push_back(i);
    }
    
    // add prob judge, no longer need sort
    // quick_sort_indice_inverse_hw(objProbs, 0, validCount - 1, indexArray);
    // sort_with_indices(objProbs, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set){
        nms_hw(validCount, filterBoxes, objProbs, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE){
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp_hw(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp_hw(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp_hw(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp_hw(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int post_process_obb_hw(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, 
    object_detect_obb_result_list* od_results, int obj_class_num) {
    rknn_output* _outputs = (rknn_output*)outputs;

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    memset(od_results, 0, sizeof(object_detect_obb_result_list));
    int index = 0;


    for (int i = 0; i < 3; i++) {
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant) {
            validCount += process_i8_obb_hw((int8_t*)_outputs[i].buf, (int8_t*)_outputs[3].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale,
                app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale, index, obj_class_num);
        }
        else {
            validCount += process_fp32_obb_hw((float*)_outputs[i].buf, (float*)_outputs[3].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale,
                app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale, index, obj_class_num);
        }
        index += grid_h * grid_w;
    }


   // std::cout << "process_i8_obb_hw" << std::endl;
    // no object detect
    if (validCount <= 0) {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse_hw(objProbs, 0, validCount - 1, indexArray);

   // std::cout << "quick_sort_indice_inverse_hw" << std::endl;
    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set) {
        nms_obb_hw(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    //std::cout << "nms_obb_hw" << std::endl;

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 5 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 5 + 1] - letter_box->y_pad;
        float w = filterBoxes[n * 5 + 2];
        float h = filterBoxes[n * 5 + 3];
        float angle = filterBoxes[n * 5 + 4];

        // std::vector<float> rbbox_to_corners(const std::vector<float> &rbbox)

        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.x = (int)(clamp_hw(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.y = (int)(clamp_hw(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.w = (int)(clamp_hw(w, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.h = (int)(clamp_hw(h, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.angle = angle;
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int post_process_pose_hw(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold,
    object_detect_pose_result_list* od_results, int obj_class_num, int kpt_num, int result_num) {
    rknn_output* _outputs = (rknn_output*)outputs;

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    memset(od_results, 0, sizeof(object_detect_result_list));
    int index = 0;

    for (int i = 0; i < 3; i++) {
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant) {
            validCount += process_i8_pose_hw((int8_t*)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index, obj_class_num);
        }
        else{
            validCount += process_fp32_pose_hw((float*)_outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale, index, obj_class_num);
        }
        index += grid_h * grid_w;
    }

    // no object detect
    if (validCount <= 0) {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    
    // quick_sort_indice_inverse_hw(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set) {
        nms_pose_hw(validCount, filterBoxes, objProbs, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];
        float x1 = filterBoxes[n * 5 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 5 + 1] - letter_box->y_pad;
        float w = filterBoxes[n * 5 + 2];
        float h = filterBoxes[n * 5 + 3];
        int keypoints_index = (int)filterBoxes[n * 5 + 4];

        for (int j = 0; j < kpt_num; ++j) {
            if (app_ctx->is_quant) {
                od_results->results[last_count].keypoints[j][0] = ((float)((rknpu2::float16*)_outputs[3].buf)[j * 3 * result_num + 0 * result_num + keypoints_index]
                    - letter_box->x_pad) / letter_box->scale;
                od_results->results[last_count].keypoints[j][1] = ((float)((rknpu2::float16*)_outputs[3].buf)[j * 3 * result_num + 1 * result_num + keypoints_index]
                    - letter_box->y_pad) / letter_box->scale;
                od_results->results[last_count].keypoints[j][2] = (float)((rknpu2::float16*)_outputs[3].buf)[j * 3 * result_num + 2 * result_num + keypoints_index];
            }
            else
            {
                od_results->results[last_count].keypoints[j][0] = (((float*)_outputs[3].buf)[j * 3 * result_num + 0 * result_num + keypoints_index]
                    - letter_box->x_pad) / letter_box->scale;
                od_results->results[last_count].keypoints[j][1] = (((float*)_outputs[3].buf)[j * 3 * result_num + 1 * result_num + keypoints_index]
                    - letter_box->y_pad) / letter_box->scale;
                od_results->results[last_count].keypoints[j][2] = ((float*)_outputs[3].buf)[j * 3 * result_num + 2 * result_num + keypoints_index];
            }
        }

        int id = classId[n];
        float obj_conf = objProbs[i];
        od_results->results[last_count].box.left = (int)(clamp_hw(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp_hw(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp_hw(x1 + w, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp_hw(y1 + h, 0, model_in_h) / letter_box->scale);
        // od_results->results[last_count].box.angle = angle;
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

int init_post_process_hw(char *label_name_txt_path, char *labels[], int obj_class_num){
    int ret = 0;
    ret = loadLabelName_hw(label_name_txt_path, labels, obj_class_num);
    if (ret < 0)
    {
        printf("Load %s failed!\n", label_name_txt_path);
        return -1;
    }
    return 0;
}

int init_post_process_hw(const char* label_txt_path, char* labels[], int obj_class_num){
    int ret = 0;
    ret = loadLabelName_hw(label_txt_path, labels, obj_class_num);
    if (ret < 0)
    {
        printf("Load %s failed!\n", label_txt_path);
        return -1;
    }
    return 0;
}

const char* dataset_cls_to_name_hw(char* labels[], int cls_id, int obj_class_num){
    if (cls_id >= obj_class_num){
        return "null";
    }

    if (labels[cls_id]){
        return labels[cls_id];
    }
    return "null";
}

void deinit_post_process(char* labels[], int cls_id, int obj_class_num){
    for (int i = 0; i < obj_class_num; i++){
        if (labels[i] != nullptr){
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
