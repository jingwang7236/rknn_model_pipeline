#include "rec_image_process.h"

std::unique_ptr<cv::Mat> Resize(const cv::Mat& src, int target_width, int target_height) {
	// 检查输入图像是否为空，如果为空，则返回空指针
	if (src.empty()) {
		return nullptr;
	}

	// 新的宽高
	cv::Size new_size_ = cv::Size(static_cast<int>(target_width), static_cast<int>(target_height));

	// 创建 resize_img 来存储缩放后的图像
	cv::Mat resize_img;
	// 使用 OpenCV 的 resize 函数，将输入图像 src 按照 new_size_ 缩放
	cv::resize(src, resize_img, new_size_);

	// 将 resize_img 包装到 std::unique_ptr 中并返回
	return std::unique_ptr<cv::Mat>(new cv::Mat(resize_img));
}
//  resize 转化
std::unique_ptr<cv::Mat> Resize(const cv::Mat& src, int orignal_width, int orignal_height, int target_width, int target_height) {
	// 检查输入图像是否为空，如果为空，则返回空指针
	if (src.empty()) {
		return nullptr;
	}

	// 计算宽度和高度的缩放比例，并取较小的缩放比例
	double width_scale = static_cast<double>(target_width) / orignal_width;
	double height_scale = static_cast<double>(target_height) / orignal_height;
	double scale_ = std::min(width_scale, height_scale);

	// 根据缩放比例计算新的宽高
	cv::Size new_size_ = cv::Size(static_cast<int>(orignal_width * scale_), static_cast<int>(orignal_height * scale_));

	// 创建 resize_img 来存储缩放后的图像
	cv::Mat resize_img;
	// 使用 OpenCV 的 resize 函数，将输入图像 src 按照 new_size_ 缩放
	cv::resize(src, resize_img, new_size_);

	// 将 resize_img 包装到 std::unique_ptr 中并返回
	return std::unique_ptr<cv::Mat>(new cv::Mat(resize_img));
}

// const char* 转 cv::mat
cv::Mat convertToCvMat(const det_model_input& input) {
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
