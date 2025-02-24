#include "rec_image_process.h"

std::unique_ptr<cv::Mat> Resize(const cv::Mat& src, int target_width, int target_height) {
	// �������ͼ���Ƿ�Ϊ�գ����Ϊ�գ��򷵻ؿ�ָ��
	if (src.empty()) {
		return nullptr;
	}

	// �µĿ��
	cv::Size new_size_ = cv::Size(static_cast<int>(target_width), static_cast<int>(target_height));

	// ���� resize_img ���洢���ź��ͼ��
	cv::Mat resize_img;
	// ʹ�� OpenCV �� resize ������������ͼ�� src ���� new_size_ ����
	cv::resize(src, resize_img, new_size_);

	// �� resize_img ��װ�� std::unique_ptr �в�����
	return std::unique_ptr<cv::Mat>(new cv::Mat(resize_img));
}
//  resize ת��
std::unique_ptr<cv::Mat> Resize(const cv::Mat& src, int orignal_width, int orignal_height, int target_width, int target_height) {
	// �������ͼ���Ƿ�Ϊ�գ����Ϊ�գ��򷵻ؿ�ָ��
	if (src.empty()) {
		return nullptr;
	}

	// �����Ⱥ͸߶ȵ����ű�������ȡ��С�����ű���
	double width_scale = static_cast<double>(target_width) / orignal_width;
	double height_scale = static_cast<double>(target_height) / orignal_height;
	double scale_ = std::min(width_scale, height_scale);

	// �������ű��������µĿ��
	cv::Size new_size_ = cv::Size(static_cast<int>(orignal_width * scale_), static_cast<int>(orignal_height * scale_));

	// ���� resize_img ���洢���ź��ͼ��
	cv::Mat resize_img;
	// ʹ�� OpenCV �� resize ������������ͼ�� src ���� new_size_ ����
	cv::resize(src, resize_img, new_size_);

	// �� resize_img ��װ�� std::unique_ptr �в�����
	return std::unique_ptr<cv::Mat>(new cv::Mat(resize_img));
}

// const char* ת cv::mat
cv::Mat convertToCvMat(const det_model_input& input) {
	// ������������Ƿ���Ч
	if (input.data == nullptr || input.width <= 0 || input.height <= 0 || input.channel <= 0) {
		throw std::invalid_argument("Invalid input data for det_model_input.");
	}

	// ����ͨ�������� cv::Mat
	int type = (input.channel == 1) ? CV_8UC1 : (input.channel == 3 ? CV_8UC3 : CV_8UC4);
	cv::Mat img(input.height, input.width, type, input.data);

	// ����� 4 ͨ����RGBA����ת��Ϊ 3 ͨ����RGB��
	if (input.channel == 4) {
		cv::Mat rgb_img;
		cv::cvtColor(img, rgb_img, cv::COLOR_RGBA2RGB);  // �� RGBA ת��Ϊ RGB
		return rgb_img;
	}

	// ����� 1 �� 3 ͨ����ֱ�ӷ���
	return img;
}
