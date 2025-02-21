#include <stdio.h>
#include <iostream>
#include <string.h>

#include "model_func.hpp"
#include "model_params.hpp"
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <chrono>  // 计算耗时
using namespace std::chrono;

#include "yaml-cpp/yaml.h"

void print_rknn_app_context(const rknn_app_context_t& ctx) {
	std::cout << "rknn_ctx: " << ctx.rknn_ctx << std::endl;
	std::cout << "model_channel: " << ctx.model_channel << std::endl;
	std::cout << "model_width: " << ctx.model_width << std::endl;
	std::cout << "model_height: " << ctx.model_height << std::endl;
	std::cout << "is_quant: " << (ctx.is_quant ? "true" : "false") << std::endl;
}

/*-------------------------------------------
				Main  Functions
-------------------------------------------*/

int main(int argc, char** argv) {
	if (argc != 3) {
		printf("%s <model_name> <image_path>\n", argv[0]);
		return -1;
	}
	int ret;
	bool open_logger = true;

	const char* model_name = argv[1];
	const char* image_path = argv[2];  // single image path or testset file path

	// Load YAML configuration
	YAML::Node config = YAML::LoadFile("models.yaml");
	YAML::Node model_node = config["models"][model_name];
	// bool open_logger = config["enable_log"].as<bool>();

	if (!model_node) {
		std::cerr << "Unknown model_name: " << model_name << std::endl;
		throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
	}

	// 模型推理单张图像示例

	// Load image
	int width, height, channel;
	unsigned char* data = stbi_load(image_path, &width, &height, &channel, 3);
	if (data == NULL) {
		printf("Failed to load image from path: %s\n", image_path);
		return -1;
	}

	// init input data
	det_model_input input_data;

	input_data.data = data;
	input_data.width = width;
	input_data.height = height;
	input_data.channel = channel;

	// header det model
	if (std::string(model_name) == "header_det") {
		//    const char* model_path = "model/HeaderDet.rknn";
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		std::cout << "model path: " << model_path << std::endl;
		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rknn_app_ctx);  // 初始化
		if (ret != 0) {
			printf("init_retinanet_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_header_det_model(&rknn_app_ctx, input_data, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);  //释放
		if (ret != 0) {
			printf("release_retinanet_model fail! ret=%d\n", ret);
			return -1;
		}
	}
	else if (std::string(model_name) == "phone_det") {
		//    const char* model_path = "model/PhoneDet.rknn";
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rknn_app_ctx);  // 初始化
		if (ret != 0) {
			printf("init_retinanet_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_phone_det_model(&rknn_app_ctx, input_data, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);  //释放
		if (ret != 0) {
			printf("release_retinanet_model fail! ret=%d\n", ret);
			return -1;
		}
	}
	else if (std::string(model_name) == "face_det") {
		//    const char* model_path = "model/RetinaFace.rknn";
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rknn_app_ctx);  // 初始化
		if (ret != 0) {
			printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		auto start = std::chrono::high_resolution_clock::now();
		retinaface_result result = inference_face_det_model(&rknn_app_ctx, input_data, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);
		if (ret != 0) {
			printf("release_retinaface_model fail! ret=%d\n", ret);
			return -1;
		}
	}
	else if (std::string(model_name) == "coco_person_det") {
		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		//    const char* model_path = "model/yolov10s.rknn";
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		ret = init_model(model_path, &rknn_app_ctx);
		if (ret != 0)
		{
			printf("init_yolov10_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_coco_person_det_model(&rknn_app_ctx, input_data, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);
		if (ret != 0)
		{
			printf("release_yolov10_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "person_det") {
		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		//    const char* model_path = "model/PersonDet.rknn";
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		ret = init_model(model_path, &rknn_app_ctx);
		if (ret != 0)
		{
			printf("init_person_det_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_person_det_model(&rknn_app_ctx, input_data, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);
		if (ret != 0)
		{
			printf("release_person_det_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "det_knife") {
		
		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		
		model_inference_params params_det_knife;
		params_det_knife.input_height = model_node["infer_img_height"].as<int>();
		params_det_knife.input_width = model_node["infer_img_width"].as<int>();
		params_det_knife.nms_threshold = model_node["nms_threshold"].as<float>();
		params_det_knife.box_threshold = model_node["box_threshold"].as<float>();
		

		ret = init_model(model_path, &rknn_app_ctx);
		if (ret != 0)
		{
			printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		// print_rknn_app_context(rknn_app_ctx);
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_det_knife_model(&rknn_app_ctx, input_data, params_det_knife, false, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);
		if (ret != 0)
		{
			printf("release_yolov8_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "det_gun") {

		/* 获取模型推理时尺寸 */
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		
		/* 推理参数 width height nms_ths box_ths*/
		model_inference_params params_det_gun;
		params_det_gun.input_height = model_node["infer_img_height"].as<int>();
		params_det_gun.input_width = model_node["infer_img_width"].as<int>();
		params_det_gun.nms_threshold = model_node["nms_threshold"].as<float>();
		params_det_gun.box_threshold = model_node["box_threshold"].as<float>();
		
		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

		ret = init_model(model_path, &rknn_app_ctx);
		if (ret != 0)
		{
			printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		/* 量化标志 */
		rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();
		/* 日志标志 */
		// bool print_logs = open_logger && (model_node["log"].as<float>());

		//print_rknn_app_context(rknn_app_ctx);
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_det_gun_model(&rknn_app_ctx, input_data, params_det_gun, false, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);
		if (ret != 0)
		{
			printf("release_yolov8_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "det_stat_door") {
		/* 获得模型推理时参数 */
		// const char* model_path = "model/jhpoc_1225_stat_door_det2_640_rk.rknn";
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		
		model_inference_params params_det_stat_door;
		params_det_stat_door.input_height = model_node["infer_img_height"].as<int>();
		params_det_stat_door.input_width = model_node["infer_img_width"].as<int>();
		params_det_stat_door.nms_threshold = model_node["nms_threshold"].as<float>();
		params_det_stat_door.box_threshold = model_node["box_threshold"].as<float>();

		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

		ret = init_model(model_path, &rknn_app_ctx);
		if (ret != 0)
		{
			printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		print_rknn_app_context(rknn_app_ctx);
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_det_stat_door_model(&rknn_app_ctx, input_data, params_det_stat_door, false, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);
		if (ret != 0)
		{
			printf("release_yolov8_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "face_attr") {
		// 检测初始化
		printf("inference_face_attr_model start\n");
		// const char* det_model_path = "model/HeaderDet.rknn";
		const std::string det_model_path_str = model_node["det_path"].as<std::string>();
		const char* det_model_path = det_model_path_str.c_str();
		rknn_app_context_t det_rknn_app_ctx;
		memset(&det_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(det_model_path, &det_rknn_app_ctx);
		// 分类初始化
		rknn_app_context_t cls_rknn_app_ctx;
		memset(&cls_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		// const char* cls_model_path = "model/FaceAttr.rknn";
		const std::string cls_model_path_str = model_node["cls_path"].as<std::string>();
		const char* cls_model_path = cls_model_path_str.c_str();
		ret = init_model(cls_model_path, &cls_rknn_app_ctx);
		object_detect_result_list det_result = inference_header_det_model(&det_rknn_app_ctx, input_data, false); //头肩检测模型推理
		det_result.count = det_result.count;
		for (int i = 0; i < det_result.count; ++i) {
			box_rect header_box;  // header的box
			header_box.left = std::max(det_result.results[i].box.left, 0);
			header_box.top = std::max(det_result.results[i].box.top, 0);
			header_box.right = std::min(det_result.results[i].box.right, width);
			header_box.bottom = std::min(det_result.results[i].box.bottom, height);
			// 人脸属性模型
			auto cls_start = std::chrono::high_resolution_clock::now();
			cls_model_result cls_result = inference_face_attr_model(&cls_rknn_app_ctx, input_data, header_box, false);
			auto cls_end = std::chrono::high_resolution_clock::now();
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(cls_end - cls_start).count() / 1000.0);

		}
		ret = release_model(&det_rknn_app_ctx);  //释放

	}
	else if (std::string(model_name) == "ppocr") {
		// const char* det_model_path = "model/ppocrv4_det.rknn";
		// const char* rec_model_path = "model/ppocrv4_rec.rknn";
		const std::string det_model_path_str = model_node["det_path"].as<std::string>();
		const char* det_model_path = det_model_path_str.c_str();
		const std::string rec_model_path_str = model_node["rec_path"].as<std::string>();
		const char* rec_model_path = rec_model_path_str.c_str();
		ppocr_system_app_context rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(ppocr_system_app_context));
		ret = init_model(det_model_path, &rknn_app_ctx.det_context);
		ret = init_model(rec_model_path, &rknn_app_ctx.rec_context);
		auto start = std::chrono::high_resolution_clock::now();
		ppocr_text_recog_array_result_t results = inference_ppocr_det_rec_model(&rknn_app_ctx, input_data, false);
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx.det_context);
		ret = release_model(&rknn_app_ctx.rec_context);
	}
	else if (std::string(model_name) == "rec_ren") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();

		cls_model_inference_params params_rec_ren;
		params_rec_ren.top_k = 1;
		params_rec_ren.img_height = model_node["infer_img_height"].as<int>();
		params_rec_ren.img_width = model_node["infer_img_width"].as<int>();

		rknn_app_context_t rec_ren_rknn_app_ctx;
		memset(&rec_ren_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rec_ren_rknn_app_ctx);

		if (ret != 0)
		{
			printf("init_rec_ren_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		rec_ren_rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		auto rec_ren_start = std::chrono::high_resolution_clock::now();
		resnet_result rec_result = inference_rec_person_resnet18_model(&rec_ren_rknn_app_ctx, input_data, params_rec_ren, open_logger);
		auto rec_ren_end = std::chrono::high_resolution_clock::now();

		if (open_logger){
			std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
			printf("ren_ren cost time: %.2f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(rec_ren_end - rec_ren_start).count() / 1000.0);
		}
		ret = release_model(&rec_ren_rknn_app_ctx);
	}
	else if (std::string(model_name) == "rec_ren_mobilenet") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();

		cls_model_inference_params params_rec_ren_mobilenet;
		params_rec_ren_mobilenet.top_k = 1;
		params_rec_ren_mobilenet.img_height = model_node["infer_img_height"].as<int>();
		params_rec_ren_mobilenet.img_width = model_node["infer_img_width"].as<int>();

		rknn_app_context_t rec_ren_mobilenet_rknn_app_ctx;
		memset(&rec_ren_mobilenet_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rec_ren_mobilenet_rknn_app_ctx);

		if (ret != 0)
		{
			printf("init_rec_ren_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		rec_ren_mobilenet_rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		auto rec_ren_start = std::chrono::high_resolution_clock::now();
		mobilenet_result rec_result = inference_rec_person_mobilenet_model(&rec_ren_mobilenet_rknn_app_ctx, input_data, params_rec_ren_mobilenet, open_logger);
		auto rec_ren_end = std::chrono::high_resolution_clock::now();

		if (open_logger){
			std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
			printf("ren_ren_mobilenet cost time: %.2f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(rec_ren_end - rec_ren_start).count() / 1000.0);
		}
		ret = release_model(&rec_ren_mobilenet_rknn_app_ctx);
	}
	else if (std::string(model_name) == "det_hand") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		
		model_inference_params params_det_hand;
		params_det_hand.input_height = model_node["infer_img_height"].as<int>();
		params_det_hand.input_width = model_node["infer_img_width"].as<int>();
		params_det_hand.nms_threshold = model_node["nms_threshold"].as<float>();
		params_det_hand.box_threshold = model_node["box_threshold"].as<float>();

		rknn_app_context_t det_hand_rknn_app_ctx;
		memset(&det_hand_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &det_hand_rknn_app_ctx);

		if (ret != 0) {
			printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		det_hand_rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_det_hand_model(&det_hand_rknn_app_ctx, input_data, params_det_hand, open_logger); //推理
		auto end = std::chrono::high_resolution_clock::now();
		
		if (open_logger){
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		}
		
		ret = release_model(&det_hand_rknn_app_ctx);

		if (ret != 0) {
			printf("release_yolov8_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "det_kx") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		
		model_inference_params params_det_kx;
		params_det_kx.input_height = model_node["infer_img_height"].as<int>();
		params_det_kx.input_width = model_node["infer_img_width"].as<int>();
		params_det_kx.nms_threshold = model_node["nms_threshold"].as<float>();
		params_det_kx.box_threshold = model_node["box_threshold"].as<float>();

		rknn_app_context_t det_kx_rknn_app_ctx;
		memset(&det_kx_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &det_kx_rknn_app_ctx);

		if (ret != 0) {
			printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		det_kx_rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		auto start = std::chrono::high_resolution_clock::now();
		object_detect_result_list result = inference_det_kx_model(&det_kx_rknn_app_ctx, input_data, params_det_kx, open_logger); //推理
		auto end = std::chrono::high_resolution_clock::now();

		if (open_logger){
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		}

		ret = release_model(&det_kx_rknn_app_ctx);

		if (ret != 0) {
			printf("release_yolov8_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "rec_hand") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();

		cls_model_inference_params params_rec_hand;
		params_rec_hand.top_k = 1;
		params_rec_hand.img_height = model_node["infer_img_height"].as<int>();
		params_rec_hand.img_width = model_node["infer_img_width"].as<int>();

		rknn_app_context_t rec_hand_rknn_app_ctx;
		memset(&rec_hand_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rec_hand_rknn_app_ctx);

		if (ret != 0)
		{
			printf("init rec_hand model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		rec_hand_rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		auto start = std::chrono::high_resolution_clock::now();
		resnet_result rec_result = inference_rec_hand_resnet18_model(&rec_hand_rknn_app_ctx, input_data, params_rec_hand, open_logger);
		auto end = std::chrono::high_resolution_clock::now();
		
		if (open_logger){
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
			std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
		}
		
		ret = release_model(&rec_hand_rknn_app_ctx);
	}
	else if (std::string(model_name) == "pose_ren") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		
		// pose_model_inference_params params_pose_ren;
		// params_pose_ren.input_height = model_node["infer_img_height"].as<int>();
		// params_pose_ren.input_width = model_node["infer_img_width"].as<int>();
		// params_pose_ren.kpt_nums = model_node["kpt_nums"].as<int>();
		// params_pose_ren.nms_threshold = model_node["nms_threshold"].as<float>();
		// params_pose_ren.box_threshold = model_node["box_threshold"].as<float>();

		rknn_app_context_t pose_ren_rknn_app_ctx;
		memset(&pose_ren_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &pose_ren_rknn_app_ctx);

		if (ret != 0) {
			printf("init pose_ren model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		auto start = std::chrono::high_resolution_clock::now();
		object_detect_pose_result_list pose_result = inference_pose_ren_model(&pose_ren_rknn_app_ctx, input_data, open_logger);
		auto end = std::chrono::high_resolution_clock::now();
		
		if (open_logger){
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		}

		ret = release_model(&pose_ren_rknn_app_ctx);
	}
	else if (std::string(model_name) == "rec_kx_orient") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();

		cls_model_inference_params params_rec_kx_orient;
		params_rec_kx_orient.top_k = 1;
		params_rec_kx_orient.img_height = model_node["infer_img_height"].as<int>();
		params_rec_kx_orient.img_width = model_node["infer_img_width"].as<int>();

		rknn_app_context_t rec_kx_orient_rknn_app_ctx;
		memset(&rec_kx_orient_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rec_kx_orient_rknn_app_ctx);

		if (ret != 0)
		{
			printf("init rec_kx_orient model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		rec_kx_orient_rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		auto start = std::chrono::high_resolution_clock::now();
		resnet_result rec_result = inference_rec_kx_orient_resnet18_model(&rec_kx_orient_rknn_app_ctx, input_data, params_rec_kx_orient, open_logger);
		auto end = std::chrono::high_resolution_clock::now();

		if (open_logger){
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
			std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
		}
		
		ret = release_model(&rec_kx_orient_rknn_app_ctx);
	}
	else if (std::string(model_name) == "pose_kx_hp") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		
		rknn_app_context_t pose_kx_hp_rknn_app_ctx;
		memset(&pose_kx_hp_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &pose_kx_hp_rknn_app_ctx);

		if (ret != 0) {
			printf("init pose_kx_hp model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		auto start = std::chrono::high_resolution_clock::now();
		object_detect_pose_result_list pose_result = inference_pose_kx_hp_model(&pose_kx_hp_rknn_app_ctx, input_data, open_logger);
		auto end = std::chrono::high_resolution_clock::now();
		
		if (open_logger){
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		}

		ret = release_model(&pose_kx_hp_rknn_app_ctx);
	}
	else if (std::string(model_name) == "pose_kx_sz") {
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();

		rknn_app_context_t pose_kx_sz_rknn_app_ctx;
		memset(&pose_kx_sz_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &pose_kx_sz_rknn_app_ctx);

		if (ret != 0) {
			printf("init pose_kx_sz model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		auto start = std::chrono::high_resolution_clock::now();
		object_detect_pose_result_list pose_result = inference_pose_kx_sz_model(&pose_kx_sz_rknn_app_ctx, input_data, open_logger);
		auto end = std::chrono::high_resolution_clock::now();
		
		if (open_logger){
			printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		}

		ret = release_model(&pose_kx_sz_rknn_app_ctx);
	}
	else if (std::string(model_name) == "obb_stick") {

		// const char* model_path = "model/jhpoc_250109-test1_obb_stick_1024_i8.rknn";
		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		/* 获取模型推理时尺寸 */
		/* 推理参数 width height nms_ths box_ths*/
		// model_inference_params params_obb_stick = { model_img_height,model_img_width,nms_threshold,box_threshold };
		model_inference_params params_obb_stick;
		params_obb_stick.input_height = model_node["infer_img_height"].as<int>();
		params_obb_stick.input_width = model_node["infer_img_width"].as<int>();
		params_obb_stick.nms_threshold = model_node["nms_threshold"].as<float>();
		params_obb_stick.box_threshold = model_node["box_threshold"].as<float>();

		rknn_app_context_t rknn_app_ctx;
		memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

		ret = init_model(model_path, &rknn_app_ctx);
		if (ret != 0)
		{
			printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}

		rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();

		//print_rknn_app_context(rknn_app_ctx);
		auto start = std::chrono::high_resolution_clock::now();
		object_detect_obb_result_list result = inference_obb_stick_model(&rknn_app_ctx, input_data, params_obb_stick, false, false); //推理
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		ret = release_model(&rknn_app_ctx);
		if (ret != 0)
		{
			printf("release_yolov8_model fail! ret=%d\n", ret);
		}
	}
	else if (std::string(model_name) == "rec_stat_door") {
		// 分类初始化

		const std::string model_path_str = model_node["path"].as<std::string>();
		const char* model_path = model_path_str.c_str();
		/* 获取模型推理时尺寸 */
		
		cls_model_inference_params cls_stat_door;
		cls_stat_door.top_k = 1;
		cls_stat_door.img_height = model_node["infer_img_height"].as<int>();
		cls_stat_door.img_width = model_node["infer_img_width"].as<int>();

		rknn_app_context_t rec_rknn_app_ctx;
		memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
		ret = init_model(model_path, &rec_rknn_app_ctx);

		if (ret != 0)
		{
			printf("init_rec_stat_door_model fail! ret=%d model_path=%s\n", ret, model_path);
			return -1;
		}
		rec_rknn_app_ctx.is_quant = model_node["qnt"].as<bool>();
		//mobilenet_result inference_rec_stat_door_mobilenetv3_model(rknn_app_context_t* app_ctx, det_model_input input_data, bool enable_logger = false)
	   // mobilenet_result rec_result = inference_rec_stat_door_mobilenetv3_model(&rec_rknn_app_ctx, input_data, cls_stat_door, true);
		auto start = std::chrono::high_resolution_clock::now();
		resnet_result rec_result = inference_rec_stat_door_resnet18_model(&rec_rknn_app_ctx, input_data, cls_stat_door, false);
		auto end = std::chrono::high_resolution_clock::now();
		printf("time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
		std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
		ret = release_model(&rec_rknn_app_ctx);
	}
	else {
		std::cerr << "Unknown model_name: " << model_name << std::endl;
		throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
	}
	stbi_image_free(data);
	return 0;
}
