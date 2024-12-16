
## 1. 可执行程序和静态算法库编译方法

使用cmake编译，包含主目录和src目录的CMakeLists.txt文件，其中src目录的CMakeLists.txt文件用于编译静态算法库文件，主目录的CMakeLists.txt文件用于将main.cpp编译成可执行文件。

```
rm -r build
mkdir build
cd build
cmake ..
make -j16
make install  # 可以将相关文件拷贝到目标目录
```

## 2. 算法库使用说明

测试用例在rk3588_test_pipeline目录下，包含以下算法：

### 2.1. 目标检测
包含行人检测、人脸检测、头部检测、手机检测;

#### 2.1.1. 行人检测
模型结构：yolov10-coco

调用方法：
```
include "outer_model/model_func.hpp"
include "outer_model/model_params.hpp"

det_model_input input_data;  // 输入数据格式
input_data.data = data;   // rgb24格式
input_data.width = width;
input_data.height = height;
input_data.channel = channel;
rknn_app_context_t rknn_app_ctx;
memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
const char* model_path = "model/yolov10s.rknn";
ret = init_yolov10_model(model_path, &rknn_app_ctx);  // 初始化模型
object_detect_result_list result = inference_person_det_model(&rknn_app_ctx, input_data, true); //模型推理
ret = release_yolov10_model(&rknn_app_ctx); // 释放模型资源
```

#### 2.1.2. 人脸检测

模型结构：RetinaFace

调用方法：
```
include "outer_model/model_func.hpp"
include "outer_model/model_params.hpp"

const char* model_path = "model/RetinaFace.rknn";
rknn_app_context_t rknn_app_ctx;
memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
ret = init_retinaface_model(model_path, &rknn_app_ctx);  // 初始化
retinaface_result result = inference_face_det_model(&rknn_app_ctx, input_data, true); //推理
ret = release_retinaface_model(&rknn_app_ctx);  //释放
```

#### 2.1.3. 头肩检测

模型结构：Retinanet, RepVgg

模型输出：头部的box

调用方法：
```
include "outer_model/model_func.hpp"
include "outer_model/model_params.hpp"

const char* model_path = "model/HeaderDet.rknn";
rknn_app_context_t rknn_app_ctx;
memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
ret = init_retinanet_model(model_path, &rknn_app_ctx);  // 初始化
ssd_det_result result = inference_header_det_model(&rknn_app_ctx, input_data, true); //推理
ret = release_retinanet_model(&rknn_app_ctx);  //释放
```

#### 2.1.4. 手机检测

模型结构：Retinanet, RepVgg

模型输出：亮屏手机的box

调用方法：
```
include "outer_model/model_func.hpp"
include "outer_model/model_params.hpp"

const char* model_path = "model/PhoneDet.rknn";
rknn_app_context_t rknn_app_ctx;
memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
ret = init_retinanet_model(model_path, &rknn_app_ctx);  // 初始化
ssd_det_result result = inference_phone_det_model(&rknn_app_ctx, input_data, true); //推理
ret = release_retinanet_model(&rknn_app_ctx);  //释放
```


### 2.2. 分类
包含人脸属性；

#### 2.2.1. 人脸属性
模型结构：RepVgg

模型输出：[int, int, int], 依次表示人脸是否被遮挡的三个属性;

（1）是否带帽子或头盔（0-不带帽子、1-带帽子、2-戴头盔）;

（2）是否带墨镜（0-不带墨镜、1-带墨镜）;

（3）是否戴口罩（0-不带口罩、1-戴口罩）；


调用方法：
```
include "outer_model/model_func.hpp"
include "outer_model/model_params.hpp"

rknn_app_context_t rknn_app_ctx;
memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
const char* model_path = "model/FaceAttr.rknn";
ret = init_classify_model(model_path, &rknn_app_ctx); //初始化
face_attr_cls_object result = inference_face_attr_model(&rknn_app_ctx, input_data, true); //推理
ret = release_classify_model(&rknn_app_ctx); // 释放
```

### 2.3. 业务逻辑

#### 2.3.1. 可疑人员检测
模型组合：头部检测 + 人脸属性；输入为场景大图，先经过头部检测模型，再将检测到的小图经过人脸属性模型推理，得到人脸属性结果；

输出结果: 头部box和对应的人脸属性结果；

调用方法：
```
include "outer_model/model_func.hpp"
include "outer_model/model_params.hpp"

// 检测初始化
const char* det_model_path = "model/HeaderDet.rknn";
rknn_app_context_t det_rknn_app_ctx;
memset(&det_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
ret = init_retinanet_model(det_model_path, &det_rknn_app_ctx);  
// 分类初始化
rknn_app_context_t cls_rknn_app_ctx;
memset(&cls_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
const char* cls_model_path = "model/FaceAttr.rknn";
ret = init_classify_model(cls_model_path, &cls_rknn_app_ctx);

face_det_attr_result result = inference_face_det_attr_model(&det_rknn_app_ctx, &cls_rknn_app_ctx, input_data, true); //推理

ret = release_retinanet_model(&det_rknn_app_ctx);  //释放
ret = release_classify_model(&cls_rknn_app_ctx);
```


## 4. 项目结构说明

需要提供给外部使用的函数接口：初始化模型、模型资源释放、模型推理；

### 目录说明

```
rknn_model_pipeline/
├── build/  # 编译输出目录
├── doc/    # 详细文档目录，如算法原理、参数说明等
├── model/  # 模型文件目录
│   └── yolov10s.rknn  # 示例模型文件
├── src/    # 源代码目录
│   ├── CMakeLists.txt  # 用于编译静态算法库的CMake配置文件
|   ├── business_pipeline/  # 业务逻辑相关源代码
│   ├── classify_model/ # 分类模型相关源代码
│   ├── ssd_det/        # ssd检测模型相关源代码
│   ├── yolo_det/       # yolo检测模型相关源代码
│   ├── utils/          # 工具函数相关源代码
├── CMakeLists.txt  # 用于编译可执行文件的CMake配置文件
├── main.cpp  # 主程序入口文件
├── README.md  # 项目说明文档
└── rk3588_test_pipeline/  # 测试用例目录
    └── ...  # 测试用例文件
```

