
## 可执行程序编译方法

使用cmake编译，包含主目录和src目录的CMakeLists.txt文件，其中src目录的CMakeLists.txt文件用于编译成静态算法库文件，主目录的CMakeLists.txt文件用于将main.cpp编译成可执行文件。

```
rm -r build
mkdir build
cd build
cmake ..
make -j16
make install  # 可以将相关文件拷贝到目标目录
```

## 算法库使用说明

测试用例在rk3588_test_pipeline目录下，包含以下算法：

### 目标检测
包含行人检测、人脸检测、头部检测、手机检测;

#### 行人检测
模型结构：yolov10
调用方法：
```
det_model_input input_data;  // 输入数据格式
input_data.data = data;   # rgb24格式
input_data.width = width;
input_data.height = height;
input_data.channel = channel;
rknn_app_context_t rknn_app_ctx;
memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
const char* model_path = "model/yolov10s.rknn";
ret = init_yolov10_model(model_path, &rknn_app_ctx);  // 初始化模型
object_detect_result_list result = inference_person_det_model(&rknn_app_ctx, input_data, true); //推理
ret = release_yolov10_model(&rknn_app_ctx); // 释放模型资源
```

#### 人脸检测



### 分类
人脸属性；

### 业务逻辑
人脸属性：帽子、头盔、墨镜、口罩；



### 人脸检测+人脸属性

输入参数：  
1. ./rknn_model_pipeline, so程序入口;
2. face_det_attr , 表示图片经过人脸检测和人脸属性算法，得到结果；
3. data/test.png , 图片路径

输出参数：

face det 0: left: 571, top: 128, right: 612, bottom: 180, score: 0.537109

face attr 0: hat 1, glass 0, mask 0

其中，face det表示人脸检测，0表示大图中第0个人脸，后续依次是x1y1x2y2四个点的左边，以及人脸检测分数；

face attr表示人脸属性，0表示大图中第0个人脸，后续依次是三个属性（帽子、眼镜和口罩）的输出值；

### 行人检测

输入参数：  
1. ./rknn_model_pipeline, so程序入口;
2. person_det , 表示图片使用行人检测算法；
3. data/test.png , 图片路径

输出参数：

person result count: 4

person det 0: left: 955, top: 139, right: 1339, bottom: 921, score: 0.907376

其中，person det表示人脸检测，0表示大图中第0个行人，后续依次是x1y1x2y2四个点的左边，以及行人检测分数；


### 行人检测+行人属性

输入参数：  
1. ./rknn_model_pipeline, so程序入口;
2. person_det_attr , 表示图片经过人脸检测和人脸属性算法，得到结果；
3. data/test.png , 图片路径

输出参数：

person det 0: left: 571, top: 128, right: 612, bottom: 180, score: 0.537109

person attr 0: hat 1, glass 0, mask 0

person det表示人脸检测，0表示大图中第0个行人，后续依次是x1y1x2y2四个点的左边，以及检测分数；

person attr表示人脸属性，0表示大图中第0个行人，后续依次是两个属性（帽子、是否携带物品）的输出值；


### 手机检测

输入参数：  
1. ./rknn_model_pipeline, so程序入口;
2. phone_det , 表示图片使用行人检测算法；
3. data/phone.png , 图片路径

输出参数：

phone @(1368 309 1392 369) score=0.645394
Draw result on result.png is finished.
phone result count: 1
phone 0: left: 1368, top: 309, right: 1392, bottom: 369, score: 0.645394

其中，phone det表示手机检测，0表示大图中第0个手机，后续依次是x1y1x2y2四个点的左边，以及手机检测分数；

### 头肩检测

输入参数：  
1. ./rknn_model_pipeline, so程序入口;
2. header_det , 表示图片经过头肩检测算法，得到结果；
3. data/test.png , 图片路径

输出参数：

header det 0: left: 571, top: 128, right: 612, bottom: 180, score: 0.537109

其中，header det表示人脸检测，0表示大图中第0个头肩框，后续依次是x1y1x2y2四个点的左边，以及头肩检测分数；


### 头肩检测+人脸属性

输入参数：  
1. ./rknn_model_pipeline, so程序入口;
2. header_det_attr , 表示图片经过头肩检测和人脸属性算法，得到结果；
3. data/test.png , 图片路径

输出参数：

header det 0: left: 571, top: 128, right: 612, bottom: 180, score: 0.537109

header attr 0: hat 1, glass 0, mask 0

其中，header det表示人脸检测，0表示大图中第0个头肩框，后续依次是x1y1x2y2四个点的左边，以及头肩检测分数；

header attr表示人脸属性，0表示大图中第0个人脸，后续依次是三个属性（帽子、眼镜和口罩）的输出值；

## 静态算法库编译方法



## 项目结构说明

提供接口：初始化、资源释放、模型推理；

新建doc目录，将详细文档放入，如算法原理、参数说明等。