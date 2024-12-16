
## 模型编译方法

```
rm -r build
mkdir build
cd build
cmake ..
make -j16
make install  # 可以将相关文件拷贝到目标目录
```

## 模型调用说明

### cpp测试用例
代码：main.cpp

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