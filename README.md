
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

## 2. 项目结构说明

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

## 3. 算法库使用说明


[算法库函数使用说明](https://github.com/jingwang7236/rknn_model_pipeline/tree/dev/doc/算法库函数使用说明.md)


## 4. 业务逻辑介绍

[业务逻辑介绍](https://github.com/jingwang7236/rknn_model_pipeline/tree/dev/doc/业务逻辑介绍.md)