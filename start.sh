rm -r build
rm -r install
./build-linux.sh -t rk3588
scp -r install/rk3588_model_pipeline_Linux/* root@192.168.141.116:/userdata/jingwang/rk3588_model_pipeline_Linux/
