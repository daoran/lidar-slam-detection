#!/bin/bash -e

# lidar model
/usr/src/tensorrt/bin/trtexec --onnx=sensor_inference/detection_lidar_rpn.onnx \
                              --saveEngine=sensor_inference/detection_lidar_rpn.trt \
                              --workspace=4096 --fp16 \
                              --inputIOFormats=fp16:chw --verbose --dumpLayerInfo \
                              --dumpProfile --separateProfileRun \
                              --profilingVerbosity=detailed

# image model
polygraphy convert sensor_inference/detection_image.onnx --convert-to trt -o sensor_inference/detection_image.trt --tf32 --fp16 --use-dla --allow-gpu-fallback

echo "generate trt success"