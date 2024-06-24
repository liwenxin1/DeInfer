for modelname in "mobilenet" "resnet-18" "resnet-34" "resnet-50" "resnet-101" "resnet-152" "shufflenet" "transformer" "vgg-11" "vgg-13" "vgg-16" "vgg-19" "squeezenet"
do
    python modelDeploy/modelProfile/model_mps_runtime.py --model_name $modelname
done