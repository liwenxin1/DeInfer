# resnet has been tuned
# if you want to tune other networks, please modify the outloop and tune.py
for name in resnet-34 resnet-50 resnet-101 resnet-152 vgg-11 vgg-13 vgg-16 vgg-19 squeezenet-1.0 squeezenet-1.1 mobilenet
do
    for batch_size in 1 2 4 8 16
    do
        python tune.py --network $name --batch_size $batch_size
    done
done