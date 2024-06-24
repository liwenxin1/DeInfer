import numpy as np
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
from torchvision import models
import torch
import argparse
def get_network(name, batch_size, layout="NCHW", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("vgg-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name.startswith("squeezenet-"):
        version=name.split("-")[1]
        assert layout == "NCHW", "squeezenet only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version=version,
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    
    elif name == "shufflenet":
        shufflenet_model = models.shufflenet_v2_x0_5().eval()
        input_data=torch.randn(input_shape)
        scripted_model = torch.jit.trace(shufflenet_model, input_data).eval()
        mod,params=tvm.relay.frontend.from_pytorch(scripted_model,[("data",(input_shape,"float32"))])
    
    
        
        
        
    return mod, params, input_shape, output_shape


# Define the neural network and compilation target
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--network", type=str, default="resnet-18")
    argparse.add_argument("--layout", type=str, default="NCHW")
    argparse.add_argument("--factor", type=str, default="untuned")
    argparse.add_argument("--target", type=str, default="cuda")
    argparse.add_argument("--batch_size", type=int, default=1)
    args = argparse.parse_args()
    network = args.network
    layout = args.layout
    factor = args.factor
    target = args.target
    batch_size = args.batch_size
    
    #for example
    batch_size_list = [1,2,4,8,16]
    factor_list=[50,100,200,400]
    network_list = ["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152"]
    network_list = ["vgg-11","vgg-13","vgg-16","vgg-19"]
    network_list=["squeezenet-1.0","squeezenet-1.1"]

    
    target = tvm.target.Target(target)
    dtype = "float32"
    
        
    log_file = "tune_file/%s-%s-B%d-%s-%s.json" % (network, layout, batch_size, factor,target.kind.name)


    mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

    if factor != "untuned":
        # Extract tasks from the network
        print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)
            
        def run_tuning():
            print("Begin tuning...")
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=len(tasks)*factor,  # change this to 20000 to achieve the best performance
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )

            tuner.tune(tune_option)

        run_tuning()


        # Compile with the history best
        print("Compile...")
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)

    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    lib_file = "tune_file/%s-%s-B%d-%s-%s.so" % (network, layout, batch_size, factor,target.kind.name)
    lib.export_library(lib_file)
    # Create graph executor
    # dev = tvm.device(str(target), 0)
    # module = graph_executor.GraphModule(lib["default"](dev))
    # data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    # module.set_input("data", data_tvm)

    # # Evaluate
    # print("Evaluate inference time cost...")
    # print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
