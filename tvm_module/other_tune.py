import numpy as np
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import torch
from tvm_module.model_implementation.lstm import LSTMModel
def get_other_model(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    
    if name.startswith("LSTM-"):
        n_layer = int(name.split("-")[1])
        LSTM_model=LSTMModel(50,100,n_layer,1)
        input_shape=(batch_size,100,50)
        output_shape=(batch_size,1)
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(LSTM_model, input_data).eval()
        mod,params=tvm.relay.frontend.from_pytorch(scripted_model,[("data",(input_shape,"float32"))])
        
    elif name=="transformer":
        from tvm_module.model_implementation.transformer import TransformerEncoder
        d_model = 512
        nhead = 8
        num_layers = 6
        d_ff = 2048
        model = TransformerEncoder(num_layers, d_model, nhead, d_ff)
        # Input shape (batch_size, sequence_length, d_model)
        # default batch_size=32, sequence_length=50, d_model=512
        sequence_length = 50
        input_shape=(batch_size, sequence_length, d_model)
        output_shape=(batch_size, sequence_length, d_model)
        input_data = torch.rand(input_shape)
        scripted_model = torch.jit.trace(model, input_data)
        input_name = "data"
        shape_list = [(input_name, input_data.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    
    elif name=="bert":
        pass
    
    elif name=="DeepSpeech":
        pass
    
    elif name=="WordEmbedding":
        pass
    
    elif name=="GNN":
        pass
    
    elif name=="GAN":
        pass
    
        
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape



import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--network",type=str,default="transformer")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--factor",type=str,default="untuned")
    args=parser.parse_args()
    network=args.network
    batch_size=args.batch_size
    factor=args.factor
    target = tvm.target.Target("cuda")
    layout="NCHW"#batch_size,channel,dim

    log_file = "tvm_module/tune_file/%s-%s-B%d-%s-%s.json" % (network, layout, batch_size, factor,target.kind.name)

    # Extract tasks from the network
    print("Extract tasks...")
    mod, params, input_shape, output_shape = get_other_model(network, batch_size)
    
    if factor!="untuned":
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

        # for idx, task in enumerate(tasks):
        #     print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        #     print(task.compute_dag)
            
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
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)

    else:
        # Compile with the history best
        print("Compile...")
        #with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    lib_file = "tvm_module/tune_file/%s-%s-B%d-%s-%s.so" % (network, layout, batch_size, factor,target.kind.name)
    lib.export_library(lib_file)
    # Create graph executor
    # dev = tvm.device(str(target), 0)
    # module = graph_executor.GraphModule(lib["default"](dev))
    # data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    # module.set_input("data", data_tvm)

    # # Evaluate
    # print("Evaluate inference time cost...")
    # print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
