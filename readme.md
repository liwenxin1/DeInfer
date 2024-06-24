# DeInfer

we propose DeInfer algorithm through following manners: 1) we identify the key factors that lead to interference, conduct a system-
atic study and develop a highly accurate interference prediction algorithm based on the random forest algorithm, achieving a four times improvement compared to the state-of-the-art interference prediction algorithms; 2) we utilize the queue theory to model the randomness of the arrival process and put forward a GPU resource allocation algorithm, which reduces the deadline miss rate by an average of over 30%.

## Prototype of DeInfer

DeInfer comprises several components: The data distributor is responsible for distributing data and triggering resource allocation based on changes in data load. The controller is tasked with i) conducting performance predictions and allocating resources, as well as ii) measuring the parameters of newly registered models. The worker controls the GPU hardware for inference. The model repository stores model parameters and measurement parameters.

Specifically, the dispatcher component incorporates a task change trigger that monitors whether the data arrival rate of the DLI task has changed. Since the arrival process of data is random, we utilize an improved CUSUM algorithm to make the determination. A data dispatcher evenly distributes data to each GPU that contains the task task.

The controller serves as a resource allocator, utilizing a monitor to obtain the current task parameters of the GPUs and combining them with a performance predictor for resource allocation. A model profiler measures the parameters of unregistered models.

The worker component consists of data buffering queues that are responsible for buffering data that can be processed before the SLO limit. The hardware executor then places the data from the buffering queue into the GPU for execution.


## Model the Interference Prediction Algorithm

We discovered a linear relationship between interference and the resource allocation amount of the model.

$$
\psi_{ij}=\alpha_1 m_{ij}
$$

The number of interference models,$n_{ij}$, has a linear relationship with interference:

$$
\psi_{ij}=\alpha_2 n_{ij}
$$

Based on the analysis results of GFLOPS and L2 cache in Section 2, we have designed a parameter $p_{ij}$ to take into account the combined impact of GFLOPS and L2 cache utilization on interference. The interference is divided into two parts: the GFLOPS and L2 cache utilization of the model itself and other models, with coefficients $\beta_1$ and $\beta_2$ respectively:

$$
p_{ij}=\beta_1 G_{ij} L_{ij} +\beta_2 \sum_{i \in I/i}\hat{G_{ij}}\hat{L_{ij}}
$$

And we found that $p_{ij}$ has a roughly linear relationship with interference:

$$
\psi_{ij}=\alpha_3 p_{ij}
$$

We use the above features to build a stochastic forest regression prediction model.

## the Queuing-based Resource Allocation Model

We consider our problem as a special case of the $M/G/1$ queue, where G represents an arbitrary distribution. According to the Pollaczek-Khintchine formula, the average queue length $L_s$ can be expressed as the following formula:

$$
L_s=\rho+ \frac{\rho^2+\lambda^2*var[T]}{2(1-\rho)}
$$

Where T represents the service processing time,$\lambda$ represents the Poisson intensity, and $\rho$ can be expressed as:

$$
\rho=\lambda *E(T)
$$

Since the data processing latency must not exceed the latency SLO, the following condition needs to be satisfied:

$$
Ws_{ij}<=SLO_j/2
$$

Where $\rho\in[0,1]$,the maximum processing time per unit of data can be derived as:

$$
MaxD_{ij}=\frac{\lambda_{ij}SLO_j+2-\sqrt{\lambda_{ij}^2SLO_j^2+4}}{2\lambda_{ij}}
$$

Similarly, if $D_{ij}$ is known, we can solve for $\lambda_{ij}$:

$$
\lambda_{ij}=\frac{2D_{ij}-SLO_i}{D_{ij}^2-D_{ij}SLO_i}
$$

In the context of inference services, the average arrival rate of data per second is a relatively large value. For the Poisson distribution, when $\lambda$ is large, it can be approximated as a Gaussian distribution. Since the Gaussian distribution exhibits symmetry, to ensure that the data waiting length does not exceed the maximum queue length, the minimum value for the queue length limit should satisfy the following condition:

$$
Lq=\frac{\rho_{ij}^2}{1-\rho_{ij}}
$$

## Dependencies and Requirements

- Description of required hardware resources
  We set up a GPU workstation, equipped with 2 NVIDIA RTX3080 GPU card and 40GB memory.
- Description of the required operating system:
  Ubuntu 20.04
- Required software libraries:
  NVIDIA Driver, cuDNN, CUDA, Python3, tvm, Torchvision, Torch, Pandas, Scikit-image, Numpy, Scipy, Pillow,TensorRT.

## Getting Started

### Compile the AI model

In order to build a delay prediction model, we first need to build a common data set of the model, and we use the model shown in this paper. We use tvm to build AI compiled versions of multiple models

It takes about a few minutes to execute the following code

```shell
bash openmps.sh
cd DeInfer/tvm_module
bash tune.sh
bash other_tune.sh
```

### Build the Dataset

After obtaining the model, you need to use the dataset construction method in paper 2.2.1 to carry out the test data set that the model is placed on GPU

It takes about a few hours to execute the following code

```
cd model_test
python model_interference_random_test.py
python test_data2csv.py
```

After execution, we get the corresponding dataset file in the profiles folder

```
model_interference_feature.csv
model_interference_feature2.csv
```

We have obtained the corresponding datasets on 3080 GPU, which are placed in the tvm_module/model_test/profiles folder

### Using Dataset to Build Forecasting Model

```
cd ..
cd RF_model
python RF_model.py
```

You will get a model parameter file in the same directory, And you need to place the parameter file under modelDeploy/forecasting_model

```
cp random_forest_model.pkl ../../modelDeploy/forecasting_model
```

## DeInfer System

### Model Registration

First, you need to register the model to the modelRepository folder

```
modelRepository
├── mobilenet
    ├── config.json
    ├── mobilenet_B16.so
    ├── mobilenet_B1.so
    ├── mobilenet_B2.so
    ├── mobilenet_B4.so
    ├── mobilenet_B8.so
    └── mps_runtime.txt

```

Config files are used for configuration parameters of some models. Where the setting of batchsize needs to correspond to lib_file, and the file name of lib_file needs to be the same as the name of the model folder

```
{
    "name": "mobilenet",
    "type": "image classification",
    "batchsize":[1,2,4,8,16],
    "input_shape": [3, 224, 224],
    "output_shape": [1000],
    "slo":5000,
    "flag":0,
    "lib_file":["mobilenet_B1.so","mobilenet_B2.so","mobilenet_B4.so","mobilenet_B8.so","mobilenet_B16.so"]
}
```



The mps_runtime.txt folder first tests the running time of the model under different resource configurations, as shown below

```
mobilenet batch_size:1 mps_set:10 percentage_runtime_cost:0.002948341417256289
mobilenet batch_size:1 mps_set:20 percentage_runtime_cost:0.0016529210937202016
mobilenet batch_size:1 mps_set:30 percentage_runtime_cost:0.0011420468087099812
mobilenet batch_size:1 mps_set:40 percentage_runtime_cost:0.0009275778479532164
mobilenet batch_size:1 mps_set:50 percentage_runtime_cost:0.0007076652995237422
mobilenet batch_size:1 mps_set:60 percentage_runtime_cost:0.0006966089413354029
mobilenet batch_size:1 mps_set:70 percentage_runtime_cost:0.0006873496191326024
mobilenet batch_size:1 mps_set:80 percentage_runtime_cost:0.0006826628598772729
mobilenet batch_size:1 mps_set:90 percentage_runtime_cost:0.0006416145064563023
mobilenet batch_size:1 mps_set:100 percentage_runtime_cost:0.0005265325781962727
...
...
...
```

Finally, we assume that the user has a need for data preprocessing, so we leave the flexibility of building the model to the user, who needs to build factory method functions for their own model. Users need to define their own model classes in the modelFactory/modelInterface.py file, and then complete the registration of the model in modelFactory/factoryMethod.py. The specific steps are shown in the source code.

### DeInfer Runtime

When the model registration is complete, only python DeInfer_system.py is needed to run the model

```
python modelDeploy/DeInfer_system.py
```

We have only implemented the prototype version of DeInfer, including resource scheduling algorithms and simple runtime code, but we have built the components needed by the entire DeInfer runtime, and users can reconstruct new scheduling algorithms according to their own requirements.
