FROM mlrun/ml-models-gpu:1.3.0
RUN pip install -U transformers[deepspeed]
RUN pip install -U datasets
RUN pip install -U accelerate
RUN pip install -U evaluate
RUN pip install -U protobuf==3.20.*
RUN pip install -U mpi4py
RUN conda install -c "nvidia/label/cuda-11.7.1" cuda-nvprof