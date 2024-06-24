#!/bin/bash
nvidia-cuda-mps-control -d
export MPSID=`echo get_server_list | nvidia-cuda-mps-control`
echo $MPSID
source tools/envivar.sh