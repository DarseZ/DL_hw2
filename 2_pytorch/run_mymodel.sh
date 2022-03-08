#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 48 \
    --epochs 10 \
    --weight-decay 0.01 \
    --momentum 0.95 \
    --batch-size 128 \
    --lr 0.01 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
