#!/bin/bash

if [ -d "./figures" ];
then
    echo "figures directory exists... removing"
    rm -rf "./figures"
else
    echo "figures directory not found"
fi
