#!/bin/bash

if [ -d "./models" ];
then
    echo "models directory exists... removing"
    rm -rf "./models"
else
    echo "models directory not found"
fi
