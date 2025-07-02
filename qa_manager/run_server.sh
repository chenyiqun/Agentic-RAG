#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python flask_server.py 8000 > "flask_server_output_8000.txt" 2>&1 &