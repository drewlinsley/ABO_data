#!/usr/bin/env bash

cd /home/drew/Documents/abo_data
rm -rf multi_cell_exps/*
cd /home/drew/Documents/contextual_circuit_bp
rm -rf dataset_processing/MULTIALLEN*
python prepare_experiments.py --initialize
cd /home/drew/Documents/abo_data

