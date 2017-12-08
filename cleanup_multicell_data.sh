#!/usr/bin/env bash

abo_dir=/home/drew/Documents/abo_data
ccbp_dir=/home/drew/Documents/contextual_circuit_bp

cd $abo_dir
rm -rf multi_cell_exps/*
cd $ccbp_dir
rm -rf dataset_processing/MULTIALLEN*
rm -rf models/structs/MULTIALLEN*
cp experiments.py.backup experiments.py
python prepare_experiments.py --initialize
cd $abo_dir

