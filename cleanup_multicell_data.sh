#!/usr/bin/env bash

echo "Do you want to reset your local database or the p-node database? (0/1)"
read decision

if [ $decision == 1 ]
then
    ccbp_dir=/media/data_cifs/cluster_projects/contextual_circuit_bp
    echo "Resetting the p node DB."
else
    ccbp_dir=/home/drew/Documents/contextual_circuit_bp
    echo "Resetting the local DB."
fi

abo_dir=/home/drew/Documents/abo_data

cd $abo_dir
rm -rf multi_cell_exps/*
cd $ccbp_dir
rm -rf dataset_processing/MULTIALLEN*
rm -rf models/structs/MULTIALLEN*
cp experiments.py.backup experiments.py
python prepare_experiments.py --initialize
cd $abo_dir

