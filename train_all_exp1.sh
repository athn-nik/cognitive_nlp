#!/usr/bin/env bash


for p in M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M13 M14 M15 M16 M17 P01; do
    for d in average pictures sentences wordclouds; do
        python train_decoder.py -i /home/nathan/Desktop/final_data/exp1/"${p}"/data_180concepts_"${d}".pkl
    done
done

