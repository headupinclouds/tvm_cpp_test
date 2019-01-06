#!/bin/bash

find _results/ndk/vulkan/ -name "tvm*.txt" | sort | while read name_android;
do
    name=$(basename $name_android);
    name_ubuntu=_builds/vulkan/${name};
    paste -d' ' $name_android <(awk '{print $NF}' $name_ubuntu) | awk '{ s1=$(NF); s2=$(NF-1); d=(s1 > s2) ? (s1-s2) : (s2-s1); if(d/(s2+s1+1e-6f) > 0.1) { print " '$name' " NR " " $0 " " d } }';
done 
