#!/bin/bash

for i in {0..1000}
do
  time python main.py expert${i} 0
done
