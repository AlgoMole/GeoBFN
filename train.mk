SHELL := /bin/bash

run:
	[[ -d dataset ]] || (mkdir dataset; tar -xvf qm9.tar.gz -C dataset;)
	python bfn4molgen_train.py --config_file configs/bfn4molgen.yaml --epochs 3000
