SHELL := /bin/bash

run:
	[[ -d dataset/qm9 ]] || (set -e; ([[ -e qm9.tar.gz ]]|| ( gdown "https://drive.google.com/file/d/1erfnk2CaeqAGaMc9L95WI9ZEvZz5GHrl/view?usp=sharing" --fuzzy)); mkdir -p dataset; tar -xvf qm9.tar.gz -C dataset;)
	python geobfn_train.py --config_file configs/bfn4molgen.yaml --epochs 3000 --no_wandb
