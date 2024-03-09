
# GeoBFN
>Official implemenation of the paper [Unified Generative Modeling of 3D Molecules with Bayesian Flow Networks](https://openreview.net/forum?id=NSVtmmzeRB)

## Quick start


### environment setup
Clone the repo with `git lfs clone`,
```bash
git lfs clone https://github.com/AlgoMole/GeoBFN.git
```

setup environment with docker,

```bash
cd ./GeoBFN/docker

make # a make is all you need
```

> The `make` will automatically build the docker image and run the container. with your home directory mounted to the home directory of the container. **highly recommended**
> 
> If you need to setup the environment manually, please refer to files `docker/Dockerfile`, `docker/asset/requirements.txt` and `docker/asset/apt_packages.txt`. 

### Train a model on qm9 dataset
**inside container, find path to your repo. inside `GeoBFN/`** run

> better run command inside a tmux session

```bash
make -f train.mk
```
or

```bash
[[ -d dataset ]] || (mkdir dataset; tar -xvf qm9.tar.gz -C data;) # this command need only be run once

python bfn4molgen_train.py --config_file configs/bfn4molgen.yaml --epochs 3000 #--no_wandb #to skip logging to wandb
```
> these commands can also be found in `GeoBFN/Makefile`
> 
> you probably will be prompted to enter your wandb api key, you can skip this by adding `--no_wandb` to the command

