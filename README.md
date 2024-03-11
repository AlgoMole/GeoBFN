
# GeoBFN
>Official implementation of **ICLR2024 Oral** [Unified Generative Modeling of 3D Molecules with Bayesian Flow Networks](https://openreview.net/forum?id=NSVtmmzeRB)

## Prerequisite
You will need to have a host machine with gpu, and have a docker with nvidia-container-runtime enabled, refer to [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if you don't have them installed.

## Quick start

### Environment setup
Clone the repo with `git clone`,
```bash
git clone https://github.com/AlgoMole/GeoBFN.git
```

setup environment with docker,

```bash
cd ./GeoBFN/docker

make # a make is all you need
```

> The `make` will automatically build the docker image and run the container. with your host home directory mounted to the `${HOME}/home` directory inside the container. **highly recommended**
> 
> If you need to setup the environment manually, please refer to files `docker/Dockerfile`, `docker/asset/requirements.txt` and `docker/asset/apt_packages.txt`. 

### Train a model on qm9 dataset
**<code style="color : green">inside container, find path to your repo.</code> inside `GeoBFN/`** run

> better run command inside a tmux session

```bash
make -f train.mk
```

> You could encounter <code style="color : red">connection error</code> if your server is in China, you can manually download the dataset from baidu netdisk with (链接: https://pan.baidu.com/s/1EUa58hkPvoYoIiLahbhnaA?pwd=i9wm 提取码: i9wm) and put it in `./GeoBFN` directory. run `make -f train.mk` again after the dataset is downloaded.
> Alternatively you can use a proxy to alow the script download the dataset automatically.
> 
> You probably will be prompted to enter your wandb api key, you can skip this by adding `--no_wandb` to the command


