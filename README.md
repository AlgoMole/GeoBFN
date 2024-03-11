
# GeoBFN

**Official implementation of **ICLR2024 Oral** [Unified Generative Modeling of 3D Molecules with Bayesian Flow Networks](https://openreview.net/forum?id=NSVtmmzeRB)**

## Prerequisite
You will need to have a host machine with gpu, and have a docker with `nvidia-container-runtime` enabled, refer to [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if you don't have them installed.

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

> [!NOTE]
> - The `make` will automatically build the docker image and run the container. with your host home directory mounted to the `${HOME}/home` directory inside the container. **highly recommended**
> 
> - If you need to setup the environment manually, please refer to files `docker/Dockerfile`, `docker/asset/requirements.txt` and `docker/asset/apt_packages.txt`. 

### Train a model on qm9 dataset

**Inside container, find path to your repo. inside `GeoBFN/`** run


```bash
make -f train.mk
```

>[!CAUTION]
> - You could encounter **connection error** if your server is in China, you can manually [download the dataset from baidu netdisk](https://pan.baidu.com/s/1EUa58hkPvoYoIiLahbhnaA?pwd=i9wm) and put it in `./GeoBFN` directory. run the code block again after the dataset is downloaded.
> Alternatively you can use a proxy to alow the script download the dataset automatically.

> [!NOTE]
> - You probably will be prompted to enter your wandb api key, you can skip this by adding `--no_wandb` to the command

> [!TIP]
> - Better run the training command inside a tmux session, as it takes long time to finish training.
> 
> - exiting from container wound't stop the container, run `make` from host at `GeoBFN/docker` to log in the running container again. if really need to kill the container run `make kill` from `GeoBFN/docker`.



