
# GeoBFN
>Official implemenation of the paper [Unified Generative Modeling of 3D Molecules with Bayesian Flow Networks](https://openreview.net/forum?id=NSVtmmzeRB)

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
**inside container, find path to your repo. inside `GeoBFN/`** run

> better run command inside a tmux session

```bash
make -f train.mk
```

> data is stored on google drive you probably will also need to setup proxy: 
> - `export https_proxy=<your_proxy> http_proxy=<your_proxy>`
> 
> you probably will be prompted to enter your wandb api key, you can skip this by adding `--no_wandb` to the command


