
# Collaborative Safe Formation Control

![Static Badge](https://img.shields.io/badge/ECC_2024-Submitted-blue)


## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n safe-formation python=3.9
conda activate safe-formation
pip install -r requirements.txt
```

## Run

In `test.py`, we create a static obstical field and provide a constant control signal to one agent to move through the field.

```bash
python test.py
```

We also provide the following flags:

- `--num-agents`: Number of agents in the formation
- `--safe-dist`: Minimum distance the agents will maintain from each obstacle in the field
- `--mass`: Virtual mass in the mass-spring formation dynamics
- `--spring-constant`: Virtual spring constant in the mass-spring formation dynamics
- `--dampen-constant`: Virtual dampening constant in the mass-spring formation dynamics
- `--rest-length`: Virtual rest length of the springs in the mass-spring formation dynamics
- `--control-lim`: Control limit applied to the modification signal for all agents
- `--dt`: Simulation time resolution
- `--sim-length`: Simulation length
- `--path`: File path to save animation and figures
- `--animate`: Boolean flag to create animation of simualtion