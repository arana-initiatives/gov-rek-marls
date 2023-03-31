### Governed Reward Shaping for MARLS

This repository contains environments, governance wrappers and implemented models for different
MARL system architectures. And, the corresponding experiments and jupyter notebooks for result replication across
different sparse environment practical use-cases.

### Developer Execution Instructions

For debugging the project and parallely interacting with the codebase follow below stated two steps:

* First, open the project in your IDE and set the project `PYTHONPATH` 
with command `export PYTHONPATH=${PYTHONPATH}:$/.`

* Second, simply run the command in your `conda` environment `/home/{**path**}/miniconda3/envs/gov-rs-marls/bin/python
/home/{**path**}/gov-rs-marls/govrs/envs/openai/road_env.py` to test specific module of your choice from the project
while your `pwd` is `{some-base-path}/gov-rs-marls/`.
  * **Note:** In the above `python` command, `gov-rs-marls` is also the name of the `conda` environment in use, same as the `pwd` or repository name.
