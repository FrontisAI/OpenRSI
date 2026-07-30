# 🪐 Superimage 🪐

Superimage is a container image for Docker and Apptainer that simulates a virtual machine. It integrates well with job schedulers like Slurm, requires no root permissions, and is fully supported in FAIR AWS clusters. The native communication mechanism it provides is SSHD.

Superimage serves as the environment to run agent actions.  It is designed to be simple but not necessarily minimal, including most of the software, packages, and tools needed for easy use by agents.

## Installation

If you don't have apptainer installed, you can install it by following the [installation instructions](https://apptainer.org/docs/admin/main/installation.html#installation-on-linux).

## Building
```bash
pushd superimage
apptainer build --fakeroot {PATH_TO_SUPERIMAGES}/superimage.root.{VERSION_ID}.sif apptainer.def
popd
```
where `{PATH_TO_SUPERIMAGES}` is where the superimage will be created and `{VERSION_ID}` is the version of the superimage you want to build (e.g., `2025-06-v1`). This will create a superimage that can be used to run tasks in the dojo.

## Setting your superimage as the default

First, Change the `superimage_version` in `src/dojo/configs/interpreter/jupyter.yaml` and give no `read_only_overlays` if you don't have any:
```yaml
...
superimage_version: "2025-05-02v2" # <---- Set to {VERSION_ID}
read_only_overlays: []
...
```
Then, set you superimage directory in your `.env` file:
```bash
SUPERIMAGE_DIR=/<PATH_TO_TEAM_STORAGE>/shared/sif/ # <---- Set to {PATH_TO_SUPERIMAGES}
```
