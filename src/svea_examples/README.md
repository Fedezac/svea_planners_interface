# svea_examples

Examples on how to use [SVEA](https://github.com/KTH-SML/svea).

## Installation

### With ROS installed

```
git clone https://github.com/KTH-SML/svea
cd svea/src
git submodule add https://github.com/KTH-SML/svea_examples
cd ..
catkin config --init --extend /opt/ros/noetic
```

### Without ROS installed

You must have docker installed.

```
util/build
util/create
util/start
```

## Usage
In order to test an endless simulation of a svea monitoring a squared trajectory in floor2 map:
```
roslaunch svea_examples floor2.launch
```
In order to start a comparison between a localization algorithm and the real svea state (e.g. provided by the mocap), make sure to be in a terminal logged in the svea (through ssh command here reported).
```bash
ssh nvidia@svea7
```
Then, launch the following command in it:
```bash
roslaunch svea_core state_publisher.launch
```
When the startup process is terminated, launch this command on the remote pc (see [main README](../../README.md) for details):
```bash
roslaunch svea_examples measure_remote.launch 
```

