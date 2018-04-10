singularity instance.start /shared/hpc/containers/ros-indigo.img roscore
singularity exec --nv instance://roscore roscore &
singularity exec --nv /shared/hpc/containers/ros-indigo.img bash