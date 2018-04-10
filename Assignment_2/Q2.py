import generate_pointcloud as gpc
import associate as ass
import os

# Read text file for pairing rgb and depth.
first_list = ass.read_file_list('rgb.txt')
second_list = ass.read_file_list('depth.txt')
results = ass.associate(first_list, second_list, 0, 0.02)

# Defining paths
Defaultwd = os.getcwd()  # Set current working directory as Defaultwd
Output_path = os.path.join(os.getcwd(), "OutputPLY")  # Generate output path for .ply files.
Enter_path1 = os.chdir(Output_path)  # Enter output path folder.
rgb_path = os.path.join(Defaultwd, "rgb").replace("\\", "/") + "/%s"  # Generate rgb path to read rgb files.
depth_path = os.path.join(Defaultwd, "depth").replace("\\", "/") + "/%s"  # Generate depth path to read depth files.

# Loop pointcloud generation.
l = len(results)
for a in range(0, l):
    b = str("{0:.6f}".format(results[a][0])) + ".png"  # Read the results for rgb and convert to string.
    c = str("{0:.6f}".format(results[a][1])) + ".png"  # Read the results for depth and convert to string.
    final = "image" + str(a+1) + ".ply"  # Output .ply naming convention.
    gpc.generate_pointcloud(open(rgb_path % b, 'rb'), open(depth_path % c, 'rb'), final)  # Read rgb and depth .png files (read, binary).

os.chdir(Defaultwd)  # Return to default working directory.
