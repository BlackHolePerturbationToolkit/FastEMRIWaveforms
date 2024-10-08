import shutil
print("Running prebuild...")

# fp_loc = __file__.split("prebuild.py")[0]
fp_loc = __file__.split()[0][0:-20]

fp_out_name = fp_loc + "/few/utils/constants.py"
fp_in_name = fp_loc + "/include/global.h"

# develop few.utils.constants.py
with open(fp_out_name, "w") as fp_out:
    with open(fp_in_name, "r") as fp_in:
        lines = fp_in.readlines()
        for line in lines:
            if len(line.split()) == 3:
                if line.split()[0] == "#define":
                    try:
                        _ = float(line.split()[2])
                        string_out = line.split()[1] + " = " + line.split()[2] + "\n"
                        fp_out.write(string_out)

                    except ValueError as e:
                        continue

# Install cpu versions of gpu modules

# need to copy cuda files to cpp for this special compiler we are using
# also copy pyx files to cpu version

print("Changing .cu and .pyx files to cpu compatible codes")
src = fp_loc + "/src/"

cp_cu_files = ["matmul", "interpolate", "gpuAAK"]
cp_pyx_files = ["pymatmul", "pyinterp", "gpuAAKWrap"]

for fp in cp_cu_files:
    shutil.copy(src + fp + ".cu", src + fp + ".cpp")

for fp in cp_pyx_files:
    shutil.copy(src + fp + ".pyx", src + fp + "_cpu.pyx")

# setup version file

with open(fp_loc + "/README.md", "r") as fh:
    lines = fh.readlines()

for line in lines:
    if line.startswith("Current Version"):
        version_string = line.split("Current Version: ")[1].split("\n")[0]

with open(fp_loc + "/few/_version.py", "w") as f:
    f.write("__version__ = '{}'".format(version_string))


# prepare the ode files
# from few.utils.odeprepare import ode_prepare

# ode_prepare()