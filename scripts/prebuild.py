import shutil
import os
print("Running prebuild...")

fp_loc = __file__.split("prebuild.py")[0]

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

cp_cu_files = ["matmul", "interpolate", "gpuAAK", "AmpInterp2D"]
cp_pyx_files = ["pymatmul", "pyinterp", "gpuAAKWrap", "pyampinterp2D"]

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

def get_ode_function_lines_names():
    """Get names for ODE derivative options


    Returns:
        tuple: (:code:`list`, :code:`dict`)
            First entry is .readlines() on :code:`ode.cc`.
            Second entry is dictionary with information on
            available ODE functions.
    """
    with open("src/ode_base_example.cc", "r") as fp:
        lines = fp.readlines()

    if "ode_base.cc" in os.listdir("src/"):
        with open("src/ode_base.cc", "r") as fp:
            lines += fp.readlines()

    # find derivative functions and get info
    functions_info = {}
    for i in range(len(lines) - 1):
        if lines[i][:9] == "__deriv__":
            if lines[i][9] == "\n":
                line = lines[i + 1]
            else:
                line = lines[i]

            if "::deriv_func" in line:
                name = line.split("::deriv_func")[0].split(" ")[-1]
                func_type = "class"

            else:
                name = line.split("(double ydot[]")[0].split(" ")[-1]
                func_type = "func"

            functions_info[name] = {"type": func_type, "files": [], "citations": []}

    # get all the additional information on functions in the c file
    for line in lines:
        if line[:7] == "#define":
            for name in functions_info.keys():
                if line.split(" ")[1][0 : 0 + len(name) + 13] == f"{name}_num_add_args":
                    functions_info[name]["num_add_args"] = int(line.split(" ")[2][:-1])

                elif line.split(" ")[1][0 : 0 + len(name) + 9] == f"{name}_spinless":
                    functions_info[name]["background"] = "Schwarzschild"

                elif line.split(" ")[1][0 : 0 + len(name) + 11] == f"{name}_equatorial":
                    functions_info[name]["equatorial"] = "true"

                elif line.split(" ")[1][0 : 0 + len(name) + 9] == f"{name}_circular":
                    functions_info[name]["circular"] = "true"

                elif line.split(" ")[1][0 : 0 + len(name) + 2] == f"{name}_Y":
                    functions_info[name]["convert_Y"] = "true"

                elif line.split(" ")[1][0 : 0 + len(name) + 30] == f"{name}_integrate_constants_of_motion":
                    functions_info[name]["integrate_constants_of_motion"] = "true"

                elif line.split(" ")[1][0 : 0 + len(name) + 25] == f"{name}_disable_integrate_phases":
                    functions_info[name]["integrate_phases"] = "false"

                elif line.split(" ")[1][0 : 0 + len(name) + 5] == f"{name}_file":
                    functions_info[name]["files"].append(line.split(" ")[2][:-1])

                elif line.split(" ")[1][0 : 0 + len(name) + 9] == f"{name}_citation":
                    functions_info[name]["citations"].append(line.split(" ")[2][:-1])

    # format defaults according to c++ setup
    defaults = {
        "num_add_args": 0,
        "background": "KERR",
        "equatorial": "false",
        "circular": "false",
        "convert_Y": "false",
        "integrate_constants_of_motion":"false",
        "integrate_phases":"true",
    }
    # fill anything that did not appear
    for name, info in functions_info.items():
        for key, val in defaults.items():
            functions_info[name][key] = info.get(key, val)

    return lines, functions_info


def ode_prepare():
    """Prepare files for ODEs"""

    # get all the info
    lines, functions_info = get_ode_function_lines_names()

    # write out the function info to a python file
    with open("./few/utils/odeoptions.py", "w") as fp:
        fp.write("ode_options = " + functions_info.__repr__())

    # start preparing ode.cc

    full = ""

    # adjust function names for functions that are not classes
    for line in lines:
        for func in functions_info:
            if "void " + func + "(double ydot[]" in line:
                line = line.replace(
                    "void " + func + "(double ydot[]",
                    "void " + func + "_base_func" + "(double ydot[]",
                )
        full += line

    # build class for functions in ode_base_example.cc
    for i, (func, info) in enumerate(functions_info.items()):
        if info["type"] == "func":
            full.replace("void " + func, "void " + func + "_base_func")

            full += """
                {0}::{0}(std::string few_dir){1}{2}

                {0}::~{0}(){1}{2}

                void {0}::deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args)
                {1}
                    {0}_base_func(ydot, y, epsilon, a, integrate_backwards, additional_args);
                {2}
            """.format(
                func, "{", "}"
            )

    # put together ODE carrier C++ class
    full += """

    ODECarrier::ODECarrier(std::string func_name_, std::string few_dir_)
    {
        func_name = func_name_;
        few_dir = few_dir_;
    """

    # setup for all functions in ode_base_example.cc
    for i, (func, info) in enumerate(functions_info.items()):
        lead = "if" if i == 0 else "else if"

        full += """
            {0} (func_name == "{1}")
            {2}
        """.format(
            lead, func, "{"
        )
        full += """
                {0}* temp = new {0}(few_dir);
                convert_Y = temp->convert_Y;
                background = temp->background;
                equatorial = temp->equatorial;
                circular = temp->circular;
                integrate_constants_of_motion = temp->integrate_constants_of_motion;
                integrate_phases = temp->integrate_phases;
                func = (void*) temp;

            """.format(
            func
        )

        full += """
            }

        """

    full += """
    }
    """

    # setup get_derivatives functions
    full += """

    void ODECarrier::get_derivatives(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args)
    {
    """

    for i, (func, info) in enumerate(functions_info.items()):
        lead = "if" if i == 0 else "else if"

        full += """
            {0} (func_name == "{1}")
            {2}
        """.format(
            lead, func, "{"
        )
        full += """
                {0}* temp = ({0}*)func;

                temp->deriv_func(ydot, y, epsilon, a, integrate_backwards, additional_args);

            """.format(
            func
        )

        full += """
            }

        """

    full += """
    }
    """
    full += """

    void ODECarrier::dealloc()
    {
    """
    for i, (func, info) in enumerate(functions_info.items()):
        lead = "if" if i == 0 else "else if"

        full += """
            {0} (func_name == "{1}")
            {2}
        """.format(
            lead, func, "{"
        )
        full += """
                {0}* temp = ({0}*)func;

                delete temp;

            """.format(
            func
        )

        full += """
            }

        """

    full += """
    }
    """

    # write out to ode.cc
    with open("src/ode.cc", "w") as fp:
        fp.write(full)

    # get ode_base_example.hh
    with open("include/ode_base_example.hh", "r") as fp:
        hh_lines = fp.read()

    if "ode_base.hh" in os.listdir("include/"):
        with open("include/ode_base.hh", "r") as fp:
            hh_lines += fp.read()

    full_hh = """
    #ifndef __ODE__
    #define __ODE__

    #include "global.h"
    #include <cstring>

    #define __deriv__

    """

    full_hh += hh_lines

    # putting together class info for functions that are not classes
    for i, (func, info) in enumerate(functions_info.items()):
        if info["type"] == "func":
            full_hh += """

            class {0}{1}
            public:
                double test;
                int background = {3};
                bool equatorial = {4};
                bool circular = {5};
                bool integrate_constants_of_motion = {6};
                bool integrate_phases = {7};
                bool convert_Y = {8};
                {0}(std::string few_dir);

                void deriv_func(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
                ~{0}();
            {2};

        """.format(
                func, "{", "}", info["background"], info["equatorial"], info["circular"],
                info["integrate_constants_of_motion"], info["integrate_phases"], info["convert_Y"],
            )

    # ode carrier hh info
    full_hh += """

    class ODECarrier{
        public:
            std::string func_name;
            std::string few_dir;
            int background;
            bool equatorial;
            bool circular;
            bool integrate_constants_of_motion;
            bool integrate_phases;
            bool convert_Y;
            void* func;
            ODECarrier(std::string func_name_, std::string few_dir_);
            void dealloc();
            void get_derivatives(double ydot[], const double y[], double epsilon, double a, bool integrate_backwards, double *additional_args);
    };

    #endif // __ODE__

    """

    # add to ode.hh
    with open("include/ode.hh", "w") as fp:
        fp.write(full_hh)

ode_prepare()