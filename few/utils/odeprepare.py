import os

# get path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_ode_function_lines_names():
    """Get names for ODE derivative options


    Returns:
        tuple: (:code:`list`, :code:`dict`)
            First entry is .readlines() on :code:`ode.cc`.
            Second entry is dictionary with information on
            available ODE functions.
    """
    with open(dir_path + "/../../src/ode_base_example.cc", "r") as fp:
        lines = fp.readlines()

    if "ode_base.cc" in os.listdir(dir_path + "/../../src/"):
        with open(dir_path + "/../../src/ode_base.cc", "r") as fp:
            lines += fp.readlines()

    # find derivative functions and get info
    functions_info = {}
    for i in range(len(lines) - 1):
        if lines[i][:9] == "__deriv__":
            if lines[i][9] != "\n":
                line = lines[i]
                name = line.split("(")[0].split(" ")[2]
                func_type = "func"
            else:
                line = lines[i + 1]

                if "::" in line:
                    name = line.split("::")[0].split(" ")[1]
                    func_type = "class"

                else:
                    name = line.split("(")[0].split(" ")[1]
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
                    functions_info[name]["equatorial"] = True

                elif line.split(" ")[1][0 : 0 + len(name) + 9] == f"{name}_circular":
                    functions_info[name]["circular"] = True

                elif line.split(" ")[1][0 : 0 + len(name) + 2] == f"{name}_Y":
                    functions_info[name]["convert_Y"] = True

                elif line.split(" ")[1][0 : 0 + len(name) + 5] == f"{name}_file":
                    functions_info[name]["files"].append(line.split(" ")[2][:-1])

                elif line.split(" ")[1][0 : 0 + len(name) + 9] == f"{name}_citation":
                    functions_info[name]["citations"].append(line.split(" ")[2][:-1])

    defaults = {
        "num_add_args": 0,
        "background": "Kerr",
        "equatorial": False,
        "circular": False,
        "convert_Y": False,
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
    with open(dir_path + "/../../few/utils/odeoptions.py", "w") as fp:
        fp.write("ode_options = " + functions_info.__repr__())

    # start preparing ode.cc

    full = ""

    # adjust function names for functions that are not classes
    for line in lines:
        for func in functions_info:
            if "void " + func + "(double* " in line:
                line = line.replace(
                    "void " + func + "(double* ",
                    "void " + func + "_base_func" + "(double* ",
                )
        full += line

    # build class for functions in ode_base_example.cc
    for i, (func, info) in enumerate(functions_info.items()):
        if info["type"] == "func":
            full.replace("void " + func, "void " + func + "_base_func")

            full += """
                {0}::{0}(std::string few_dir){1}{2}

                {0}::~{0}(){1}{2}

                void {0}::deriv_func(double* pdot, double* edot, double* Ydot,
                                  double* Omega_phi, double* Omega_theta, double* Omega_r,
                                  double epsilon, double a, double p, double e, double Y, double* additional_args)
                {1}
                    {0}_base_func(pdot, edot, Ydot, Omega_phi, Omega_theta, Omega_r,
                                  epsilon, a, p, e, Y, additional_args);
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

    void ODECarrier::get_derivatives(double* pdot, double* edot, double* Ydot,
                      double* Omega_phi, double* Omega_theta, double* Omega_r,
                      double epsilon, double a, double p, double e, double Y, double* additional_args)
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

                temp->deriv_func(pdot, edot, Ydot, Omega_phi, Omega_theta, Omega_r,
                                epsilon, a, p, e, Y, additional_args);

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
    with open(dir_path + "/../../src/ode.cc", "w") as fp:
        fp.write(full)

    # get ode_base_example.hh
    with open(dir_path + "/../../include/ode_base_example.hh", "r") as fp:
        hh_lines = fp.read()

    if "ode_base.hh" in os.listdir(dir_path + "/../../include/"):
        with open(dir_path + "/../../include/ode_base.hh", "r") as fp:
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

                {0}(std::string few_dir);

                void deriv_func(double* pdot, double* edot, double* Ydot,
                                  double* Omega_phi, double* Omega_theta, double* Omega_r,
                                  double epsilon, double a, double p, double e, double Y, double* additional_args);
                ~{0}();
            {2};

        """.format(
                func, "{", "}"
            )

    # ode carrier hh info
    full_hh += """

    class ODECarrier{
        public:
            std::string func_name;
            std::string few_dir;
            void* func;
            ODECarrier(std::string func_name_, std::string few_dir_);
            void dealloc();
            void get_derivatives(double* pdot, double* edot, double* Ydot,
                              double* Omega_phi, double* Omega_theta, double* Omega_r,
                              double epsilon, double a, double p, double e, double Y, double* additional_args);

    };

    #endif // __ODE__

    """

    # add to ode.hh
    with open("include/ode.hh", "w") as fp:
        fp.write(full_hh)
