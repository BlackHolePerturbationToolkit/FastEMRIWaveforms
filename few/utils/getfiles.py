import requests
import os
import subprocess
import warnings
import few

record_by_version = {"1.0.0": 3981654}


def check_for_file_download(fp, few_dir):
    try:
        os.listdir(few_dir + "few/files/")
    except OSError:
        os.mkdir(few_dir + "few/files/")

    if fp not in os.listdir(few_dir + "few/files/"):
        warnings.warn(
            "The file {} did not open sucessfully. It will now be downloaded to the proper location.".format(
                fp
            )
        )

        record = record_by_version.get(few.__version__)
        url = "https://zenodo.org/record/" + str(record) + "/files/" + fp

        # download to proper location
        subprocess.run(["wget", url])

        os.rename(fp, few_dir + "few/files/" + fp)
