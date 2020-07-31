import requests
import os
import subprocess
import warnings

record = 653693


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

        url = "https://sandbox.zenodo.org/record/" + str(record) + "/files/" + fp

        # download to proper location
        subprocess.run(["wget", url])

        os.rename(fp, few_dir + "few/files/" + fp)
