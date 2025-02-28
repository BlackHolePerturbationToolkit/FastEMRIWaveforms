"""
File only present in from-source installations to detect editable mode.

It this file is present, the project storage path and download path are set
to $root/src/few/data by default.
This default is still overwritten by `few.ini`, environment variable or other
ways of enforcing configuration options.
"""
