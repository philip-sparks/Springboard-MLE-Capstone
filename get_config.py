"""
The get_config.py helps setup the Flask application

"""

# external imports
import yaml


def get_config():
    """
    Read the yaml file containing the application setup

    Args:
    Returns:
        conf (dict) - contains the configurations and constants
    """

    with open("config.yaml") as file:
        conf = yaml.load(file)

    return conf
