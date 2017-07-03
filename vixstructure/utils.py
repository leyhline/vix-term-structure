"""
vix-term-structure.utils

Some helper functions for evaluating the data.

:copyright: (c) 2017 Thomas Leyh
"""

import datetime


def parse_model_repr(repr_str: str):
    """
    Parse the basename of a model file to get its parameters
    returned as tuple.
    """
    repr_list = repr_str.split("_")
    assert len(repr_list) >= 7, "String does not match representation format."
    repr_dict = dict()
    repr_dict["datetime"] = datetime.datetime.strptime(repr_list[0], "%Y%m%d%H%M%S")
    repr_dict["hostname"] = repr_list[1]
    repr_dict["depth"] = int(repr_list[2].lstrip("depth"))
    repr_dict["width"] = int(repr_list[3].lstrip("width"))
    repr_dict["dropout"] = float(repr_list[4].lstrip("dropout"))
    repr_dict["optimizer"] = repr_list[5].lstrip("optim")
    repr_dict["lr"] = float(repr_list[6].lstrip("lr"))
    repr_dict["normalized"] = True if len(repr_list) >= 8 and repr_list[7] == "normalized" else False
    return repr_dict

