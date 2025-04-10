# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Implementation from fvcore (https://github.com/facebookresearch/fvcore)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
from typing import Dict, Optional
# from tabulate import tabulate

class Registry(object):
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert name not in self._obj_map, "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, name: str = None, obj: object = None) -> Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                nonlocal name
                if name is None:
                    name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        return str(self._obj_map)
        # table_headers = ["Names", "Objects"]
        # table = tabulate(self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid")
        # return "Registry of {}:\n".format(self._name) + table

