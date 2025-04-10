from typing import Any
import yaml

class Args(object):
    def __init__(self) -> None:
        self.params = {}

    def load_from_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_params = yaml.safe_load(f)
            if yaml_params is not None:
                self.load_from_dict(yaml_params)
            pass
        return self
        pass
    
    def load_from_dict(self, n_params: dict):
        self._load_dict(self.params, n_params)
        pass
    
    def _load_dict(self, params, dict_params: dict):
        for key, item in dict_params.items():
            if isinstance(item, dict):
                params[key] = Args()
                self._load_dict(params[key].params, item)
                pass
            else:
                params[key] = item
                pass
            pass
        pass
    def __getattr__(self, name: str) -> Any:
        return self.params.get(name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def items(self):
        """以字典形式返还当前参数

        Returns:
            _type_: _description_
        """
        return self._items(self.params)
    
    def _items(self, params):
        param_dict = {}
        for key, value in params.items():
            if isinstance(value, Args):
                value = value.items()
            param_dict[key] = value    
            pass
        return param_dict
        pass

if __name__ == "__main__":
    args = Args()
    args.load_from_yaml("/home/luyx/env_sn/workspace/ri/cfg/VecStegaFont/main.yaml")
    print(args.img_dim)
    print(args.a)
    print(args.b)
    args.a = 1
    args.b = 32
    print(args.a)
    print(args.b)
    print(args.noiser.diff_binarize)