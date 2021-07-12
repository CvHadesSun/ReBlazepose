import yaml
import os ,sys

f=open(os.path.join(os.path.dirname( os.path.abspath(__file__) ),'config.yaml'))
cfg= yaml.safe_load(f)


# print(cfg)