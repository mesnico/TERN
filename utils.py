import numpy
from models.tern import TERN

def get_model(config):
    model = TERN(config)
    return model
