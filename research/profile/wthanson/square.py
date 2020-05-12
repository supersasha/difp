import json
import numpy as np

def one_at(fr, to, z=0):
    return [1 if i >= fr and i < to else z for i in (np.arange(65)*5+380)]

def gen_datasheet():
    ds = {
        'red': {
            'sense': one_at(650, 655, z=-10),
            'dye': { 'data': one_at(600, 710) }
        },
        'green': {
            'sense': one_at(550, 555, z=-10),
            'dye': { 'data': one_at(500, 600) }
        },
        'blue': {
            'sense': one_at(450, 455, z=-10),
            'dye': { 'data': one_at(380, 500) }
        }
    }
    return json.dumps(ds, indent=4)

if __name__ == '__main__':
    print(gen_datasheet())
