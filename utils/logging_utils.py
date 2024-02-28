from os import mkdir
from os.path import join

def configure_save_path(base, max_dirs = 5000):
    for i in range(1, max_dirs+1):
        try:
            mkdir(join(base, f'run_{i}'))
            break
        except FileExistsError:
            pass
    return join(base, f'run_{i}')

