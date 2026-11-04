import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Make runs reproducible. Model libraries also get random_state=seed later."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)