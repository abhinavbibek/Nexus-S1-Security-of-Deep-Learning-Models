#gtsrb_backdoor.py
import numpy as np
import pickle
import os

def generate_backdoor_poison(seed=100):
    np.random.seed(seed)

    poison_levels = [0.05, 0.1, 0.2]
    pairs = [
        (14, 2), (12, 13), (1, 3), (5, 8),
        (9, 10), (17, 14), (25, 38), (33, 34)
    ]
    methods = ["pixel", "pattern", "ell"]

    for source, target in pairs:
        method = methods[np.random.randint(3)]
        position = np.random.randint(30, size=(2,))
        color = np.random.randint(255, size=(3,))

        for f in poison_levels:
            params = {
                "method": method,
                "position": position,
                "color": color,
                "fraction_poisoned": f,
                "seed": seed + int(f * 100) + source,
                "source": source,
                "target": target
            }

            fname = f"datasets/gtsrb-backdoor-{source}-to-{target}-{f}.pickle"
            with open(fname, "wb") as f:
                pickle.dump(params, f)

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    generate_backdoor_poison()
