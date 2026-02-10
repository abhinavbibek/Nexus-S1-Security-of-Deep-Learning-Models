# # data/gtsrb_poison.py
# import pickle
# import numpy as np
# from torchvision.datasets import GTSRB
# from PIL import Image

# class PoisonGTSRBDataset(GTSRB):
#     def __init__(self, root, train=True, transform=None,
#                  download=True, poison_params=None):
#         super().__init__(root=root, split='train' if train else 'test',
#                          transform=transform, download=download)

#         self.data = self._samples_as_numpy()

#         # 🔑 FIX: extract labels from GTSRB samples
#         self.targets = np.array([label for _, label in self._samples])
#         self.true_targets = self.targets.copy()

#         self.clean_samples = None


#         if poison_params is not None:
#             with open(poison_params, "rb") as f:
#                 params = pickle.load(f)

#             self.method = params["method"]
#             self.position = params["position"]
#             self.color = params["color"]
#             self.fraction_poisoned = params["fraction_poisoned"]
#             self.poison_seed = params["seed"]
#             self.source = params["source"]
#             self.target = params["target"]

#             self.poison()

#     def _samples_as_numpy(self):
#         imgs = []
#         for path, _ in self._samples:
#             img = Image.open(path).convert("RGB")
#             img = img.resize((32, 32))   # ✅ REQUIRED FIX
#             imgs.append(np.array(img))
#         return np.stack(imgs)


#     def poison(self):
#         idxs = np.where(self.targets == self.source)[0]
#         poison_count = int(self.fraction_poisoned * len(idxs))

#         if self._split == 'train':   # 🔑 FIX
#             rng = np.random.RandomState(self.poison_seed)
#             poisoned_idxs = rng.choice(idxs, poison_count, replace=False)
#         else:
#             poisoned_idxs = idxs


#         for i in poisoned_idxs:
#             self.data[i] = poison_image(
#                 self.data[i],
#                 self.method,
#                 self.position,
#                 self.color
#             )
#             self.targets[i] = self.target

#         poisoned_mask = np.isin(np.arange(len(self.targets)), poisoned_idxs)
#         self.clean_samples = np.where(~poisoned_mask)[0]


# def poison_image(image, method, position, color):
#     poisoned = np.copy(image)
#     c = np.asarray(color)

#     x, y = position
#     if method == "pixel":
#         poisoned[x, y] = c
#     elif method == "pattern":
#         poisoned[x, y] = c
#         poisoned[x+1, y+1] = c
#         poisoned[x-1, y+1] = c
#         poisoned[x+1, y-1] = c
#         poisoned[x-1, y-1] = c
#     elif method == "ell":
#         poisoned[x, y] = c
#         poisoned[x+1, y] = c
#         poisoned[x, y+1] = c

#     return poisoned

# data/gtsrb_poison.py
import pickle
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image


class PoisonGTSRBDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, poison_params=None):
        split = "train" if train else "test"

        super().__init__(
            root=f"{root}/GTSRB/{split}",
            transform=transform
        )

        self.train = train
        self.targets = np.array([label for _, label in self.samples])
        self.true_targets = self.targets.copy()
        self.clean_samples = np.arange(len(self.targets))

        self.poisoned_mask = np.zeros(len(self.targets), dtype=bool)

        if poison_params is not None:
            with open(poison_params, "rb") as f:
                params = pickle.load(f)

            self.method = params["method"]
            self.position = params["position"]
            self.color = params["color"]
            self.fraction_poisoned = params["fraction_poisoned"]
            self.poison_seed = params["seed"]
            self.source = params["source"]
            self.target = params["target"]

            self._select_poisoned_indices()

    def _select_poisoned_indices(self):
        idxs = np.where(self.targets == self.source)[0]
        poison_count = int(self.fraction_poisoned * len(idxs))

        if self.train:
            rng = np.random.RandomState(self.poison_seed)
            poisoned = rng.choice(idxs, poison_count, replace=False)
        else:
            poisoned = idxs

        self.poisoned_mask[poisoned] = True
        self.targets[poisoned] = self.target
        self.clean_samples = np.where(~self.poisoned_mask)[0]

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.poisoned_mask[index]:
            img = poison_image(img, self.method, self.position, self.color)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(self.targets[index])


def poison_image(image, method, position, color):
    image = image.resize((32, 32))
    poisoned = np.array(image)
    c = np.asarray(color)

    x, y = position
    if method == "pixel":
        poisoned[x, y] = c
    elif method == "pattern":
        poisoned[x, y] = c
        poisoned[x+1, y+1] = c
        poisoned[x-1, y+1] = c
        poisoned[x+1, y-1] = c
        poisoned[x-1, y-1] = c
    elif method == "ell":
        poisoned[x, y] = c
        poisoned[x+1, y] = c
        poisoned[x, y+1] = c

    return Image.fromarray(poisoned)
