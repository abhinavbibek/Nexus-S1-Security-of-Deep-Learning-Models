# backdoor_tests_gtsrb.py
import sys
import torch
from evaluation.run_defense import run_defense
from data.gtsrb import gtsrb_loader
from model.preact_resnet import PreActResNet18
from model.resnet_paper import resnet32
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pairs = [
    (14, 2), (12, 13), (1, 3), (5, 8),
    (9, 10), (17, 14), (25, 38), (33, 34)
]
poison_levels = [0.05, 0.1, 0.2]

def get_datasets():
    ds = []
    for f in poison_levels:
        for s, t in pairs:
            ds.append(f"datasets/gtsrb-backdoor-{s}-to-{t}-{f}.pickle")
    return ds


def run(dataset, model):
    if "32" in model:
        def make_model():
            net = resnet32()
            net.linear = torch.nn.Linear(net.linear.in_features, 43)
            return net
        model_ctor = make_model
    else:
        def make_model():
            net = PreActResNet18()
            net.linear = torch.nn.Linear(net.linear.in_features, 43)
            return net
        model_ctor = make_model

    train_op = lambda p: torch.optim.SGD(
        p, lr=0.02, momentum=0.9, weight_decay=5e-4
    )

    sched = lambda o, e=200: torch.optim.lr_scheduler.MultiStepLR(
        o, milestones=[100, 150, 180], gamma=0.1
    )

    run_defense(
        dataset,
        model_ctor,
        [train_op, train_op],
        gtsrb_loader,
        200,
        128,
        device,
        [sched, sched],
        num_classes=43
    )



if __name__ == "__main__":
    exp_id = int(sys.argv[1])
    model = sys.argv[2]
    dataset = get_datasets()[exp_id]
    run(dataset, model)
