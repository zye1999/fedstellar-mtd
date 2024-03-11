import os
import sys
import time
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fedstellar.learning.pytorch.mnist.mnist import MNISTDataset
from fedstellar.learning.pytorch.fashionmnist.fashionmnist import FashionMNISTDataset
from fedstellar.learning.pytorch.syscall.syscall import SYSCALLDataset
from fedstellar.learning.pytorch.cifar10.cifar10 import CIFAR10Dataset

from fedstellar.config.config import Config
from fedstellar.learning.pytorch.mnist.models.mlp import MNISTModelMLP
from fedstellar.learning.pytorch.mnist.models.cnn import MNISTModelCNN
from fedstellar.learning.pytorch.fashionmnist.models.mlp import FashionMNISTModelMLP
from fedstellar.learning.pytorch.fashionmnist.models.cnn import FashionMNISTModelCNN
from fedstellar.learning.pytorch.syscall.models.mlp import SyscallModelMLP
from fedstellar.learning.pytorch.syscall.models.autoencoder import SyscallModelAutoencoder
from fedstellar.learning.pytorch.cifar10.models.resnet import CIFAR10ModelResNet
from fedstellar.learning.pytorch.cifar10.models.fastermobilenet import FasterMobileNet
from fedstellar.learning.pytorch.cifar10.models.simplemobilenet import SimpleMobileNetV1
from fedstellar.learning.pytorch.cifar10.models.cnn import CIFAR10ModelCNN
from fedstellar.learning.pytorch.cifar10.models.cnnV2 import CIFAR10ModelCNN_V2
from fedstellar.learning.pytorch.cifar10.models.cnnV3 import CIFAR10ModelCNN_V3
from fedstellar.learning.pytorch.syscall.models.svm import SyscallModelSGDOneClassSVM
from fedstellar.node import Node, MaliciousNode
from fedstellar.learning.pytorch.datamodule import DataModule

from sklearn.svm import LinearSVC
from fedstellar.learning.scikit.mnist.mnist import MNISTDatasetScikit

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# os.environ["GRPC_VERBOSITY"] = "DEBUG"
# os.environ["GRPC_TRACE"] = "http"
# os.environ["GRPC_PYTHON_BUILD_SYSTEM_OPENSSL"] = "1"
# os.environ["GRPC_PYTHON_BUILD_SYSTEM_ZLIB"] = "1"

# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"


def main():
    config_path = str(sys.argv[1])
    config = Config(entity="participant", participant_config_file=config_path)

    n_nodes = config.participant["scenario_args"]["n_nodes"]
    experiment_name = config.participant["scenario_args"]["name"]
    model_name = config.participant["model_args"]["model"]
    idx = config.participant["device_args"]["idx"]
    host = config.participant["network_args"]["ip"]
    port = config.participant["network_args"]["port"]
    neighbors = config.participant["network_args"]["neighbors"].split()
    
    additional_node_status = config.participant["mobility_args"]['additional_node']['status']
    additional_node_round = config.participant["mobility_args"]['additional_node']['round_start']

    rounds = config.participant["scenario_args"]["rounds"]
    epochs = config.participant["training_args"]["epochs"]

    # Config of attacks
    attacks = config.participant["adversarial_args"]["attacks"]
    poisoned_persent = config.participant["adversarial_args"]["poisoned_sample_percent"]
    poisoned_ratio = config.participant["adversarial_args"]["poisoned_ratio"]
    targeted = str(config.participant["adversarial_args"]["targeted"])
    target_label = config.participant["adversarial_args"]["target_label"]
    target_changed_label = config.participant["adversarial_args"]["target_changed_label"]
    noise_type = config.participant["adversarial_args"]["noise_type"]
    iid = config.participant["data_args"]["iid"]

    indices_dir = config.participant['tracking_args']["model_dir"]
    label_flipping = False
    data_poisoning = False
    model_poisoning = False

    # config of attacks
    if attacks == "Label Flipping":
        label_flipping = True
        poisoned_ratio = 0
        if targeted == "true" or targeted == "True":
            targeted = True
        else:
            targeted = False
    elif attacks == "Sample Poisoning":
        data_poisoning = True
        if targeted == "true" or targeted == "True":
            targeted = True
        else:
            targeted = False
    elif attacks == "Model Poisoning":
        model_poisoning = True
    else:
        label_flipping = False
        data_poisoning = False
        targeted = False
        poisoned_persent = 0
        poisoned_ratio = 0

    dataset = config.participant["data_args"]["dataset"]
    num_workers = config.participant["data_args"]["num_workers"]
    model = None
    if dataset == "MNIST":
        dataset = MNISTDataset(num_classes=10, sub_id=idx, number_sub=n_nodes, iid=iid, partition="percent", seed=42, config=config)
        if model_name == "MLP":
            model = MNISTModelMLP()
        elif model_name == "CNN":
            model = MNISTModelCNN()
        else:
            raise ValueError(f"Model {model} not supported")
    elif dataset == "FashionMNIST":
        dataset = FashionMNISTDataset(num_classes=10, sub_id=idx, number_sub=n_nodes, iid=iid, partition="percent", seed=42, config=config)
        if model_name == "MLP":
            model = FashionMNISTModelMLP()
        elif model_name == "CNN":
            model = FashionMNISTModelCNN()
        else:
            raise ValueError(f"Model {model} not supported")
    elif dataset == "SYSCALL":
        dataset = SYSCALLDataset(sub_id=idx, number_sub=n_nodes, root_dir=f"{sys.path[0]}/data", iid=iid)
        if model_name == "MLP":
            model = SyscallModelMLP()
        elif model_name == "SVM":
            model = SyscallModelSGDOneClassSVM()
        elif model_name == "Autoencoder":
            model = SyscallModelAutoencoder()
        else:
            raise ValueError(f"Model {model} not supported")
    elif dataset == "CIFAR10":
        dataset = CIFAR10Dataset(num_classes=10, sub_id=idx, number_sub=n_nodes, iid=iid, partition="percent", seed=42, config=config)
        if model_name == "ResNet9":
            model = CIFAR10ModelResNet(classifier="resnet9")
        elif model_name == "ResNet18":
            model = CIFAR10ModelResNet(classifier="resnet18")
        elif model_name == "fastermobilenet":
            model = FasterMobileNet()
        elif model_name == "simplemobilenet":
            model = SimpleMobileNetV1()
        elif model_name == "CNN":
            model = CIFAR10ModelCNN()
        elif model_name == "CNN_V2":
            model = CIFAR10ModelCNN_V2()
        elif model_name == "CNN_V3":
            model = CIFAR10ModelCNN_V3()
        else:
            raise ValueError(f"Model {model} not supported")
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    dataset = DataModule(train_set=dataset.train_set, train_set_indices=dataset.train_indices_map, test_set=dataset.test_set, test_set_indices=dataset.test_indices_map, num_workers=num_workers, sub_id=idx, number_sub=n_nodes, indices_dir=indices_dir, label_flipping=label_flipping, data_poisoning=data_poisoning, poisoned_persent=poisoned_persent, poisoned_ratio=poisoned_ratio, targeted=targeted, target_label=target_label,
                         target_changed_label=target_changed_label, noise_type=noise_type)

    # TODO: Improve support for scikit-learn models
    # - Import MNISTDatasetScikit (not torch component)
    # - Import scikit-learn model
    # - Import ScikitDataModule
    # - Import ScikitLearner as learner
    # - Import aggregation algorithm adapted to scikit-learn models (e.g. FedAvgSVM)

    if not config.participant["device_args"]["malicious"]:
        node_cls = Node
    else:
        node_cls = MaliciousNode

    # Adjust the GRPC_TIMEOUT and HEARTBEAT_TIMEOUT dynamically based on the number of nodes
    config.participant["GRPC_TIMEOUT"] = n_nodes * 10
    
    # Adjust REPORT_FREQUENCY dynamically based on the number of nodes (default is 10), nodes have to report in different times (+- 5 seconds)
    config.participant["REPORT_FREQUENCY"] = (n_nodes * 0.4) + random.randint(-5, 5) if n_nodes > 10 else 10 + random.randint(-5, 5)

    node = node_cls(
        idx=idx,
        experiment_name=experiment_name,
        model=model,
        data=dataset,
        host=host,
        port=port,
        config=config,
        encrypt=False,
        model_poisoning=model_poisoning,
        poisoned_ratio=poisoned_ratio,
        noise_type=noise_type
    )

    node.start()
    time.sleep(config.participant["COLD_START_TIME"])
    # TODO: If it is an additional node, it should wait until additional_node_round to connect to the network
    # In order to do that, it should request the current round to the API
    if additional_node_status:
        print(f"Waiting for round {additional_node_round} to start")
        time.sleep(6000) # DEBUG purposes
        import requests
        url = f'http://{node.config.participant["scenario_args"]["controller"]}/scenario/{node.config.participant["scenario_args"]["name"]}/round'
        current_round = int(requests.get(url).json()['round'])
        while current_round < additional_node_round:
            print(f"Waiting for round {additional_node_round} to start")
            time.sleep(10)
        print(f"Round {additional_node_round} started, connecting to the network")

    # Node Connection to the neighbors
    for i in neighbors:
        addr = f"{i.split(':')[0]}:{i.split(':')[1]}"
        node.connect(addr)
        time.sleep(2)

    # time.sleep(5)

    if config.participant["device_args"]["start"]:
        time.sleep(config.participant["GRACE_TIME_START_FEDERATION"]) # Wait for the grace time to start the federation (default is 20 seconds)
        node.set_start_learning(rounds=rounds, epochs=epochs)  # rounds=10, epochs=5

    node.grpc_wait()

if __name__ == "__main__":
    os.system("clear")
    main()
