# FLPurifier: Backdoor Defense in Federated Learning via Supervised Contrastive Training (PyTorch)

Implementation of the vanilla federated learning paper : [Backdoor Defense in Federated Learning via Supervised Contrastive Training].


Contains the implement of FlPurifier. Experiments are produced on MNIST, Fashion MNIST , CIFAR10 and FEMNIST. 

## Requirments
Install all the packages from requirments.txt
* Python 3.6 or higher 
* Pytorch 1.8 or higher


## Parameters

| Parameter    | Description                                                           |
|--------------|-----------------------------------------------------------------------|
| `dataset`    | Dataset to use. Options: `mnist`, `fashion mnist`,`cifar10`,`emnist`. |
| `model`      | The model architecture. Options: `cnn`, `byol`, `simsiam`.            |
| `gpu`        | Run on GPU or CPU. Options: `None`. `GPU ID`.                         |
| `epochs`     | Number of communication rounds.                                       |
| `lr`         | learning rate.                                                        |
| `seed`       | Random Seed.                                                          |
| `num_users`  | Number of parties.                                                    |
| `frac`       | the fraction of parties to be sampled in each round.                  |
| `local_ep`   | Number of local training epochs in each user.                         |
| `local_bs`   | Batch size of local updates in each user..                            |
| `iid`        | Distribution of data amongst users. Options: `IID`. `Non-IID`         |

## Usage
Here are some example to run FLPurifier.
* To run the experiment with EMNIST(IID) using GPU:
```
python main.py --dataset=emnist \
    --model=byol \
    --lr=0.01 \
    --gpu=0 \
    --epochs=20 \
    --local_ep=20 \
    --local_bs=64 \
    --num_users=100 \
    --frac=0.1
    --iid=0\

```
## Other source
### Datasets:
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [CIFAR10](https://www.cs.toronto.edu/kriz/cifar.html)
* [FEMNIST](https://github.com/TalwalkarLab/leaf)
### Attacks:
* [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr)  *[[CODE]](https://github.com/AI-secure/DBA)
* [How To Backdoor Federated Learning](https://arxiv.org/abs/1807.00459) *[[CODE]](https://github.com/ebagdasa/backdoor_federated_learning)
* [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/abs/2007.05084) *[[CODE]](https://github.com/ksreenivasan/OOD_Federated_Learning)

### Defense:
* [BaFFLe: Backdoor detection via Feedback-based Federated Learning](https://arxiv.org/abs/2011.02167)
* [BaFFLe: Backdoor detection via Feedback-based Federated Learning](https://arxiv.org/abs/2011.02167)
* [CONTRA: Defending against Poisoning Attacks in Federated Learning](https://link.springer.com/chapter/10.1007/978-3-030-88418-5_22)
* [Defending against Backdoors in Federated Learning with Robust Learning Rate](https://ojs.aaai.org/index.php/AAAI/article/view/17118) *[[CODE]](https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate)
