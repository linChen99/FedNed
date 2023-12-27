# Federated Learning with Extremely Noisy Clients via Negative Distillation

This is the code for paper: **Federated Learning with Extremely Noisy Clients via Negative Distillation.**

**Abstract:** Federated learning (FL) has shown remarkable success in cooperatively training deep models, while typically struggling with noisy labels. Advanced works propose to tackle label noise by a re-weighting strategy with a strong assumption, i.e., mild label noise. However, it may be violated in many real-world FL scenarios because of highly contaminated clients, resulting in extreme noise ratios, e.g., $>$90%. To tackle extremely noisy clients, we study the robustness of the re-weighting strategy, showing a pessimistic conclusion: minimizing the weight of clients trained over noisy data outperforms re-weighting strategies. To leverage models trained on noisy clients, we propose a novel approach, called negative distillation (FedNed). FedNed first identifies noisy clients and employs rather than discards the noisy clients in a knowledge distillation manner. In particular, clients identified as noisy ones are required to train models using noisy labels and pseudo-labels obtained by global models. The model trained on noisy labels serves as a 'bad teacher' in knowledge distillation, aiming to decrease the risk of providing incorrect information. Meanwhile, the model trained on pseudo-labels is involved in model aggregation if not identified as a noisy client. Consequently, through pseudo-labeling, FedNed gradually increases the trustworthiness of models trained on noisy clients, while leveraging all clients for model aggregation through negative distillation. To verify the efficacy of FedNed, we conduct extensive experiments under various settings, demonstrating that FedNed can consistently outperform baselines and achieve state-of-the-art performance.



### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4



### Dataset

- CIFAR-10
- CIFAR-100
- ImageNet



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                        | Description                                         |
| ------------------------------- | --------------------------------------------------- |
| `num_classes`                   | Number of classes                                   |
| `num_clients`                   | Number of all clients.                              |
| `join_ratio`                    | Ratio of participating local clients.               |
| `global_rounds`                 | Number of communication rounds.                     |
| `num_data_train`                | Number of training data.                            |
| `local_steps`                   | Number of local epochs.                             |
| `batch_size`                    | Batch size of local training.                       |
| `mini_batch_size_distillations` | Batch size of negative distillation.                |
| `temperature`                   | Temperature of negative distillation.               |
| `global_learning_rate`          | Learning rate of server updating.                   |
| `local_learning_rate`           | Learning rate of client updating.                   |
| `non_iid_alpha`                 | Control the degree of non-IIDness.                  |
| `alpha`                         | Control the first parameter of Beta Distribution.   |
| `beta`                          | Control the second  parameter of Beta Distribution. |
| `lamda`                         | Control the threshold of uncertainty value.         |



### Usage

Here is an example to run FedNed on CIFAR-10:

```python
python main.py --num_classes=10 \ 
--num_clients=20 \
--join_ratio=0.5 \
--global_rounds=100 \
--batch_size=32 \
--mini_batch_size_distillations=128 \
--temperature=2 \
--global_learning_rate=0.01 \
--local_learning_rate=0.05 \
--non-iid_alpha=10 \
--alpha=0.1 \ 
--beta=0.1 \ 
--lamda=0.12
```



