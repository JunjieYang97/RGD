# Adversarial Attack algorithm RGD
Codes for paper [Rethinking PGD Attack: Is Sign Function Necessary?](https://arxiv.org/pdf/2312.01260.pdf) by Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang.

## How to use our code.
Our adversarial attack algorithm is built on [AdverTorch](https://github.com/BorealisAI/advertorch), [AutoAttack](https://github.com/fra31/auto-attack) and [RobustBench](https://github.com/RobustBench/robustbench#model-zoo) where we demonstrate the performance improvement of proposed RGD over PGD and Auto-Attack. Please make sure to install above packages and its dependecies before using this code. We have tested this code in Python 3.8 with PyTorch 2.0.1.

### Detailed instructions.

We introduce the following hyperparameters to run the code in [attack.py](https://github.com/JunjieYang97/RGD/blob/master/attack.py).


+ `--nb_iter`: Attack steps of each algorithm.
+ `--eps`: Epsilon ball size, 0.0314=8/255, we should use the value which is smaller than 1.
+ `--alpha`: The step size for each update. For RGD, we should expect to use large alpha, e.g., 100. For PGD, smaller alpha is preferred, e.g., 2/255.
+ `--update_method`: Specify the attack algorihm. "sign_grad" refers the PGD/AutoAttack, "raw_grad" refers the PGD(Raw), "noclip_raw" refers the RGD, "rgd+aa" refer to update with RGD first and switch to AutoAttack.
+ `--alg`: If we train the algorithm in vanilla PGD or AutoAttack contexts.
+ `--model_name`: Imported model from [RobustBench](https://github.com/RobustBench/robustbench#model-zoo).
+ `--init_method`: Zero initial or random initial. Zero initial is preferred for RGD and random initial is preferred for PGD.
+ `--rgd_iter`: In RGD+AA algorithm, the number of steps to update with RGD.

We provide following commands to run the code:

#### Vanilla PGD with 8/255 epsilon ball:

```python
python attack.py --alpha 0.02 --update_method sign_grad --eps 0.0314
```

#### RGD with 8/255 epsilon ball:

```python
python attack.py --alpha 3000 --update_method noclip_raw --init_method zero_init --eps 0.0314
```

#### RGD+AA with 8/255 epsilon ball:
```python
python attack.py --alpha 3000 --update_method rgd_aa --alg AA --init_method zero_init --eps 0.0314
```


## Citation

If this repo is useful for your research, please cite our paper:

```tex
@article{yang2023rethinking,
  title={Rethinking PGD Attack: Is Sign Function Necessary?},
  author={Yang, Junjie and Chen, Tianlong and Chen, Xuxi and Wang, Zhangyang and Liang, Yingbin},
  journal={arXiv preprint arXiv:2312.01260},
  year={2023}
}
```

