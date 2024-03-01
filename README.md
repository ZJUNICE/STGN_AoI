



## Introduction

This repository provides the codes of the paper **"Semantics-enhanced Temporal Graph Networks for Content Popularity Prediction"** and **"AoI-Based Temporal Attention Graph Neural Network for Popularity Prediction and Content Caching"**. 

Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange. We are sharing the codes under the condition that reproducing full or part of codes must cite our papers.

#### Paper link: 
  [AoI-Based Temporal Attention Graph Neural Network for Popularity Prediction and Content Caching](https://ieeexplore.ieee.org/document/9978680)

  [emantics-enhanced Temporal Graph Networks for Content Popularity Prediction](https://ieeexplore.ieee.org/document/10380461/)


## Feature

* An attention aggregator for the raw message processing.

* An AoI-based message filter with the attention aggregator.

* Different semantic aggregator for the representation learning.

## Running the experiments

### Requirements


```{bash}
pip install -r requirements.txt
```

### Dataset and Preprocessing

#### The public data
Download the wikipedia and reddit datasets from
[here](http://snap.stanford.edu/jodie/) and netflix dataset from [here](https://www.kaggle.com/datasets/vodclickstream/netflix-audience-behaviour-uk-movies). 

You can also use the data we saved in the folder *./data*.

#### Preprocess the data

```{bash}
python utils/netflix_process.py --bipartite --coder BERT
```



### Model Training

Training AoI-Based TGN:
```{bash}
# AoI Attention:
python train_self_supervised.py train_self_supervised.py --n_runs 5 --n_epoch 50 --aggregator attn --prefix TGN-A --data wikipedia --n_neighbor 6 --use_memory --use_age --bs 200
```

Training STGN:
```{bash}
# M2-STGN:
python train_self_supervised.py train_self_supervised.py --n_runs 5 --n_epoch 50 --aggregator attn --prefix STGN-A --data netflix --n_neighbor 6 --use_memory --use_age --bs 200 --Sem --mix Attn
```




## Cite us

```bibtex
@ARTICLE{9978680,
  author={Zhu, Jianhang and Li, Rongpeng and Ding, Guoru and Wang, Chan and Wu, Jianjun and Zhao, Zhifeng and Zhang, Honggang},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={AoI-Based Temporal Attention Graph Neural Network for Popularity Prediction and Content Caching}, 
  year={2023},
  volume={9},
  number={2},
  pages={345-358},
  doi={10.1109/TCCN.2022.3227920}}

@ARTICLE{10380461,
  author={Zhu, Jianhang and Li, Rongpeng and Chen, Xianfu and Mao, Shiwen and Wu, Jianjun and Zhao, Zhifeng},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Semantics-enhanced Temporal Graph Networks for Content Popularity Prediction}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TMC.2023.3349315}}
```


## Acknowledgements 

This repository is based on modifications and extensions of [TGN](https://github.com/twitter-research/tgn). We express our gratitude for the original contributions.

