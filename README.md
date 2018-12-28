A ongoing pytorch version of Transformer model to medical concept mapping.

`bert` embedding and spell check are unfinished.

## Requirements
* Python
* Pytorch
* Allennlp
* Pytorch version BERT

## Pretrained standard embedding

You can download ELMo and BERT pretrained models into `models` folder.

## Experiment

Once you have data files prepared, simply run:

	python code/train.py

There are two ways to apply ELMo embedding `Pretrain_type`:

	elmo_repre: using already weighted ELMo representation
	elmo_layer: get 3 layers of ELMo output and compute weights in the model

To reproduce the result:

```python
Spell_check = False  # unfinshed
Pretrain_type = 'elmo_repre'  # bert / elmo_repre / elmo_layer

Max_seq_len = 35
HealthVec_size = 200
Embedding_size = 200 + 768 if Pretrain_type == 'bert' else 200 + 1024
Hidden_size = 200
Inner_hid_size = 1024
D_k = 64
D_v = 64
Num_layers = 6
Num_head = 5
Dropout = 0.2

Learning_rate = 0.0001
Weight_decay = 0.0015
LR_decay = 0.5
Epoch = 600
LR_decay_epoch = 300
Batch_size = 128
```
