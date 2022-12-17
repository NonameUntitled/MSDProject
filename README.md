# Efficient transformers for financial data

**Authors:**  Baikalov Vladimir, Kovaleva Maria, Shlychkov Konstantin, Vo Ngoc Bich Uyen

## Problem

The attention-based methods and transformers made a significant breakthrough in the deep learning area and greatly impacted NLP task solutions [3]. Although recent works show that they could potentially improve results in different tasks domains, the application of transformer for financial data in particular transactions data is underexplored.

While applying attention mechanisms, one can face the apparent restriction on input sequence length due to the method's quadratic complexity. Recent papers proposed different ways to overcome this problem, but we want to concentrate on two promising approaches: Informer and Performer [1, 2].

The Informer is the most current and prospective approach. Its main assumption is that the model should have an "infinite memory" and fit a sequence with arbitrary length. The Performer model shows good results in the NLP task but is not well-explored for other datatypes. Its main idea is to use some trigonometric approximation of the attention matrix to decrease memory consumption.

To sum up, the project aims to compare several recent methods proposed to decrease the evaluation complexity in particular tasks predicting the user's gender based on transactions. 

## What have been done

 - baseline model (by Baikalov Vladimir)
 - training and data processing pipeline (by Baikalov Vladimir)
 - performer attention (by Shlychkov Konstantin)
 - informer attention (by Kovaleva Maria)
 - banchmarking all models in terms of speed and memory consumption (by Shlychkov Konstantin)
 - report (by all team members)
 - presentation (by all team members)
 
## Code

 You can see all models realization and results of experiments in [this](https://github.com/NonameUntitled/MSDProject/blob/results/notebooks/main.ipynb) notebook, also you can repeat by yourself. 
 
 Or run:
 
 ```
 python3 ./train.py --params ../configs/baseline_config_train.json
 python3 ./train.py --params ../configs/performer_config_train.json
 python3 ./train.py --params ../configs/informer_config_train.json
 ```
 for training (it is required to use cuda)
 
 and 
 
 ```
 python3 ./inference.py --params ../configs/baseline_config_inference.json
 python3 ./inference.py --params ../configs/performer_config_inference.json
 python3 ./inference.py --params ../configs/informer_config_inference.json
 ```
 for experiments.
 
## Results 

## Literature

[1] Martins, Pedro Henrique, Zita Marinho, and André FT Martins. "oo-former: Infinite Memory Transformer." - "Informer" model

[2] Choromanski, Krzysztof, et al. "Rethinking attention with performers." arXiv preprint arXiv:2009.14794 (2020). - "Performer" model

[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. - "Full attention" model



 

