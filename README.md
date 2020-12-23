# Domain-Specific-IR
Information retrieval improved by a lot over the last year specially with pretrained language models.  But the main bottleneck is scarcity of domain specific QA pairs.

## This repositary closely look at unsupervised IR specially in relation to QA.

### By carefully analysing the previous work, we identfied three main areas of work. 

1. Craeting systems with exisiting IR models and collect data when users are interacting with them. Later such data get used to fine-tune systems.
2. Synthetic QA pair generation.
3. Concider already avaiable meta data in documents as user quaries and answers to train a system.


In this work, we are mainly interested in last two methods. 


### Synthetic QA pair generation.

The prior work so far, use a another QA dataset to train a generative tranformer to generate QA pairs given a content. The main issue in this method is , distribution mismatch between the dataset used to train and domain specific dataset. 
