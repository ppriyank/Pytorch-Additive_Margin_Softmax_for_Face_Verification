# Pytorch-Additive_Margin_Softmax_for_Face_Verification
Pytorch Implementation of AMSoftmax 

**Additive Margin Softmax for Face Verification**  
*Feng Wang, Weiyang Liu, Haijun Liu, Jian Cheng*  
https://arxiv.org/pdf/1801.05599.pdf


```
from AM_Softmax import AM_Softmax
features = Model(input)
criterion_xent = AM_Softmax()
Loss = criterion_xent(features, labels)
```


Since weights are in Loss function only, add these to the model optimizer as well.

```
for key, value in criterion_xent.named_parameters():
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
optimizer = torch.optim.Adam(params)
```
