# Pytorch-Additive_Margin_Softmax_for_Face_Verification
Pytorch Implementation of AMSoftmax 

**Additive Margin Softmax for Face Verification**  
*Feng Wang, Weiyang Liu, Haijun Liu, Jian Cheng*  
https://arxiv.org/pdf/1801.05599.pdf


```
from AM_Softmax import AM_Softmax_v1 , AM_Softmax_v2
features = Model(input)
criterion_xent = AM_Softmax_v1()
Loss = criterion_xent(features, labels)
```
or
```
criterion_xent = AM_Softmax_v2()
Loss = criterion_xent(features ,  labels, model.module.classifier )
```

\[Only for V_1\] Since weights are in Loss function only, add these to the model optimizer as well.

```
for key, value in criterion_xent.named_parameters():
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
optimizer = torch.optim.Adam(params)
```
