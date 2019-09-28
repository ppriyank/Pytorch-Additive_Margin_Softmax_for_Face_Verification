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
