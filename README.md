# ERA-SESSION15 Vanilla Transformer Implementation in PytorchLightning

### Achieved:
1. **Training Loss: 2.920**
2. **CER Score: **
3. **BLEU Score: **
4. **WER Score: **

### Tasks:
1. :heavy_check_mark: Rewrite the whole code covered in the class in Pytorch-Lightning
2. :heavy_check_mark: Train the model for 10 epochs
3. :heavy_check_mark: Achieve a loss of less than 4

### Results
![image](https://github.com/RaviNaik/ERA-SESSION15/assets/23289802/686670d4-e142-498f-9329-ffc7a72da57f)
**Note:** Detailed results are presnt in results folder as a CSV file

### Model Summary
```python
   | Name                                    | Type               | Params
--------------------------------------------------------------------------------
0  | transformer                             | Transformer        | 75.1 M
1  | transformer.encoder                     | Encoder            | 18.9 M
2  | transformer.encoder.layers              | ModuleList         | 18.9 M
3  | transformer.encoder.norm                | LayerNormalization | 2     
4  | transformer.decoder                     | Decoder            | 25.2 M
5  | transformer.decoder.layers              | ModuleList         | 25.2 M
6  | transformer.decoder.norm                | LayerNormalization | 2     
7  | transformer.src_embed                   | InputEmbeddings    | 8.0 M 
8  | transformer.src_embed.embedding         | Embedding          | 8.0 M 
9  | transformer.tgt_embed                   | InputEmbeddings    | 11.5 M
10 | transformer.tgt_embed.embedding         | Embedding          | 11.5 M
11 | transformer.src_pos                     | PositionalEncoding | 0     
12 | transformer.src_pos.dropout             | Dropout            | 0     
13 | transformer.tgt_pos                     | PositionalEncoding | 0     
14 | transformer.tgt_pos.dropout             | Dropout            | 0     
15 | transformer.projection_layer            | ProjectionLayer    | 11.5 M
16 | transformer.projection_layer.projection | Linear             | 11.5 M
17 | loss_fn                                 | CrossEntropyLoss   | 0     
18 | cer_metric                              | CharErrorRate      | 0     
19 | wer_metric                              | WordErrorRate      | 0     
20 | bleu_metric                             | BLEUScore          | 0     
--------------------------------------------------------------------------------
75.1 M    Trainable params
0         Non-trainable params
75.1 M    Total params
300.532   Total estimated model params size (MB)
```
### Loss & Other Metrics
**Training Loss:**
![image](https://github.com/RaviNaik/ERA-SESSION15/assets/23289802/917bd1db-01c3-4f40-8c2a-47920648d717)

**CER, WER & BLEU Score:**
![image](https://github.com/RaviNaik/ERA-SESSION15/assets/23289802/ddff32f5-61e1-41db-903b-b8235e576c34)


### Tensorboard Plots 
![image](https://github.com/RaviNaik/ERA-SESSION15/assets/23289802/61517ac4-9abf-438e-93bd-cf16694a620a)
