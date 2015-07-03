## Experiments Results

### Exp4, train 14, test 22, 74 epochs

#### 3 clusters, soft_sum_pred_cm, greedy clustering
(Note: this was done on the test set, not validation set.)
Generalist: logloss: 0.68272, accuracy: 0.772435
spec logloss: 0.396276 / 0.1594188 / 4.252624
spec accuracy: 0.0 / 0.023 / 0.358

*   Unweighted merging: logloss 0.6593994, accuracy 0.794070
*   Weighted merging: (np.max(g)) logloss 0.68704, accuracy 0.79537
    
