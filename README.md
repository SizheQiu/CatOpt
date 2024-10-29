# Seq2pHopt_v2.0
A deep learning-based predictor of enzyme optimal pH
Protein sequence -> enzyme optimal pH <br>
Accuracy: RMSE=0.833, R2=0.479. <br>
Seq2pHopt-v1.0 can be found at https://github.com/SizheQiu/Seq2Topt/blob/main/code/seq2pHopt.py. <br>
## How to use:
1. Prepare the input file: a CSV file containing a column "sequence" for protein sequences.<br>
2. Enter `/code` directory and run prediction: <br>
```
python seq2pHopt2.py --input [input.csv] --output [output file name]
```
## Dependency:
1.Pytorch: https://pytorch.org/<br>
2.ESM: https://github.com/facebookresearch/esm<br>
3.Scikit-learn: https://scikit-learn.org/<br>
4.Seaborn statistical data visualization:https://seaborn.pydata.org/index.html<br>
## Citation
