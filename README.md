# CatOpt
A deep learning-based predictor of enzyme optimal pH <br>
Protein sequence -> enzyme optimal pH <br>
Accuracy: RMSE=0.833, R2=0.479. <br>
The comparison of CatOpt, [Seq2pHopt](https://github.com/SizheQiu/Seq2Topt), [EpHod](https://github.com/jafetgado/EpHod), and [OphPred](https://github.com/i-Molecule/optimalPh) can be found in `/data/comparison`. <br>
## How to use:
1. Prepare the input file: a CSV file containing a column "sequence" for protein sequences.<br>
2. Enter `/code` directory and run prediction: <br>
```
python catopt.py --task (pHopt,topt,tm) pHopt --input [input.csv] --output [output file name]
```
3. Train the model: <br>
```
python run_diffparams.py --train_path [trainset.csv] --test_path [testset.csv]
```
The output path and hyperparameters need to be manually edited.
## Dependency:
1.Pytorch: https://pytorch.org/<br>
2.ESM: https://github.com/facebookresearch/esm<br>
3.Scikit-learn: https://scikit-learn.org/<br>
4.Seaborn statistical data visualization:https://seaborn.pydata.org/index.html<br>
## Citation
Sizhe Qiu, Yishun Lu, Nan-Kai Wang, Jin-Song Gong, Jin-Song Shi, Aidong Yang, Deep learning-based prediction of enzyme optimal pH and design of point mutations to improve acid resistance, https://doi.org/10.1101/2024.11.16.623957
