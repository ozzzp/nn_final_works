# Othello 翻转棋
系统要求 python3, pip 要求：

- pytorch
- numpy
- Cython

训练步骤:
````
cd Othello/Othello
python3 setup.py build_ext --inplace
cd ..
python3 main.py --mode Train
````

评估步骤:
````
cd Othello/Othello
python setup.py build_ext --inplace
cd ..
python3 main.py --mode Eval --pertrain V8
````
V8 为预训练集，在目录vp下.