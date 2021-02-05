# x-vector
A pytorch x-vector implementation following the paper:
Spoken Language Recognition using x-vector. 
Feel free to tell me if you find any errors in this repo.:)

1. The training is on multi-GPU by default using the pytorch DataParallel, if your device has only one GPU, pls comment the corresponding codes;
2. (For Freshman to pytorch/x-vector) To run the xv_training.py, pls check the args. Then use a command window to run the code, e.g.: python xv_training.py --dim 23 --data /home/freshman/data/train.txt --epoch 40 --lang 10

Please adjust the parameters according to your task!!!

My email is: HEXIN002@e.ntu.edu.sg, feel free to email me if you have any questions/suggestions!
