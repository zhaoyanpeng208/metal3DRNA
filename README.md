Metal3DRNA is a method for the prediction of metal ion binding sites in RNA structure.
====
Authors are YanPeng Zhao, Weikang Gong, Jingjing Wang, Yang Liu and Chunhua Li.<br> 
The pdb, model and result directories contain tested structures, the trained model and generated results, respectively. The depended python packages are listed in requirements.txt. The package versions should be followed by users in their environments to achieve the supposed performance.<br> 

How to run
---
The program is in Python 3.6 using Tensorflow 2.0 backends. <br> 
The first step: Put the PDB files to be predicted into the PDB folder<br> 
The second step: Use the below bash command to run Metal3DRNA.<br> 

    python independent_test.py -s structure -m model(optional)
    
The parameter of -s should be the file name of the PDB structure you want to predict.<br> 
The parameter of -m could be mg, na and k which means the ion type you want to predict. <br>  <br>
Then, Metal3DRNA will perform predict process, this will take a few minutes.<br> 
The results will be given in the results folder, named the model of the ion type append the time when the prediction began.<br> 

Description of result file
---
After the prediction is completed, you will see the folder in which you made the prediction under the Result folder, which contains an Excel file for your prediction results. <br>
The first column of Excel is the 3D coordinate in the PDB file of the grid point, the second column is the file name, the third column is the probability of the gird point which predictd as negative, and the fourth column is the probability of the grid point which predicted as positive. 

Help
---
For any questions, feel free to contact me by zhaoyp@emails.bjut.edu.cn or chunhuali@bjut.edu.cn or start an issue instead.


