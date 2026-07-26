# neural-network-custom
Custom Neural Network framework in C++. Can train on regression and classification datasets, with examples below.

## download instructions
```
git clone https://github.com/AyushBajaj1/neural-network-custom
cd neural-network-custom
make custom && ./custom
make london && ./london
make mnist && ./mnist
```
The first two make commands (custom and london) can be ran right away but mnist requires downloading a data set.

If you run this, you will see the neural network trains well on custom_data.csv, reaching an NMAE (Normalized Mean Absolute Error) of about 3.22%. You can also look at the stream output and see the predicted and actual data points are relatively close for most data points. 

This is the formula I used to make custom_data.csv:

column1 = random number -20 to 20
column2 = random number -20 to 20
column3 = log(abs(column1)) x sin(column1 x column2) + 15 - .1 x (column1^2) + .4 x column2

You can see that this is a very strange non linear function, showcasing the strength of the neural network in being able to learn this pairing almost perfectly.

I also trained the neural network on two well known datasets, MNIST and London Housing to be able to benchmark the performance of this custom neural network implemenation.

Run `make london && ./london` and you'll see London Housing achieves an NMAE of about 3.6% and a MAE of £352,467. On Pytorch, I achieved an MAE of £384,284 on the same dataset with a similiar network architecture. 

To try MNIST, go to `https://drive.google.com/drive/folders/1GmyWeEzBR7ikjsr7kbAo2MoTanEorA_c?usp=sharing`, drag the data folder and place it in this directory, replacing the current data folder. Then make `mnist && ./mnist` will work fine. You'll see MNIST achieves a 97.23% accuracy, which is around the traditional benchmark of 98%.

