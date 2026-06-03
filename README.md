# neural-network-custom
implementing Neural Network from scratch in C++

Instructions:
```
git clone https://github.com/AyushBajaj1/neural-network-custom
cd neural-network-custom
cd src
make
./main
```
If you run this, you will see the neural network trains well on sample_data.csv, reaching an error of about 6%. You can also look at the stream output and see the predicted and actual data points are relatively close for most data points. 

Also, if you're curious about how sample_data.csv was made, this is the formula I used:

column1 = random number -20 to 20

column2 = random number -20 to 20

column3 = log(abs(column1)) x sin(column1 x column2) + 15 - .1 x (column1^2) + .4 x column2

With column1 and column2 forming the x data, column3 forming the y data.
You can see that this is a very strange non linear function, (and if the formula was unknown then normal regression would be very difficult) showcasing the strength of the neural network, still being able to learn this pairing decently well. 
