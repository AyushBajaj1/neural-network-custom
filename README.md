# Neural-Network-Custom
- Custom Neural Network framework in C++
- Can train on regression and classification datasets
- Batch Gradient Descent, no optimizer
- Only C++ standard library was used, works on C++17 or later

## Architecture
The Neural Network class is built using a custom built Matrix class which provides all the useful linear algebra functions I needed to write the Neural Network 

#### Matrix Class:
- The Matrix class is just a 1d array of doubles on the heap under the hood, with rows and cols variables. The .at() method provides 2d indexing.
- Used a cache friendly matmul (traversing in ikj order rather than ijk to avoid cache misses that come with column wise traversal which is really just row sized jumps on a 1d array) Cache friendly matmul **brought time down from 146.58s to 38.79s** on MNIST (1 hidden layer w/ 300 neurons, batch_size 32, epochs 10).
- Also Implemented...
- scalarmul to apply learning rate to gradient matrices.
- splice_row and splice_col to do train test splits and batch splitting
- randfill for weight initialization. randfill uses a uniform real distribution for initialization, and you can provide an argument for the range.
- Implemented ReLU and Softmax activation functions, and heavside for relu backprop. hadamard_product to apply heavside during backprop.
- operator-= to decrement weights against the direction of the gradient
- column_wise_sum to calculate update matrix for the bias, and add_row_vector_inplace because I store the bias matrix as just a row vector, and this method essentially broadcasts that single row vector to an entire matrix, allowing the bias to be added to all rows of the matrix.
- I also created a lot of in place versions of mentioned methods to minimize creating temporary objects as much as possible. For some algorithms like matmul it wasn't possible.

#### Neural Network Class:
- Constructor: initializes weight, bias, and output matrices and their dimensions based on the number of neurons in each layer which is given by the user as a vector argument layer_sizes. Weight matrices are initialized with values ranging from -sqrt(6/fan_in) to +sqrt(6/fan_in) to achieve the desired variance for He-initalization. Biases are zero-initalized. I also select the output layer's activation based on the provided loss function (cross entropy -> softmax, mse -> linear)
- fit: This is the method which does the main training loop. User provides batch_size, learning_rate, and input-output data to train on, and the number of epochs. In each epoch, the network iterates through the entire dataset selecting batch_size rows at a time. If the batch_size does not evenly divide the dataset, the last couple examples are dropped, as that makes the math technically cleaner, and I noticed no difference in batching the last few examples vs ignoring them. (No accuracy difference to four decimal places on MNIST)
- forward_pass: A[n] = f(A[n-1] matmul W[n] + b[n]) is essentially the formula I used to calculate each output layer. A[-1] would be X (input data) and f is either relu or softmax. And b[n] is a row vector so to add it to a Matrix I used a specific add_row_vector method which essentially broadcast adds the row vector to the matrix.
- backward_pass:
-   1. L = (Y_hat - Y)/batch_size
    2. Weight_Update[i] = A[i-1].transpose matmul L
    3. Bias_Update[i] = L.columnwise_sum()
    4. L starts as the Loss gradient matrix wrt the final output, but with each backwards iteration starting from the end, L = (L matmul W[i].transpose) hadamard_multiply A[i-1]. This ensures the Loss gradient is updated via chain rule to be wrt the current layer's pre-activation output (assuming relu is the activation). Then the formulas from above are used to calculate the Loss Gradient wrt the weight and bias matrices. Handwritten derivations for these formulas done by me can be found at the bottom of this readme.
    5. Finally, each weight and bias matrix is subtracted by its corresponding update matrix, stepping against the direction of the gradient
 - predict: Just runs a forward pass on given input data and returns predicted output.


## Download and Running Instructions
```
git clone https://github.com/AyushBajaj1/neural-network-custom
cd neural-network-custom
make custom && ./custom
make london && ./london
make mnist && ./mnist
```
The first two make commands (custom and london) can be ran right away but mnist requires downloading a data set.

If you run `make custom && ./custom`, you will see the neural network trains well on custom_data.csv, reaching an NMAE (Normalized Mean Absolute Error) of about 3.22%. You can also look at the stream output and see the predicted and actual data points are relatively close for most data points. NMAE is normalized with respect to the column-wise range of the data.

This is the formula I used to make custom_data.csv:

column1 = random number -20 to 20

column2 = random number -20 to 20

column3 = log(abs(column1)) x sin(column1 x column2) + 15 - .1 x (column1^2) + .4 x column2

You can see that this is a very strange non linear function, showcasing the strength of the neural network in being able to learn this pairing almost perfectly.


I also trained the neural network on two well known datasets, MNIST and London Housing to be able to benchmark the performance of this custom neural network implementation.

Run `make london && ./london` and you'll see London Housing achieves an NMAE of about 3.6% and a MAE of £352,467. On Pytorch, I achieved an MAE of £384,284 on the same dataset with a similar network architecture (32 neuron hidden layer, epochs = 1000, batch_size=40)

To try MNIST, go to `https://drive.google.com/drive/folders/1GmyWeEzBR7ikjsr7kbAo2MoTanEorA_c?usp=sharing`, drag the data folder and place it in this directory, replacing the current data folder. Then `make mnist && ./mnist` will work fine. You'll see MNIST achieves a 97.23% accuracy, which is around the traditional benchmark of 98%. If you increase the number of hidden neurons in mnist.cpp from 64 to 128, the accuracy will climb to 97.8%, but the runtime will slow down.

Derivations provided here:
`https://drive.google.com/drive/folders/1lXvpbJF-DEuJU4dRWbaEo3l1HVAnCquD`

