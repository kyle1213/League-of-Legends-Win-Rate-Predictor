Second project

This model predicts which team is gonna win with only champion's IDs

What I've learned : the meaning of loss:0.6931(-ln0.6931 = 1/2, means network choose only one thing with two choices)

Limits : data input size was only 10(I think it's too small to predict something, while computer vision tasks have hundreads of input(pixels)s)

What I've tried : added batch normalization, dropout, input regularization, changed input shape(not size the style(?)), tried to use shortcut method but failed to apply, positional encoding

LOGs
2021 04 17 : Added multi head attention module, results seems similar with below.

2021 04 16 : Added single head attention module and reduced depth of fully connected layer. Test accuracy was always 41.33% before(75 test sets(31 A class, 44 B class), 31/75 = 41.33) but changed to 52%, sometimes 43.xx%. Attention module seems to work but performance was not very good(But mutch better I think)

2021 04 14 : Changed model back to original method(normal MLP) but added positional encoding to input vector(array). Didn't used same positional encoding from Transformer but added [0.1, 0.2, ... , 0.9, 1.0] or [0,1, 0,2, 0,3, 0.4, 0.5, -0.1, -0.2, ... , -0.5]. Both didn't work well.

2021 04 08 : First model was MLP with 10 input size array. Second model was 17 array inputs MLP, 5 was obtained from 2x1 matrix parameters(changed input array with size 10 into 5x2 matrix and matmul with 2x1 matrix parameters) and 2 was obtained from 5x1 matrix parameters(changed input array with size 10 into 2x5 matrix and matmul with 5x1 matrix parameters).
Third model is MLP with 10 input size array. Summed two 10 size array obtained by multiplying 5 size array obtained above with original input and 2 size array obtained above with original input.
Model choose only one answer to every input data(if answer 0 is more than 1, than model always says 0)
