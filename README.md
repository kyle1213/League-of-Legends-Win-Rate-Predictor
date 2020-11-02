Second project

This model predicts which team is gonna win with only champion's IDs

What I've learned : the meaning of loss:0.6931(-ln0.6931 = 1/2, means network choose only one thing with two choices)

Limits : data input size was only 10(I think it's too small to predict something, otherwise in computer vision they have hundreads of input(pixels)s)

What I've tried : added batch normalization, input regularization, changed input shape(not size the style(?))