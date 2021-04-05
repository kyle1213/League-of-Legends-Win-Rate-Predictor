Second project

This model predicts which team is gonna win with only champion's IDs

1.
What I've learned : the meaning of loss:0.6931(-ln0.6931 = 1/2, means network choose only one thing with two choices)

Limits : data input size was only 10(I think it's too small to predict something, while computer vision tasks have hundreads of input(pixels)s)

What I've tried : added batch normalization, dropout, input regularization, changed input shape(not size the style(?)), tried to use shortcut method but failed to apply



2. MLP(fully connected layer) models need sequential informations if the data is sequential data. So I need to use RNN based model or Transformer(Multi-head attention) because this project's data seem sequential(team, line)
