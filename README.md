This is an LSTM model built entirely from scratch using only the numpy library.

 I intend to provide a detailed explanation of each line of code in this repository in the future.

Currently, the model is trained on text data sourced from the Panchatantra stories, which I manually collected from online books and blogs. I've opted for the ADAM optimizer in this model, though I believe I used Adagrad in a previous vanilla RNN implementation.

My understanding of RNNs and LSTMs was developed through studying blogs by Colah and Andrej Karpathy, followed by analyzing various LSTM architecture implementations, which ultimately led me to create this model.

Due to my laptop's limitations, I've been unable to fully train the model as it tends to overheat quickly. As a result, I haven't yet included a prediction function, which would typically allow the model to predict the next 'n' letters based on given input after training. This prediction function would be similar to the sampling function and could be implemented if you have access to sufficient GPU resources.

I plan to train this model on a GPU when I gain access to one through my teacher and will update the repository with the prediction code at that time.

Hope you gonna enjoy the code