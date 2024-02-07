
Could also just watch somebody else go through the process of TensorFlow CNN implementation


A high training accuracy alongside a much lower test accuracy typically indicates that the model is overfitting. Overfitting happens when a model learns the training data too well, including noise and details that do not generalize to new data. Here are several strategies you can use to combat overfitting:

Data Augmentation: This increases the diversity of your training data by applying random transformations like rotation, zoom, flip, etc. It can make your model more robust and less likely to memorize the training data.

Regularization: Apply regularization techniques such as L1 or L2 regularization, which add a penalty for larger weights in the model.

Dropout: Increase the dropout rate in your model. Dropout layers randomly set input units to 0 with a frequency of rate at each step during training, which helps to prevent overfitting.

Reduce Model Complexity: Simplify your model by reducing the number of layers or the number of neurons in the layers, so it's less likely to learn the noise in the training set.

Early Stopping: Use early stopping to end training before the model becomes too specialized to the training data. You monitor the performance on a validation set and stop the training when performance on the validation set begins to degrade.

Use Pre-trained Networks: Transfer learning from pre-trained networks can also help, as these networks have already learned a good set of features for image tasks.

Batch Normalization: Although you're already using it, you can experiment with its placement in the architecture or its parameters.

Cross-validation: Instead of a simple train-test split, use k-fold cross-validation to ensure that the model's performance is consistent across different subsets of the data.

Class Weights: If your dataset is imbalanced, use class weights to give more importance to less frequent classes during the training.

Learning Rate: Adjust the learning rate. Sometimes a smaller learning rate can prevent overfitting as the model makes smaller changes to the weights during training.

Modify Loss Function: For an imbalanced dataset, you can use a loss function that accounts for the imbalance, such as focal loss.

Increase Image Size: If computational resources allow, use a larger input image size to preserve more information during training.

Advanced Optimizers: Experiment with different optimizers and their parameters (like AdamW, RMSprop).