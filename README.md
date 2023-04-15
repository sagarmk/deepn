# Deep Learning Visualization

This is a PyQt5-based desktop application that allows you to visualize the training and prediction of a simple neural network model.

## Requirements

 pip install -r requirements.txt

## Installation
![alt text](/images/deepqn.png "title")

1. Clone the repository to your local machine.
2. Install the required packages by running `pip install -r requirements.txt` in the project directory.
3. Run the application by executing `python main.py` in the project directory.

## Usage

The application allows you to specify the input data and model hyperparameters through a series of input widgets. Once the input data and hyperparameters are specified, you can train the model by clicking the "Fit" button, and visualize the predictions by clicking the "Predict" button.

The left panel of the application displays the input widgets, while the right panel displays a plot of the ground truth and predicted values.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.


## TODO

- [ ] Add support for saving and loading trained models.
- [ ] Add support for different activation functions.
- [ ] Add support for different optimization algorithms.
