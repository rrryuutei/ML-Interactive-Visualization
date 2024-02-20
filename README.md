# Visualizing Neural Network Architecture
Team Members: Nora (Ting) Liu; Theodore Zhao

How is the Neural Network "Thinking"?
We visualize the internal status of a neural network predicting weekly sales amount of Walmart from a set of input values. The nodes in the network are the "neurons", activated by the input values at the left most 5 nodes. The neural network "thinks" by computing the neuron activation propagated from left to right through the weighted links, with the predicted sales amount reflected by the right most neuron.

Hover your mouse on it to see the predicted value!



How to Interact with the Neural Network?
1. Tune the input values for Holiday, Temperature, Fuel Price, CPI, and Unemployment at the top.

2. Observe how the neurons get activated differently (size change). The predicted sales value is the right most neuron!

3. Hover your mouse on a neuron to see its current value.

4. Use the selection bar at the bottom to filter the links in the neural network by their weights. Note how sparse the neural network is!

5. If you like, drag a node to reposition it. We provided three models of different sizes. Feel free to choose from the drop-down menu!

* The models were trained using data from Walmart Dataset on Kaggle.