%% ニューラルネットワークの重みとバイアスを読み取り、前馈制御に使用

w0 = readmatrix('result_NNFF&ROB\20251002_153045\z2_weights_layer_0.csv','Range','A1:AD4');
% wei1 = transpose(wei1);
b0 = readmatrix('result_NNFF&ROB\20251002_153045\z2_biases_layer_0.csv','Range','A1:A30');
w1 = readmatrix('result_NNFF&ROB\20251002_153045\z2_weights_layer_1.csv','Range','A1:T30');
% wei2 = transpose(wei2);
b1 = readmatrix('result_NNFF&ROB\20251002_153045\z2_biases_layer_1.csv','Range','A1:A20');
w2 = readmatrix('result_NNFF&ROB\20251002_153045\z2_weights_layer_2.csv','Range','A1:J20');
% wei3 = transpose(wei3);
b2 = readmatrix('result_NNFF&ROB\20251002_153045\z2_biases_layer_2.csv','Range','A1:A10');
w3 = readmatrix('result_NNFF&ROB\20251002_153045\z2_weights_layer_3.csv','Range','A1:A10');
% wei4 = transpose(wei4);
b3 = readmatrix('result_NNFF&ROB\20251002_153045\z2_biases_layer_3.csv','Range','A1:A1');