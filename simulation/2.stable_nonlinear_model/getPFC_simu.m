%% 非線形状況下での前馈補償器ニューラルネットワークの重みとバイアスを読み取り、前馈補償制御に使用

w0pfc = readmatrix('NNPFC_tra\z2pfc_weights_layer_0.csv','Range','A1:AD4');
% wei1 = transpose(wei1);
b0pfc = readmatrix('NNPFC_tra\z2pfc_biases_layer_0.csv','Range','A1:A30');
w1pfc = readmatrix('NNPFC_tra\z2pfc_weights_layer_1.csv','Range','A1:T30');
% wei2 = transpose(wei2);
b1pfc = readmatrix('NNPFC_tra\z2pfc_biases_layer_1.csv','Range','A1:A20');
w2pfc = readmatrix('NNPFC_tra\z2pfc_weights_layer_2.csv','Range','A1:J20');
% wei3 = transpose(wei3);
b2pfc = readmatrix('NNPFC_tra\z2pfc_biases_layer_2.csv','Range','A1:A10');
w3pfc = readmatrix('NNPFC_tra\z2pfc_weights_layer_3.csv','Range','A1:A10');
% wei4 = transpose(wei4);
b3pfc = readmatrix('NNPFC_tra\z2pfc_biases_layer_3.csv','Range','A1:A1');