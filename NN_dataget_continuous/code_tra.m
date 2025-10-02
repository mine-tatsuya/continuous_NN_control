clear
close all
time = 800;
Td = 0.001;
Ts_log = 0.001;

%% ランダムなu(t)入力
seed = randi(2^31-2);     % 毎回違うSeed;
ran_min = 5;
ran_max = 20;
sam_time = 40; % 入力切替周期

%% 初期信号生成器
numerator_0 = 1;
denominator_0 = [1 2 1];
p0_s = tf(numerator_0, denominator_0);

%% 初期ランダム信号生成器 - ランダムステップ信号を使用して真のr(t)を作る信号生成器
A0 = [0, 1; -1, -1.4];
B0 = [0; 1];
C0 = [1, 0];
D0 = [0];

%% 線形離散制御対象（双線形変換）
A = [0.7, 0.2; 0, 0.5];
B = [1; 0.5];
C = [0.2, 0.8];
D = [0];
[numerator, denominator] = ss2tf(A, B, C, D);

%% 統合理想概強正実（絶対正実安定）モデル
num_aspr = [1, -0.6, 0.61];
den_aspr = [1, -1.2, 0.35];

%% モデルを実行
sim("dataget_continuous.slx");


%% CSV形式に変換（ヘッダー付き）
load('dataget', 'dataget');

% 行列を転置（6行×400001列 → 400001行×6列）
dataget = dataget.';

% テーブル化（列名を指定）
T = array2table(dataget, ...
    'VariableNames', {'t','y','yd','ydd','yddd','r'});

% CSV出力（ヘッダー付き）
writetable(T, 'dataget.csv');

