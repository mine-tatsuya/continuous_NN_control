clear
close all

time   = 2000;
Td     = 0.005;
Ts_log = 0.005;

%% ランダムなu(t)入力（各回でseedは更新します）
seed = randi(2^31-2);     % 初期Seed（あとで各ランごとに更新）
ran_min = 5;
ran_max = 20;
omega = 3.14159265*0.2;
sam_time = 32;            % 入力切替周期
wav = 0.7;
sigma = 0.1;
gamma = 10000;

%% 初期信号生成器
numerator_0 = 1;
denominator_0 = [1 2 1];
p0_s = tf(numerator_0, denominator_0);

 %% 初期ランダム信号生成器 - ランダムステップ信号を使用して真のr(t)を作る信号生成器
% A0 = [0, 1; -1, -1.4];
% B0 = [0; 1];
% C0 = [1, 0];
% D0 = [0];
% 
 %% 線形離散制御対象（双線形変換）
% A = [0.7, 0.2; 0, 0.5];
% B = [1; 0.5];
% C = [0.2, 0.8];
% D = [0];
% [numerator, denominator] = ss2tf(A, B, C, D);

%% 統合理想概強正実（絶対正実安定）モデル
num_aspr = [1, -0.6, 0.61];
den_aspr = [1, -1.2, 0.35];

%% ===== ここから：シナリオ順に20回実行してCSVに追記 =====
% スクリプトの場所を基準に出力先フォルダを取得
script_path = fileparts(mfilename('fullpath'));  % dataget_code_2.mのあるフォルダ
output_dir = fullfile(script_path, '..');  % 親フォルダ（NN_dataget_continuous）
outCsv = fullfile(output_dir, 'dataget.csv');
outMat = fullfile(output_dir, 'dataget.mat');
outCsv_PFC = fullfile(output_dir, 'dataget_PFC.csv');
outMat_PFC = fullfile(output_dir, 'dataget_PFC.mat');
% 注: writetable の 'overwrite' モードで最初に上書きするため、削除不要

scenarios = [ 0 0;   % [dis_switch, sin_switch]
              1 0;
              0 1;
              1 1 ];
repeats = 5;         % 4シナリオ×5 = 20回
firstWrite = true;
firstWrite_PFC = true;

for rep = 1:repeats
    for s = 1:size(scenarios,1)
        % シナリオ設定
        dis_switch = scenarios(s,1);
        sin_switch = scenarios(s,2);

        % 各ランでseed更新（必要ならモデル内で利用）
        seed = randi(2^31-2);

        % モデルが base ワークスペースの変数を参照している前提で明示的に渡す
        assignin('base','time',time);
        assignin('base','Td',Td);
        assignin('base','Ts_log',Ts_log);
        assignin('base','ran_min',ran_min);
        assignin('base','ran_max',ran_max);
        assignin('base','sam_time',sam_time);
        assignin('base','wav',wav);
        assignin('base','dis_switch',dis_switch);
        assignin('base','sin_switch',sin_switch);
        assignin('base','seed',seed);

        % 取り違え防止：古いデータが base に残っていたら消す
        evalin('base','if exist(''dataget'',''var''), clear(''dataget''); end');
        evalin('base','if exist(''dataget_PFC'',''var''), clear(''dataget_PFC''); end');

        %% モデルを実行（結果は base.workspace の 'dataget' に出力される前提）
        % カレントディレクトリを親フォルダに変更してから実行（dataget.matが正しい場所に保存されるように）
        original_dir = pwd;
        cd(output_dir);
        slx_path = fullfile(script_path, 'dataget_continuous_2.slx');
        try
            sim(slx_path);
        catch ME
            cd(original_dir);  % エラーが発生してもディレクトリを戻す
            rethrow(ME);
        end
        cd(original_dir);  % カレントディレクトリを元に戻す
 
        %% === dataget の取得：base → MAT(あれば) → 取得不能ならスキップ ===
        gotData = false;
        if evalin('base','exist(''dataget'',''var'')')
            dataget = evalin('base','dataget');
            gotData = true;
        elseif exist(outMat,'file')
            S = load(outMat,'dataget');
            if isfield(S,'dataget')
                dataget = S.dataget;
                gotData = true;
            end
        end

        %% === dataget_PFC の取得：base → MAT(あれば) → 取得不能ならスキップ ===
        gotData_PFC = false;
        if evalin('base','exist(''dataget_PFC'',''var'')')
            dataget_PFC = evalin('base','dataget_PFC');
            gotData_PFC = true;
        elseif exist(outMat_PFC,'file')
            S_PFC = load(outMat_PFC,'dataget_PFC');
            if isfield(S_PFC,'dataget_PFC')
                dataget_PFC = S_PFC.dataget_PFC;
                gotData_PFC = true;
            end
        end

        %% === 両方のデータが取得できているか確認 ===
        if ~gotData
            warning('dataget が取得できませんでした（rep=%d, scenario=[%d,%d]）。このランはスキップします。', ...
                rep, dis_switch, sin_switch);
            continue;
        end
        if ~gotData_PFC
            warning('dataget_PFC が取得できませんでした（rep=%d, scenario=[%d,%d]）。このランはスキップします。', ...
                rep, dis_switch, sin_switch);
            continue;
        end

        %% ========== 検証フェーズ：両方のデータを先に検証 ==========
        
        %% === 整形と検証：dataget（6行×N or N×6 を N×6 に統一）===
        if ndims(dataget) ~= 2
            warning('dataget の次元が想定外です（rep=%d, scenario=[%d,%d]）。スキップします。', ...
                rep, dis_switch, sin_switch);
            continue;
        end

        if size(dataget,1) == 6 && size(dataget,2) ~= 6
            dataget = dataget.';   % 6×N → N×6
        end

        % ここで N×6（t,y,yd,ydd,yddd,u）になっていることが前提
        if size(dataget,2) ~= 6
            warning('dataget の列数が %d です。期待は6列（t,y,yd,ydd,yddd,u）。スキップします。', size(dataget,2));
            continue;
        end

        % 数値型を保証（timeseries/structで来た場合の保険）
        if ~isnumeric(dataget)
            try
                dataget = double(dataget);
            catch
                warning('dataget を数値配列に変換できませんでした。スキップします。');
                continue;
            end
        end

        %% === 整形と検証：dataget_PFC（7行×N or N×7 を N×7 に統一）===
        if ndims(dataget_PFC) ~= 2
            warning('dataget_PFC の次元が想定外です（rep=%d, scenario=[%d,%d]）。スキップします。', ...
                rep, dis_switch, sin_switch);
            continue;
        end

        if size(dataget_PFC,1) == 7 && size(dataget_PFC,2) ~= 7
            dataget_PFC = dataget_PFC.';   % 7×N → N×7
        end

        % ここで N×7（t,ud,udd,yad,yadd,ya,y）になっていることが前提
        if size(dataget_PFC,2) ~= 7
            warning('dataget_PFC の列数が %d です。期待は7列（t,ud,udd,yad,yadd,ya,y）。スキップします。', size(dataget_PFC,2));
            continue;
        end

        % 数値型を保証（timeseries/structで来た場合の保険）
        if ~isnumeric(dataget_PFC)
            try
                dataget_PFC = double(dataget_PFC);
            catch
                warning('dataget_PFC を数値配列に変換できませんでした。スキップします。');
                continue;
            end
        end

        %% ========== 保存フェーズ：両方のデータが検証済みなので保存 ==========
        
        %% === dataget を CSVに追記 ===
        T = array2table(dataget, ...
            'VariableNames', {'t','y','yd','ydd','yddd','u'});

        if firstWrite
            writetable(T, outCsv, 'WriteMode','overwrite');    % ヘッダーあり
            firstWrite = false;
        else
            writetable(T, outCsv, 'WriteMode','append', 'WriteVariableNames', false); % 追記（ヘッダーなし）
        end

        %% === dataget_PFC を CSVに追記 ===
        T_PFC = array2table(dataget_PFC, ...
            'VariableNames', {'t','ud','udd','yad','yadd','ya','y'});

        if firstWrite_PFC
            writetable(T_PFC, outCsv_PFC, 'WriteMode','overwrite');    % ヘッダーあり
            firstWrite_PFC = false;
        else
            writetable(T_PFC, outCsv_PFC, 'WriteMode','append', 'WriteVariableNames', false); % 追記（ヘッダーなし）
        end
    end
end
%% ===== 追加ここまで =====

disp('>> 指定順 [0,0]→[1,0]→[0,1]→[1,1] を5回繰り返し、dataget.csv と dataget_PFC.csv に追記しました。')
fprintf('出力ファイル:\n  - %s\n  - %s\n  - %s\n  - %s\n', outCsv, outMat, outCsv_PFC, outMat_PFC);
