%% run_external_and_save_io.m  — External modeで t, u, y を必ず保存
clear; clc;

% === ユーザー設定 ===
model   = 'system_id_new';   % 使う .slx のモデル名
SimTime = 60;                % 実行時間 [sec]（必要に応じて調整）
outdir  = fullfile(pwd, 'logs_io');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% === モデルを読み込み & External mode に設定 ===
load_system(model);
set_param(model, 'SimulationMode', 'external');  % External mode で走らせる
set_param(model, 'StopTime', num2str(SimTime));

% === 信号ロギング有効化（logsout を使う） ===
set_param(model, 'SignalLogging', 'on');

% === 1) 入力ブロック（input_test / input_text）を探してログ名を u に固定 ===
blk_candidates = [ ...
    find_system(model, 'SearchDepth', inf, 'Regexp', 'off', 'Name', 'input_test'); ...
    find_system(model, 'SearchDepth', inf, 'Regexp', 'off', 'Name', 'input_text') ...
];
blk_candidates = unique(string(blk_candidates));
if isempty(blk_candidates)
    warning('ブロック "input_test" / "input_text" が見つかりません。入力 u の取得に失敗する可能性があります。');
else
    blk = blk_candidates(1);
    ph  = get_param(blk, 'PortHandles');
    if isfield(ph,'Outport') && ~isempty(ph.Outport)
        % 出力ポートに信号ロギングをON
        set_param(ph.Outport(1), 'DataLogging', 'on');
        % ログ名を u に固定（可能なら）
        try
            set_param(ph.Outport(1), 'DataLoggingNameMode', 'Custom');
            set_param(ph.Outport(1), 'DataLoggingName', 'u');
        catch
            warning('入力ポートの DataLoggingName 設定に失敗。既定名でログします。');
        end
    else
        warning('"%s" の Outport が見つかりません。', blk);
    end
end

% === 2) ルート Outport の信号をすべてロギング（y1, y2, ...） ===
outports = find_system(model, 'SearchDepth', 1, 'BlockType', 'Outport');
for i = 1:numel(outports)
    ph = get_param(outports{i}, 'PortHandles');
    % Outport ブロックは「入力ポート（Inport）」側に信号が入ってくる
    if isfield(ph,'Inport') && ~isempty(ph.Inport)
        set_param(ph.Inport(1), 'DataLogging', 'on');
        try
            set_param(ph.Inport(1), 'DataLoggingNameMode', 'Custom');
            set_param(ph.Inport(1), 'DataLoggingName', sprintf('y%d', i));
        catch
            % 名前を付けられない環境でもログ自体は残る
        end
    end
end
if isempty(outports)
    warning('ルート Outport が見つかりません。出力 y が取得できない可能性があります。');
end

% === 3) 既存の logsout をクリア（古い残骸を避ける） ===
evalin('base', 'clear logsout');

% === 4) External mode で開始 → 所定時間待機 → 停止 ===
set_param(model, 'SimulationCommand', 'start');
pause(SimTime + 1.0);  % 少し余裕を持たせる
try
    set_param(model, 'SimulationCommand', 'stop');
catch
end

% === 5) logsout から t, u, y* を取り出し ===
% External mode では結果は base workspace の logsout に入る
if evalin('base', 'exist(''logsout'',''var'')')
    ds = evalin('base', 'logsout');  % Simulink.SimulationData.Dataset
else
    error('logsout が見つかりません。信号ロギング設定を確認してください。');
end

% まず u を取得（名前 'u' を最優先、なければ最初の要素をフォールバック）
u = []; t_u = [];
try
    el_u = [];
    try
        el_u = ds.get('u');
    catch
        el_u = [];
    end
    if isempty(el_u) && ds.numElements > 0
        % フォールバック：input_test に近い名前を探す
        names = strings(1, ds.numElements);
        for k = 1:ds.numElements
            try
                names(k) = string(ds{k}.Name);
            catch
                names(k) = "";
            end
        end
        idx = find(contains(lower(names), "input") | contains(lower(names), "u"), 1, 'first');
        if ~isempty(idx), el_u = ds{idx}; end
    end
    if ~isempty(el_u)
        vals = el_u.Values;  % timeseries 期待
        if isa(vals, 'timeseries')
            t_u = vals.Time;
            u   = vals.Data;
        elseif isprop(vals, 'Time') && isprop(vals, 'Data')
            t_u = vals.Time; u = vals.Data;
        end
    end
catch ME
    warning('u の抽出に失敗: %s', ME.message);
end

% 次に y1, y2, ... を集めて結合
y_list = {};
t_ref  = [];   % 代表時間軸
for i = 1:ds.numElements
    try
        name_i = "";
        try, name_i = string(ds{i}.Name); end
        if startsWith(name_i, "y") || startsWith(name_i, "Y")
            vals = ds{i}.Values;
            if isa(vals, 'timeseries')
                if isempty(t_ref), t_ref = vals.Time; end
                y_list{end+1} = vals; %#ok<AGROW>
            end
        end
    catch
        % スキップ
    end
end

% 時間軸の選定：u が取れていれば u の時間、なければ y の最初の時間
t = [];
if ~isempty(t_u)
    t = t_u;
elseif ~isempty(y_list)
    t = y_list{1}.Time;
else
    error('時間軸が取得できませんでした（u も y も空）。ロギング設定をご確認ください。');
end

% y 列を t に合わせて結合（複数出力対応）
y = [];
if ~isempty(y_list)
    for k = 1:numel(y_list)
        % 時間軸を t に合わせてリサンプル
        try
            yk = resample(y_list{k}, t);
            col = yk.Data;
        catch
            % 時間軸が完全一致ならそのまま
            col = y_list{k}.Data;
            % 長さが違うならトリム/パディング
            L = min(numel(t), size(col,1));
            col = col(1:L, :);
            if numel(t) > L
                col = [col; nan(numel(t)-L, size(col,2))];
            end
        end
        % 列方向に積む（ベクトル/多次元対応）
        if isvector(col)
            col = col(:);
        end
        if isempty(y)
            y = col;
        else
            % 次元を合わせて横に結合
            L = min(size(y,1), size(col,1));
            y = y(1:L, :);
            col = col(1:L, :);
            y = [y, col]; %#ok<AGROW>
            t = t(1:L);
        end
    end
end

% === 6) サマリと保存 ===
if isempty(u), warning('入力 u が空です。input_test のロギング設定/接続を確認してください。'); end
if isempty(y), warning('出力 y が空です。Outport のロギング設定/接続を確認してください。'); end

dt_guess = [];
if numel(t) >= 2, dt_guess = median(diff(t)); end

stamp   = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
matname = fullfile(outdir, sprintf('io_external_%s.mat', stamp));
save(matname, 't', 'u', 'y', 'dt_guess', '-v7.3');

fprintf('\n[保存完了] %s\n', matname);
fprintf('  t: %d samples (%.6g → %.6g sec), 推定 dt ≈ %s sec\n', ...
    numel(t), t(1), t(end), ternary(isempty(dt_guess),'N/A',num2str(dt_guess)));
fprintf('  u: %s\n', ternary(isempty(u),'--- (未取得)', sprintf('%d × %d', size(u,1), size(u,2))));
fprintf('  y: %s\n', ternary(isempty(y),'--- (未取得)', sprintf('%d × %d', size(y,1), size(y,2))));

% 三項演算子的ヘルパ
function out = ternary(cond,a,b), if cond, out=a; else, out=b; end
end
