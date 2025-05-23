clear all
%close all


%% Changable parameters
    N_d = 100;% # of sampling frequencies (O(N^3))
    N_for = 1;% O(N)
    Sim_time = 60*10; % [sec] O(N)
    f_low = -1.0;
    f_up = 2.3;% log(250) = 2.3979.., 250Hz is the Nyquist frequency
    
    %f_low = 0.5; f_up = 1;% around the zero on imaginary axis

    
    
   %% Setting Simulink parameters   
   sim_filename = 'system_id_new';
    open_system(sim_filename)
    set_param(sim_filename,'StopTime','Sim_time+10')% Set the terminal time
    dt = 0.002;% sampling time [sec]    
    decim = round(Sim_time/dt);% To set the decimation for the "To Workspace" block
    % decim = 1; % default
    set_param([sim_filename,'/To Workspace'],'Decimation','decim')

%    G = zeros(1,2*N_d,2);
    
    % temtative parametesrs to build Simulink, updated in the for loop below
    gain_tuning = 20/N_d;
    frequency = rand(N_d,1);
    phase = rand(size(frequency));
    gain = gain_tuning*rand(size(frequency));
    
    qc_build_model;% Quarc command to build a model
    
    
    % Data accumulation 
for k=1:N_for
    freq_range = (f_up - f_low)*rand(N_d,1) + f_low;% from 10^f_low Hz to 10^f_up Hz. 
    % Note: Sim_time should be set more than 50/(10^f_low) in practice, morethan 10/(10^f_low) sec for simulation.   
    freq_range = sort(freq_range);% sorting
    frequency = 10.^[f_low:(f_up - f_low)/N_d:f_up-(f_up - f_low)/N_d]';% if specific frequencies are necessary
    phase = rand(size(frequency));
    gain = gain_tuning*rand(size(frequency));
    
    tic 
    qc_start_model;% Quarc 
    %set_param('system_id','SimulationCommand', 'start')% To start Simulink simulation
    toc
    pause(Sim_time + 2);
    set_param(sim_filename,'SimulationCommand', 'stop')
    % Obtain the frequency response
    toc
    pause(1)
    
end

