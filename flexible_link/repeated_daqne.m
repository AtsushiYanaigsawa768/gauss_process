
%close all


%% Changable parameters
N_d = 100;% # of sampling frequencies (O(N^3))
N_for = 12;% O(N)
Sim_time = 10*3600; % [sec] O(N)
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
        
    freq_range = (f_up - f_low)*rand(N_d,1) +f_low ;% from 10^f_low Hz to 10^f_up Hz. 
    % Note: Sim_time should be set more than 50/(10^f_low) in practice, morethan 10/(10^f_low) sec for simulation.   
    freq_range = sort(freq_range);% sorting
    frequency = 10.^freq_range;
    frequency = 10.^[f_low:(f_up - f_low)/N_d:f_up-(f_up - f_low)/N_d]';% if specific frequencies are necessary
    phase = rand(size(frequency));
    gain = gain_tuning*rand(size(frequency));
    
    tic 
    qc_start_model;% Quarc 
%    set_param('system_id','SimulationCommand', 'start')% To start Simulink simulation
    toc
    pause(Sim_time + 2);
    set_param(sim_filename,'SimulationCommand', 'stop')
    % Obtain the frequency responce
    toc
    pause(1)
    
    
    size_G = size(G);
    omega_temp = 2*pi*frequency';
    
    if length(size(G))==3
        term = size_G(3);
        
        SysGain_temp = sqrt(sum([G(1,1:N_d,term).^2; G(1,N_d+1:2*N_d,term).^2]));
        SysPhase_temp = atan2(-G(1,N_d+1:2*N_d,term),-G(1,1:N_d,term));% arctan(cos / sin)
    else
        SysGain_temp = sqrt(sum([G(1,1:N_d).^2; G(1,N_d+1:2*N_d).^2]));
        SysPhase_temp = atan2(-G(1,N_d+1:2*N_d),-G(1,1:N_d));% arctan(cos / sin)
    end
    
    
    
    if k==1
        omega = omega_temp;
        SysGain = SysGain_temp;
        SysPhase = SysPhase_temp;
    else
        omega = [omega omega_temp];
        SysGain = [SysGain SysGain_temp];
        SysPhase = [SysPhase SysPhase_temp];
        
    end
    
    clear SysGain_temp SysPhase_temp omega_temp
    
end

% sorting
P = sortrows([omega; SysGain; SysPhase]')';
omega = P(1,:);
SysGain = P(2,:);
SysPhase = P(3,:);

%% save
% Set file name
clockdata = fix(clock);
clocky=num2str(clockdata(1));
    if clockdata(2)<10
        clockm=['0', num2str(clockdata(2))];
    else
        clockm=num2str(clockdata(2));    
    end    
    if clockdata(3)<10        
        clockd=['0', num2str(clockdata(3))];        
    else        
        clockd=num2str(clockdata(3));        
    end    
    if clockdata(4)<10        
        clockh=['0', num2str(clockdata(4))];        
    else        
        clockh=num2str(clockdata(4));        
    end
    if clockdata(5)<10
        clockmin=['0', num2str(clockdata(5))];        
    else
        clockmin=num2str(clockdata(5));
    end

filename = ['SKE2024_data',date,'_',clockh,clockmin,'.mat'];
save(filename,'P')
filename = ['SKE2024_data',date,'_',clockh,clockmin,'.dat'];
csvwrite(filename,P)

%% Plot figures
figure
semilogx(omega,20*log10(SysGain),'b*','linewidth',2)
xlabel('$$\omega$$ [rad/sec]','Interpreter', 'Latex','fontsize',16)
ylabel('$$20\log |G(j \omega)|$$','Interpreter', 'Latex','fontsize',16)
title('Bode Gain plot','fontsize',16)
grid on

figure
plot(real(SysGain.*exp(1i*SysPhase)),imag(SysGain.*exp(1i*SysPhase)),'b*','linewidth',3)% Nyquist diagram
xlabel('$$\mathrm{Re}(G(j \omega))$$','Interpreter', 'Latex','fontsize',16)
ylabel('$$\mathrm{Im}(G(j \omega))$$','Interpreter', 'Latex','fontsize',16)
title('Nyquist Plot','fontsize',16)
grid on
