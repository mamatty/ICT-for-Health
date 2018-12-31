close all;
clear all;
clc;

Nstates = 8;
seed = 123456789;

%Healthy Dataset
train_health = zeros(2000,5);
test_health = zeros(2000,5);

dir = 'data/healthy/';
fix = 'a1.wav';
for i=0:1:9
    st = num2str(i);
    files=strcat(dir,'H00',st,fix);
    [voice, FS] = audioread(files);
    
    Nsub=5;
    Nmin=Nsub*500;
    N=Nsub*4500;
    FSa=FS/5;
    voice=voice(Nmin:Nsub:N-Nsub);
    
    if i == 0
        t=(0:length(voice)-1)/FSa;
        figure();
        grid on;
        plot(t,voice);
        title ('Original Signal');
        xlabel('Time');
    end
    
    Tmin = 1./200;
    [PKS,LOCS] = findpeaks(voice,FSa,'MinPeakDistance',Tmin);
    
    tt = zeros((length(LOCS)-1)*Nstates,1);
    ind = 0;
    for k = 1:(length(LOCS)-1)
        sam = (LOCS(k+1) - LOCS(k))/(Nstates);
        for j = 1:Nstates
            time_max = LOCS(k);
            ind = ind + 1;
            tt(ind)=time_max+(j*sam);
        end
    end
    
    voice1=interp1(t,voice,tt);
    voice1 = voice1(1:1:2000);
    
    Kquant = 8;
 
    amax = max(voice1);
    amin = min(voice1);
    delta = (amax-amin)/(Kquant-1); 
    ar = round((voice1-amin)/delta)+1;  
    
    if i == 0
        figure();
        plot(ar);
        title ('Signal Quantized');
        xlabel('Samples');
    end
   
    if(i < 5)
        train_health(:,i+1) = ar;
    else
        test_health(:,mod(i,5)+1) = ar;
    end
end

%Parkinson Dataset
train_park = zeros(2000,5);
test_park = zeros(2000,5);

dir = 'data/parkins/';
fix = 'a1.wav';
for i=0:1:9
    st = num2str(i);
    files=strcat(dir,'P00',st,fix);
    [voice, FS] = audioread(files);
    
    Nsub=5;
    Nmin=Nsub*500;
    N=Nsub*4500;
    FSa=FS/5;
    voice=voice(Nmin:Nsub:N-Nsub);
    
    t=(0:length(voice)-1)/FSa;
    %figure();
    %grid on;
    %plot(t,voice);
    %title ('Original Signal');
    %xlabel('Time');
    
    Tmin = 1./200;
    [PKS,LOCS] = findpeaks(voice,FSa,'MinPeakDistance',Tmin);
    
    tt = zeros((length(LOCS)-1)*Nstates,1);
    ind = 0;
    for k = 1:(length(LOCS)-1)
        sam = (LOCS(k+1) - LOCS(k))/(Nstates);
        for j = 1:Nstates
            time_max = LOCS(k);
            ind = ind + 1;
            tt(ind)=time_max+(j*sam);
        end
    end
    
    voice1=interp1(t,voice,tt);
    voice1 = voice1(1:1:2000);
    
    Kquant = 8;
 
    amax = max(voice1);
    amin = min(voice1);
    delta = (amax-amin)/(Kquant-1); 
    ar = round((voice1-amin)/delta)+1;  
    
    %figure();
    %plot(ar);
    %title ('Signal Quantized');
    %xlabel('Samples');
   
    if(i < 5)
        train_park(:,i+1) = ar;
    else
        test_park(:,i-4) = ar;
    end
end

rng(seed);

TRANS_HAT = zeros(Nstates,Nstates);

for i=1:Nstates
    if i==1
        TRANS_HAT(i,i+1)= 0.9-(rand(1)/8);
        var = 1 - TRANS_HAT(i,i+1);
        for j=[1,i+2:Nstates]
            TRANS_HAT(i,j)= var/(Nstates-1);         
        end
    elseif i==2
        TRANS_HAT(i,i+1)= 0.55-(rand(1)/8);  
        TRANS_HAT(i,i-1)= 0.4-(rand(1)/8);
        temp = TRANS_HAT(i,i+1) + TRANS_HAT(i,i-1);
        var = 1 - temp;
        for j=[i,i+2:Nstates]
             TRANS_HAT(i,j)= var/(Nstates-1);     
        end
    elseif i==Nstates
        TRANS_HAT(i,i-1)= 0.9-(rand(1)/8);
        var = 1 - TRANS_HAT(i,i-1);
        for j=[1:Nstates-2,i]
            TRANS_HAT(i,j)= var/(Nstates-1);         
        end
    else
        TRANS_HAT(i,i+1)= 0.55-(rand(1)/8);  
        TRANS_HAT(i,i-1)= 0.4-(rand(1)/8);
        temp = TRANS_HAT(i,i+1) + TRANS_HAT(i,i-1);
        var = 1 - temp;
        for j=[1:i-2,i,i+2:Nstates]
            TRANS_HAT(i,j)= var/(Nstates-1);
        end
    end    
end

EMIT_HAT = zeros(Nstates,Kquant);   

for i=1:1:Nstates
    var = rand(Kquant,1);
    EMIT_HAT(i,:) = var./sum(var);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BAUM-WELCH algorithm  

hit_tr_health_b = zeros(5,1);
hit_te_health_b = zeros(5,1);
hit_tr_park_b = zeros(5,1);
hit_te_park_b = zeros(5,1);

[ESTTR_health_BAUM,ESTEMIT_health_BAUM] = hmmtrain(train_health,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200);
[ESTTR_parkinson_BAUM,ESTEMIT_parkinson_BAUM] = hmmtrain(train_park,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200);

%Health part
for n = 1:1:5
  [~,LOGP] = hmmdecode(test_health(:,n).',ESTTR_health_BAUM,ESTEMIT_health_BAUM);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(test_health(:,n).',ESTTR_parkinson_BAUM,ESTEMIT_parkinson_BAUM);
  logp_p = LOGP;
  if(logp_h > logp_p)
    hit_te_health_b(n) = 1;
  else
    hit_te_health_b(n) = 0;
  end
  
  [~,LOGP] = hmmdecode(test_health(:,n).',ESTTR_health_BAUM,ESTEMIT_health_BAUM);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(train_health(:,n).',ESTTR_parkinson_BAUM,ESTEMIT_parkinson_BAUM);
  logp_p = LOGP;
  if(logp_h > logp_p)
    hit_tr_health_b(n) = 1;
  else
    hit_tr_health_b(n) = 0;
  end
  
end

%Parkinson part
for n = 1:1:5
  [~,LOGP] = hmmdecode(test_park(:,n).',ESTTR_health_BAUM,ESTEMIT_health_BAUM);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(test_park(:,n).',ESTTR_parkinson_BAUM,ESTEMIT_parkinson_BAUM);
  logp_p = LOGP;
  if(logp_p > logp_h)
    hit_te_park_b(n) = 1;
  else
    hit_te_park_b(n) = 0;
  end
  
  [~,LOGP] = hmmdecode(train_park(:,n).',ESTTR_health_BAUM,ESTEMIT_health_BAUM);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(train_park(:,n).',ESTTR_parkinson_BAUM,ESTEMIT_parkinson_BAUM);
  logp_p = LOGP;
  if(logp_p > logp_h)
    hit_tr_park_b(n) = 1;
  else
    hit_tr_park_b(n) = 0;
  end
  
end

disp('BAUM-WELCH probabilities:')

% Training sequences 
prob_train_health_b = sum(hit_tr_health_b)/length(hit_tr_health_b);
fprintf('Train Probability Health: %.2f\n', prob_train_health_b)

prob_train_park_b = sum(hit_tr_park_b)/length(hit_tr_park_b);
fprintf('Train Probability Parkinson: %.2f\n', prob_train_park_b)

% Test sequences 
prob_test_health_b = sum(hit_te_health_b)/length(hit_te_health_b);
fprintf('Test Probability Health: %.2f\n', prob_test_health_b)

prob_test_park_b = sum(hit_te_park_b)/length(hit_te_park_b);
fprintf('Test Probability Parkinson: %.2f\n', prob_test_park_b)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VITERBI algorithm 

hit_tr_health_v = zeros(5,1);
hit_te_health_v = zeros(5,1);
hit_tr_park_v = zeros(5,1);
hit_te_park_v = zeros(5,1);

[ESTTR_health_VIT,ESTEMIT_health_VIT] = hmmtrain(train_health,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200,'ALGORITHM','Viterbi');
[ESTTR_parkinson_VIT,ESTEMIT_parkinson_VIT] = hmmtrain(train_park,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200,'ALGORITHM','Viterbi');

%Health part
for n = 1:1:5
  [~,LOGP] = hmmdecode(test_health(:,n).',ESTTR_health_VIT,ESTEMIT_health_VIT);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(test_health(:,n).',ESTTR_parkinson_VIT,ESTEMIT_parkinson_VIT);
  logp_p = LOGP;
  if(logp_h > logp_p)
    hit_te_health_v(n) = 1;
  else
    hit_te_health_v(n) = 0;
  end
  
  [~,LOGP] = hmmdecode(train_health(:,n).',ESTTR_health_VIT,ESTEMIT_health_VIT);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(train_health(:,n).',ESTTR_parkinson_VIT,ESTEMIT_parkinson_VIT);
  logp_p = LOGP;
  if(logp_h > logp_p)
    hit_tr_health_v(n) = 1;
  else
    hit_tr_health_v(n) = 0;
  end
  
end

%Parkinson part
for n = 1:1:5
  [~,LOGP] = hmmdecode(test_park(:,n).',ESTTR_health_VIT,ESTEMIT_health_VIT);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(test_park(:,n).',ESTTR_parkinson_VIT,ESTEMIT_parkinson_VIT);
  logp_p = LOGP;
  if(logp_p > logp_h)
    hit_te_park_v(n) = 1;
  else
    hit_te_park_v(n) = 0;
  end
  
  [~,LOGP] = hmmdecode(train_park(:,n).',ESTTR_health_VIT,ESTEMIT_health_VIT);
  logp_h = LOGP;
  [~,LOGP] = hmmdecode(train_park(:,n).',ESTTR_parkinson_VIT,ESTEMIT_parkinson_VIT);
  logp_p = LOGP;
  if(logp_p > logp_h)
    hit_tr_park_v(n) = 1;
  else
    hit_tr_park_v(n) = 0;
  end
  
end

disp('VITERBI probabilities:')

% Training sequences 
prob_train_health_v = sum(hit_tr_health_v)/length(hit_tr_health_v);
fprintf('Train Probability Health: %.2f\n', prob_train_health_v)

prob_train_park_v = sum(hit_tr_park_v)/length(hit_tr_park_v);
fprintf('Train Probability Parkinson: %.2f\n', prob_train_park_v)

% Test sequences 
prob_test_health_v = sum(hit_te_health_v)/length(hit_te_health_v);
fprintf('Test Probability Health: %.2f\n', prob_test_health_v)

prob_test_park_v = sum(hit_te_park_v)/length(hit_te_park_v);
fprintf('Test Probability Parkinson: %.2f\n', prob_test_park_v)
