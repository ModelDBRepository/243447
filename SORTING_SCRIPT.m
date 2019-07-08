clear all
close all
clc 
%script to phase sort the neurons after FORCE training.  simulates the
%trained network for a short period of time  (5 s) with no septal inputs.  

load('FORCE_trained.mat','N','NE','NI','tref','tm','vreset','vpeak','td','tr','BIAS','OMEGA','EPlus','EMinus','BPhi1','BPhi2','BPhi','E','W','WIN')
%% Neuronal Parameters
T = 5
dt = 0.00005;
nt = round(T/dt);
%% Coupling Weight Matrix 
 OMEGA0 = OMEGA + EPlus*BPhi1'+EMinus*BPhi2'; %static plus learned FORCE componenets 
%% 
k = min(size(BPhi)); %number of decoded oscillators
input = -((1 + cos(2*pi*8*(1:nt)*dt))); %MS inputs
%%
IPSC = zeros(N,1); %post synaptic current storage variable 
h = zeros(N,1); %Storage variable for filtered firing rates
r = zeros(N,1); %second storage variable for filtered rates 
hr = zeros(N,1); %Third variable for filtered rates 

JD = 0*IPSC; %storage variable required for each spike time 
tspike = zeros(10*nt,2); %Storage variable for spike times 
ns = 0; %Number of spikes, counts during simulation  
z = zeros(k,1);  %Initialize the approximant 
v = vreset + rand(N,1)*(30-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v;  %v_ is the voltage at previous time steps  
mq = 10; 
REC2 = zeros(round(1.1*nt/(mq)),N);
REC = zeros(nt,10);

current = zeros(nt,k);  %storage variable for output current/approximant 
current2 = zeros(nt,2*k);
z1 = z; z2 = z;
i = 1;
kd = 0;


BIAS(1:NE) = -10;
tcrit = 0; %tcrit determines when to turn of the MS, keep the MS off with tcrit = 0; 
tlast = zeros(N,1); %This vector is used to set  the refractory times 
%%
ilast = i; 

for i = ilast:1:nt          
 
I = IPSC + BIAS + WIN.*input(i)*(dt*i<tcrit); %Neuronal Current
I(1:NE) = I(1:NE) + 20*(dt*i>tcrit);
dv = (dt*i>tlast + tref).*(-v+I)./tm; %Voltage equation with refractory period 
v = v + dt*(dv);
index = find(v>=vpeak);  %Find the neurons that have spiked 


%Store spike times, and get the weight matrix column sum of spikers 
if length(index)>0
JD = sum(OMEGA0(:,index),2); %compute the increase in current due to spiking  
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i];
ns = ns + length(index);  % total number of psikes so far
end

tlast = tlast + (dt*i -tlast).*(v>=vpeak);  %Used to set the refractory period of LIF neurons 

% Code if the rise time is 0, and if the rise time is positive 
if tr == 0  
    IPSC = IPSC*exp(-dt/td)+   JD*(length(index)>0)/(td);
    r = r *exp(-dt/td) + (v>=vpeak)/td;
else
    IPSC = IPSC*exp(-dt/tr) + h*dt;
h = h*exp(-dt/td) + JD*(length(index)>0)/(tr*td);  %Integrate the current
r = r*exp(-dt/tr) + hr*dt; 
hr = hr*exp(-dt/td) + (v>=vpeak)/(tr*td);

end


z = BPhi'*r;
v = v + (30 - v).*(v>=vpeak);
REC(i,:) = [v(1:5);v(NE+1:NE+5)]; %Record a random voltage 
v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  
current(i,:) = z;

if mod(i,mq)==1 
kd = kd + 1;
REC2(kd,:) = I; 
end


if mod(i,round(0.05/dt))==1
dt*i/T
end
end
%% 
%% sort neurons according to phase preferences.  
tp = 1/8.5; %period of theta_int oscillator 
dt2 = T/kd;  %shorter integration time constant because of storage variables
dq = round(tp/dt2);  %number of time steps in one period 
VEC = REC2(kd-dq+1:kd,1:NE); %the currents for one period 
[mx,ix] = max(VEC); %location of the peak 
[ix,ix2] = sort(ix); %sort where the peak location is for SHOT-CA3E neurons 
VEC = REC2(kd-dq+1:kd,NE+1:N);   %the currents for one period 
[mx,ix] = max(VEC); %location of the peak 
[ix,ix3] = sort(ix); %sort where the peak location is for SHOT-CA3I neurons


save('sortingid.mat','ix2','ix3')

%% 
figure(2) 
OMEGA0 = OMEGA + EPlus*BPhi1'+EMinus*BPhi2';
OMEGA0(NE+1:N,NE+1:N) = OMEGA0(ix3+NE,ix3+NE);
OMEGA0(1:NE,NE+1:N) = OMEGA0(ix2,NE+ix3);
subplot(1,2,1)
imagesc(OMEGA0(NE+1:N,NE+1:N),[-0.03,0])
xlabel('pre')
ylabel('post')
colorbar
set(gca,'ydir','normal')
title('Phase Sorted I to I') 
subplot(1,2,2)
imagesc(OMEGA0(1:NE,NE+1:N),[-0.03,0])
title('Phase Sorted I to E')
xlabel('pre')
ylabel('post')
colorbar
set(gca,'ydir','normal')