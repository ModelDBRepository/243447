clear all
close all
clc 

%load previously implemented files.  If you train a new set of weights,
%you'll need to uncomment the next line.  
%load FORCE_trained.mat N NE NI tref tm vreset vpeak td tr BIAS OMEGA EPlus EMinus BPhi1 BPhi2 BPhi E W WIN zx 
load FORCE_trained_weights.mat 
%% Neuronal Parameters
T = 5;  %total simulation time 
dt = 0.00005; %integration time step 
nt = round(T/dt); %total number of time steps 
NT = N + NI;  %total number of neurons (SHOT + REV) 

%% load SHOT-CA3E and SHOT-CA3I sorting id's 
%load sortingid.mat ix2 ix3 

%% Coupling Weight Matrix 
%sort the SHOT-CA3 neurons and the decoders by phase 
OMEGA0 = OMEGA + EPlus*BPhi1'+EMinus*BPhi2';
OMEGA0(NE+1:N,NE+1:N) = OMEGA0(ix3+NE,ix3+NE);
OMEGA0(1:NE,NE+1:N) = OMEGA0(ix2,NE+ix3);
BPhi(1:NE,:) = BPhi(ix2,:);
BPhi(NE+1:N,:) = BPhi(ix3+NE,:);


%load inhibitory_reversion_neuron_phi.mat phi; %load files from the interneuron reversion training 
PHI = phi; %used to implement the interneuron reversion 

%Bias currents
BIAS(1:NE) = 15; %SHOT-CA3E 
BIAS(NE+1:N) = 10; %SHOT-CA3I 
BIAS(N+1:NT) = -40;  %REV 

%set connections to/form the reversion population, the last NI neurons 

OMEGA0 = [OMEGA0,zeros(N,NI);zeros(NI,N),zeros(NI,NI)];  %full weight matrix    
GI = -1;   %connection magnitude from SHOT-CA3I to REV 
OMEGA0(N+1:NT,NE+1:N) = eye(NI)*GI; %OMEGA(NE+1:N,NE+1:N); %connection from SHOT-CA3I to REV 
%% 
k = min(size(BPhi)); %number of FORCE componenets 
input = -((1 + cos(2*pi*8*(1:nt)*dt))); %septal inputs 


%%
IPSC = zeros(NT,1); %post synaptic current storage variable 
h = zeros(NT,1); %Storage variable for filtered firing rates
r = zeros(NT,1); %second storage variable for filtered rates 
h2 = h; 
r2 = r;
hr = zeros(NT,1); %Third variable for filtered rates 

JD = 0*IPSC; %storage variable required for each spike time 
tspike = zeros(10*nt,2); %Storage variable for spike times 
ns = 0; %Number of spikes, counts during simulation  
z = zeros(k,1);  %Initialize the approximant 
v = vreset + rand(NT,1)*(30-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v;  %v_ is the voltage at previous time steps  
mq = 10; %used to avoid storing every time step, reduces RAM required for the simulation. 
REC2 = zeros(round(1.1*nt/(mq)),NT); %store filtered spike times 
REC = zeros(nt,10); %store voltage traces 

i = 1;
kd = 0;%varaible for storage 
tcrit = 0.8*T; %turn septal input off  at this time 
phi = 2*pi*(1:NE)'/NE;  %phases of the excitatory neurons inhibition from the SHOT-CA3I, note that this is an approximation assuming near uniform phases, can be improved upon by directly estimating phase 
qin = 4; %If the user has altered the random number seeds in previous .m files, they may need to alter qin to satify the phase relationship in the Materials and Methods 


tlast = zeros(NT,1); %This vector is used to set  the re*fractory times 
%%
ilast = i; 

for i = ilast:1:nt          
    
    
  %Euler integration for the neurons   
I = IPSC + BIAS;%+ WIN.*input(i)*(dt*i<tcrit); %Neuronal Current
I(NE+1:N) = I(NE+1:N) + WIN(NE+1:N).*(input(i))*(dt*i<tcrit);
I(1:NE) = I(1:NE) + 20*(dt*i>tcrit); %trigger compression 
I(N+1:NT) = I(N+1:NT) + 30*(dt*i>tcrit); %trigger reversion 
zq = PHI'*r(N+1:NT);
I(1:NE) = I(1:NE) - 20*(1-sin(phi)*zq(qin));  %current to excitatory neurons from reversion population 
dv = (dt*i>tlast + tref).*(-v+I)/tm; %Voltage equation with refractory period 
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

%store voltages 
v = v + (30 - v).*(v>=vpeak);
REC(i,:) = [v(1:5);v(NE+1:NE+5)]; %Record a random voltage 
v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  

%store filtered spike trains
if mod(i,mq)==1 
kd = kd + 1;
REC2(kd,:) = r;
end


     if mod(i,round(0.05/dt))==1
     dt*i/T

     end
end
time = 1:1:nt; 

%% plotting stuff
tspike = tspike(1:ns,:);
revspike = tspike(tspike(:,1)>N,:);
ispike = tspike((tspike(:,1)<=N).*(tspike(:,1)>NE)==1,:);
espike = tspike(tspike(:,1)<=NE,:); 
plot(espike(:,2),espike(:,1),'r.'), hold on 
plot(ispike(:,2),ispike(:,1),'b.')
plot(revspike(:,2),revspike(:,1),'.','color',  [0.9100 0.4100 0.1700])
legend('SHOT-CA3E','SHOT-CA3I','Reversion Interneurons')
xlabel('Time (s)')
ylabel('Neuron Indices')
title('Spike Raster Plot')

%% 
