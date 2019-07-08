%% Code to FORCE Train an interneuron network to oscillate at a frequency theta_int, while receiving a secondary frequency of theta_ms 
clear all
clc 
%close all
rng(1) %fix RNG just for refersion interneurons down the road (see documentation), user can change this otherwise.  

%% Neuronal Parameters
NE = 2000;  %number of excitatory neurons 
NI = 2000; %number of inhibitory neuron 
mu = NE/(NE+NI); %ratio of E to total 
T = 40; %total simulation time (s) 
N = NE + NI;  
dt = 0.00005; %time step (s) 
nt = round(T/dt); 
tref = 0.002; %Refractory time constant in seconds 
tm = 0.01; %Membrane time constant 
vreset = -65; %Voltage reset 
vpeak = -40; %Voltage peak. 
td = 0.02; %decay time; 
tr = 0.002; %rise  time;

%% 
lambda = dt*0.05; %Sets the rate of weight change, too fast is unstable, too slow is bad as well.  
Pinv = eye(N)*lambda; %initialize the correlation weight matrix for RLMS
p = 0.1; %Set the network sparsity 


%% Coupling Weight Matrix
disp('Initializing Weight Matrix')
G = 0.1;  %Static Weight Magnitude 
W = 15; %Recurrent Weight Magnitude
AW = 10;  %Magnitude of Oscillatory Inputs  
p = 0.1; 
CE = round(p*NE); %Number of E connections 
CI = round(p*NI); %Number of I connectiosn 
tic
%create weight matrix 
tempE = zeros(1,NE); tempE(1,1:CE) = 1;
tempI = zeros(1,NI); tempI(1,1:CI) = -1;
 for i = 1:1:N 
OMEGA(i,:) = G*[(sqrt(CI)/sqrt(CE))*tempE(randperm(NE))/sqrt(CE),tempI(randperm(NI))/sqrt(CI)];
 end 
 toc
 
%% Kill EE and EI connections.  
OMEGA(NE+1:N,1:NE)=0;
OMEGA(1:NE,1:NE)=0;   

%% Input from medial septum
input = -(1+cos(2*pi*(1:1:nt)*dt*8)); %8hz input.  
% Input weight 
WIN(1:N,1) = AW;   
WIN(1:NE,1) = 0;  %Kill GABAergic inputs to Excitatory Neurons.  




%% Basis and supervisor 
nb = 100; %number of oscillators in the basis 
for k = 1:nb
zx(k,:) = (cos(2*pi*(1:1:nt)*dt*8.5 + 2*pi*rand));  %theta_int, 8.5 hz.  
end

%% FORCE parameters 
imin = round(1/dt); %start RLS 
icrit = round(21/dt); %stop RLS 
step = 10; %implement RLS every step interval



%% initialization parameters for the network 
k = min(size(zx)); %size of supervisor 
IPSC = zeros(N,1); %post synaptic current storage variable 
h = zeros(N,1); %Storage variable for filtered firing rates
r = zeros(N,1); %second storage variable for filtered rates 
hr = zeros(N,1); %Third variable for filtered rates 
JD = 0*IPSC; %storage variable required for each spike time 
tspike = zeros(4*nt,2); %Storage variable for spike times 
ns = 0; %Number of spikes, counts during simulation  
z = zeros(k,1);  %Initialize the approximant 
BPhi = zeros(N,k); %The initial matrix that will be learned by FORCE method
v = vreset + rand(N,1)*(vpeak-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v;  %v_ is the voltage at previous time steps  
RECB = zeros(nt,10);  %Storage matrix for the synaptic weights (a subset of them) 
kd = 0; 


%% Encoders 
E = zeros(N,k);
for j = 1:N
  in = ceil(k*rand);
  E(j,in) = W;
end

%% Storage matrices
nq = 20;
REC = zeros(nt,20);
REC2 = zeros(round(nt/nq),N);
current = zeros(nt,k);  %storage variable for output current/approximant 
i = 1; 


%% auxiliary parameters to implement FORCE training in a plausible way. 
z1 = z; z2 = z;
dec = zeros(N,2);
BPhi1 = 0*BPhi; BPhi2 = 0*BPhi;
EPlus = E;
EPlus(EPlus<0) = 0;
EMinus = E - EPlus;
mask1 = [ones(NE,k); -ones(NI, k)];
kd = 0; 
tlast = zeros(N,1); %This vector is used to set  the refractory times 
%Parameter used to compute the histogram/population activity online
bin = zeros(round(nt/round(0.001/dt)),1); 
binI = bin;
BIAS(1:NE,1)= -40;  %Background current to Excitatory neurons 
BIAS(NE+1:N,1)= 10; %Background current to Inhibitory neurons 

bs = 0; bsI = 0; 
ks = 0;






%%  START INTEGRATION 
for i = 1:1:nt 

z1 = BPhi1' * r;
z2 = BPhi2' * r;



I = IPSC + BIAS + EPlus*z1 + EMinus*z2 + WIN.*(input(i))*(dt*i<30) ; %Current to Neurons 
dv = (dt*i>tlast + tref).*( (-v+I)/tm)    ; %Voltage equation with refractory period 
v = v + dt*(dv); 

index = find(v>=vpeak);  %Find the neurons that have spiked 


%Store spike times, and get the weight matrix column sum of spikers 
if length(index)>0
JD = sum(OMEGA(:,index),2); %compute the increase in current due to spiking  
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i]; %store spike times 
ns = ns + length(index);  % total number of spikes so far
bs = bs + length(index(index<NE));
bsI = bsI + length(index(index>=NE));
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



%Implement RLS
 z = BPhi'*r; %approximant 
 err = z - zx(:,i); %error 
 %% RLMS 
 if mod(i,step)==1 
if i > imin 
 if i < icrit 
   cd = Pinv*r;
   BPhi = BPhi - (cd*err')/(1+(r')*cd);
   Pinv = Pinv -((cd)*(cd'))/( 1 + (r')*(cd));
BPhiEP = BPhi(1:NE,:).*(BPhi(1:NE,:)>0);
BPhiEM = BPhi(1:NE,:).*(BPhi(1:NE,:)<0);
BPhiIP = BPhi(NE+1:N,:).*(BPhi(NE+1:N,:)>0);
BPhiIM = BPhi(NE+1:N,:).*(BPhi(NE+1:N,:)<0);
BPhi1 = [BPhiEP;BPhiIM*mu/(1-mu)];
BPhi2 = [BPhiEM;BPhiIP*mu/(1-mu)];
end
end 
end

 

 

v = v + (30 - v).*(v>=vpeak); %rest the voltage and apply a cosmetic spike.  
REC(i,:) = [v(1:10);v(NE+1:NE+10)]; %Record a random voltage 
v = v + (vreset - v).*(v>=vpeak); %reset spike time 



current(i,:) = z; %store oscillators 
RECB(i,:) = BPhi(NE+1:NE+10); %store 10 inhibitory weights 

% Store filtered spike trains for 10 neurons.  
if mod(i,nq)==1
kd = kd + 1;
REC2(kd,:) = v; 
end

% compute histogram 
if mod(i,round(0.001/dt))==1    
    ks = ks + 1;
    bin(ks) = bs;
    binI(ks) = bsI;
    bs = 0; bsI = 0;
end



%% plotting results 
    if mod(i,round(0.5/dt))==1
   prog = dt*i/T
  drawnow
figure(100)
%plot voltage traces  
for j = 1:1:20
    if j > 10
 plot((1:1:i)*dt,REC(1:1:i,j)/(30-vreset)+j,'b'), hold on 
    else
         plot((1:1:i)*dt,REC(1:1:i,j)/(30-vreset)+j,'r'), hold on
    end

end
xlabel('Time (s)')
ylabel('Voltage (mv)')
ylim([0,21])
hold off


%plot supervisor and decoded network approximant
figure(200) 
for ffd = 1:3
plot(dt*(1:1:i),zx(ffd,1:1:i)/(max(zx(ffd,:))-min(zx(ffd,:)))+ffd,'k','LineWidth',2), hold on
plot(dt*(1:1:i),current(1:1:i,ffd)/(max(zx(ffd,:))-min(zx(ffd,:)))+ffd,'LineWidth',2)
end
xlim([dt*i-1,dt*i])
xlabel('Time (s)')
ylabel('Decoded Oscillators, \theta_{int}')
hold off 

%plot histograms 
figure(90) 
plot(dt*i*(1:ks)/ks,bin(1:ks),'r'), hold on 
plot(dt*i*(1:ks)/ks,binI(1:ks),'b'), hold off
xlabel('Time (s)')
ylim([0,100])
ylabel('Population Activity') 

%plot the decoders as they are being learned. 
figure(5) 
plot(dt*(1:1:i),RECB(1:1:i,1:10),'.')
xlabel('Time (s)')
ylabel('Decoders')
     end
    end

%% save the data, this file will be about 1-2gig's (depending on what network size, much larger for O(10^3)>neurons). 
save force_trained.mat -v7.3 

%% 
clear all
clc
sorting_script

