clear all
close all
clc 

%load trained data 
load force_trained.mat N NE NI tref tm vreset vpeak td tr BIAS OMEGA EPlus EMinus BPhi1 BPhi2 BPhi E W WIN BIAS
% Neuronal Parameters
T = 5; %total simulation time 
tcrit1 = 4.03; %MS Off, you might need to change this for different realizations of the simulation to minimize phase distortion error.  
tcrit2 = 4.13; %MS ON again 
dt = 0.00005; %integration time step 
nt = round(T/dt); %total number of time steps 
load sortingid.mat %load sorting index for the neurons 

%% generate global variables required for all simulations 
NTOT = N + N; %total number of neurons \
NER = 2000; %number of RO-CA1E neurons 
BIAS(1:NE) = -5;  %SHOT-CA3E 
BIAS(NE+1:N) = 10;  %SHOT-CA3I 
BIAS(N+1:N+NER) = -42.5; %RO-CA1E 
BIAS(N+1+NER:NTOT) = -40;%RO-CA1I






epsilon = 7.875e-7;
mag0 = 20; %pulse of 
time = (1:nt)*dt;
digcur = zeros(nt,NER);
Z = 2;
t1 = 1; t2 = t1+2;
for i = 1:NER 
digcur(:,i) = (1+0*time).*(time<t1 + 2*(i-1)/(Z*NER)+0.04).*(time>t1+2*(i-1)/(Z*NER));
end
pcur = digcur;


G2 = 25; %12
G3 = - 0.03;
tic
 
PHI = zeros(NER,NE);
OMEGA0 = zeros(NTOT,NTOT); 
OMEGAS = OMEGA + EPlus*BPhi1'+EMinus*BPhi2';
OMEGAS(1:NE,:)=OMEGAS(ix2,:);
OMEGA0(1:N,1:N) = OMEGAS; 

G1 = 0; %G1 = 10;



OMEGA0(N+1:N+NER,N+1+NER:NTOT) = G3*rand(NER,NI)/NI;  %connections from RO-CA1I to RO-CA1E   
OMEGA2 = 0*OMEGA0;  %runs on a faster time
OMEGA2(N+1+NER:NTOT,N+1:N+NER) = G2*(rand(NI,NER))/(NER);  %connections from RO-CA1E to RO-CA1I


k = min(size(BPhi));
tstar = (1:nt)*dt;
input = -(1+cos(2*pi*8*(1:nt)*dt)); %input current 
%% 


%% Initialize Network 
IPSC = zeros(NTOT,1); %post synaptic current storage variable (slow synapses)
IPSC2 = zeros(NTOT,1); %postsynaptic current storagevsariable (fast synapses) 
h = zeros(NTOT,1); %Storage variable for filtered firing rates (slow) 
r = zeros(NTOT,1); %second storage variable for filtered rates (slow)) 
hr = zeros(NTOT,1); %Third variable for filtered rates 
h2 = zeros(NTOT,1); %storage variavle for filtered firing rates (Fast)
r2 = zeros(NTOT,1); %postsynaptic currnet for filtered firing rates (Fast)
hr2 = zeros(NTOT,1); %storage variavle for filtered firing rates (Fast)

td2 = 0.005; %AMPA time scale 
tr2 = 0.002; %AMPA time scale 

JD = 0*IPSC; %storage variable required for each spike time.
JD2 = 0*IPSC2; %storage variable required for each spike time.  
tspike = zeros(50*nt,2); %Storage variable for spike times. 
ns = 0; %Number of spikes, counts during simulation.
v = vreset + rand(NTOT,1)*(vpeak-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v;  %v_ is the voltage at previous time steps  
tlast = zeros(NTOT,1); %This vector is used to set  the refractory times 


sigx = zeros(NTOT,1); %standard deviation of noise
sigx(N+NER+1:NTOT) = 0.2; %only RO-CA1I gets noisy inputs. 


mq = 15; %only integrate Hebbian plasticity rule every 20 time steps (used to speed up simulations, can be dropped lower) 
times = (1:nt)*dt;% 
sigT = (0.05).^2; 

%apply extra current to SHOT-CA3E neurons
finput = input.*(1 - (times>tcrit1).*(times<tcrit2)); 
einput = (times>tcrit1).*(times<tcrit2);
AG = zeros(NER,1); %initialize current to RO-CA1E neurons from SHOT-CA3E 
AF = 22; %extra current to initialize SPW. 
tic

%% 
for i = 1:nt 

%currents 
I = BIAS;
I(1:NE) = I(1:NE) + AF*einput(i);
I(1:NTOT,1) = I(1:NTOT,1) + IPSC + IPSC2;
I(NE+1:N) = I(NE+1:N) + (WIN(NE+1:N))*(finput(i)); 
I(N+1:N+NER,1) = I(N+1:N+NER,1) + mag0*digcur(i,:)';
if dt*i>t1 
I(N+1:N+NER) = I(N+1:N+NER) + AG;  
end


%integrate voltage 
dv = (dt*i>tlast + tref).*(-v+I + sigx.*randn(NTOT,1).*sqrt(tm/dt))./tm; %Voltage equation with refractory period 
v = v + dt*(dv);
index = find(v>=vpeak);  %Find the neurons that have spiked 


%Store spike times, and get the weight matrix column sum of spikers 
if length(index)>0
JD = sum(OMEGA0(:,index(index<=NTOT)),2); %compute the increase in current due to spiking (fast synapse)
JD2 = sum(OMEGA2(:,index(index<=NTOT)),2); %compute the increase in current due to spiking  (slow synapse) 
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

%faster synapse 
if tr2 == 0 
IPSC2 = IPSC2*exp(-dt/td2)+   JD2*(length(index)>0)/(td2);
r2 = r2*exp(-dt/td2) + (v>=vpeak)/(td2);
else 
IPSC2 = IPSC2*exp(-dt/tr2) + h2*dt;
h2 = h2*exp(-dt/td2) + JD2*(length(index)>0)/(tr2*td2);  %Integrate the current
    
r2 = r2*exp(-dt/tr2) + hr2*dt; 
hr2 = hr2*exp(-dt/td2) + (v>=vpeak)/(tr2*td2);
   
end


%reset voltages 
v = v + (30 - v).*(v>=vpeak);
v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  

%Apply Fourier Rule/Hebbian plasticity.  As supervisor is shorter then
%period of plasiticty, forgetting term is unnecessary and not implemented.
%
 if dt*i>t1
    if dt*i<t2
        if mod(i,mq)==0
      PHI = PHI + mq*epsilon*dt*(r(N+1:N+NER))*(r(1:NE)');
    end
    end
 end 
 
 
    AG = PHI*r(1:NE);
    if mod(i,round(0.1/dt))==1
      dt*i/T  
    end
end
   
%% Plotting 
mx = 3;
my = 3;
subplot(mx,my,1:2)
set(gca,'FontSize',24)
CA1 = tspike(tspike(:,1)>N,2);
CA3 = tspike(tspike(:,1)<N,2);
[hCA1,xCA1] = hist(CA1,0:0.0001:T);
[hCA3,xCA3] = hist(CA3,0:0.0001:T);
plot(xCA3,hCA3), hold on 
plot(xCA3,smooth(hCA3,10),'k','LineWidth',2)
ylim([0,50])
xlim([0,5])
title('SHOT-CA3 Multi-Unit Activity')
ylabel('Spike/ms')
xlabel('Time (s)')
patch([4,4.3,4.3,4],[0,0,50,50],'r','facealpha',0.2,'edgecolor','none')
set(gca,'FontSize',14)
subplot(mx,my,3)
plot(xCA3,hCA3), hold on 
plot(xCA3,smooth(hCA3,10),'k','LineWidth',2)
xlim([4,4.3])



set(gca,'FontSize',14)

set(gca,'FontSize',14)
ylim([0,5*10^1])
subplot(mx,my,4:5)
ylabel('Spike/ms')
xlabel('Time (s)')



plot(xCA1,hCA1)
xlim([0,5])
set(gca,'FontSize',14)
plot(xCA1,hCA1), hold on 
plot(xCA1,smooth(hCA1,10),'k','LineWidth',2)
title('RO-CA1 Multi-Unit Activity')
set(gca,'FontSize',14)
ylim([0,200])
patch([4,4.3,4.3,4],[0,0,200,200],'r','facealpha',0.2,'edgecolor','none')
xlim([0,5])
subplot(mx,my,6)
plot(xCA1,hCA1), hold on 
plot(xCA1,smooth(hCA1,10),'k','LineWidth',2)
xlim([4,4.3])
set(gca,'FontSize',14)
ylim([0,200])
subplot(mx,my,7:8)
plot(tspike(:,2),tspike(:,1),'k.')
ylim([N+1,N+NER]);
ylabel('Neuron Index')
xlabel('Time (s)')

patch([4,4.3,4.3,4],[4000,4000,6000,6000],'r','facealpha',0.2,'edgecolor','none')

title('RO-CA1 Raster Plot')
set(gca,'FontSize',14)


xlim([0,5])

subplot(mx,my,9)
plot(tspike(:,2),tspike(:,1),'k.')
ylim([N+1,N+NER]);
xlim([4,4.3])
set(gca,'FontSize',14)
