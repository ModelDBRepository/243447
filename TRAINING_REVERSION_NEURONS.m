clear all
clc 

load FORCE_TRAINED.mat N NE NI tref tm vreset vpeak td tr BIAS OMEGA EPlus EMinus BPhi1 BPhi2 BPhi E W WIN zx
load sortingid.mat ix2 ix3 
%% Simulation parameters 
T = 50; %total training time
dt = 0.00005;
nt = round(T/dt);
NT = N + NI; %Last NI neurons are the reversion interneurons


%% Coupling Weight Matrix 
%sort the SHOT-CA3 neurons and the decoders by phase 
OMEGA0 = OMEGA + EPlus*BPhi1'+EMinus*BPhi2';
OMEGA0(NE+1:N,NE+1:N) = OMEGA0(ix3+NE,ix3+NE);
OMEGA0(1:NE,NE+1:N) = OMEGA0(ix2,NE+ix3);
BPhi(1:NE,:) = BPhi(ix2,:);
BPhi(NE+1:N,:) = BPhi(ix3+NE,:);



%Bias currents
BIAS(NE+1:N) = 10; %SHOT-CA3I 
BIAS(N+1:NT) = -20;  %REV 

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
h2 = h; r2 = r;
hr = zeros(NT,1); %Third variable for filtered rates 

JD = 0*IPSC; %storage variable required for each spike time 
tspike = zeros(10*nt,2); %Storage variable for spike times 
ns = 0; %Number of spikes, counts during simulation  
z = zeros(k,1);  %Initialize the approximant 
v = vreset + rand(NT,1)*(30-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v;  %v_ is the voltage at previous time steps  
mq = 10; %used to avoid storing every time step, reduces RAM required for the simulation. 
REC2 = zeros(round(1.1*nt/(mq)),NI); %store filtered spike times, only for the REV interneurons
REC = zeros(nt,10); %store voltage traces 


current = zeros(nt,k);  %storage variable for output current/approximant 
z1 = z; z2 = z;
i = 1;
kd = 0;

%ceil(NE*rand(500,1)); 
tcrit = 0.9*T; %stop septal inputs at this time. 
tlast = zeros(NT,1); %This vector is used to set  the re*fractory times 
%%
ilast = i; 

for i = ilast:1:nt          
I = IPSC + BIAS; %Neuronal Current
I(NE+1:N) = I(NE+1:N) + WIN(NE+1:N).*(input(i))*(dt*i<tcrit);


z = BPhi'*r(1:N);
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



v = v + (30 - v).*(v>=vpeak);
REC(i,:) = [v(1:5);v(NE+1:NE+5)]; %Record a random voltage 
v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  


if mod(i,mq)==1 
kd = kd + 1;
REC2(kd,:) = r(N+1:NT);
end
current(i,:) = z;

     if mod(i,round(0.05/dt))==1
     dt*i/T
   drawnow

     end
end
time = 1:1:nt; 

%% 
plot(tspike(:,2),tspike(:,1),'k.')
%% Train the reversion interneurons to decoded out all FORCE oscillator componenets.  One componenet will subsequently be used for 
% the reversion current.  

target = current(1:mq:0.5*kd*mq,:);  %train the reversion neurons to output all FORCE oscillator components through an offline decoder 
basis = REC2(1:0.5*kd,:); %basis, only use half the simulation time 
basislong = REC2(1:kd,:); %full basis, for testing. 

lambda = 50;  %regularization amount
phi = pinv(basis'*basis + eye(NE)*lambda)*(basis'*target);
%% 
xhat = basislong*phi;  %decoded oscillator component 
%% plot SHOT-CA3I decoded vs. REV decoded 
plot((1:kd)*T/kd,xhat(:,1)), hold on  
plot((1:nt)*dt,current(:,1))
xlim([48,50])
xlabel('Time (s)')
legend('REV decoded','SHOT-CA3I decoded')
%% 
save trainingreversioninterneuronsworkspace.mat -v7.3 
save inhibitory_reversion_neuron_phi.mat phi 