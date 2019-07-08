clear all
clc 

%uncomment this line if you train a new weight matrix 
load FORCE_trained.mat N NE NI tref tm vreset vpeak td tr BIAS OMEGA EPlus EMinus BPhi1 BPhi2 BPhi E W WIN

%% Neuronal Parameters
T = 10; %simulation time 
dt = 0.00005; %time step 
nt = round(T/dt);
%% Coupling Weight Matrix
load sortingid.mat 
%sort pyramidal and interneurons 
OMEGA0 = OMEGA + EPlus*BPhi1'+EMinus*BPhi2';
OMEGA0(NE+1:N,NE+1:N) = OMEGA0(ix3+NE,ix3+NE);
OMEGA0(1:NE,NE+1:N) = OMEGA0(ix2,NE+ix3);
BPhi(1:NE,:) = BPhi(ix2,:); %Decoders for FORCE oscilatrsos, also need sorting 
BPhi(NE+1:N,:) = BPhi(ix3+NE,:);
gie = 0.1; %e to i connection (initiators to the SHOT-CA3I's)
gee = 0.1; %e to e connection (initators to the rest of excitatory)
geeR = 1; %e to e connection (initiators only)
%%  
k = min(size(BPhi)); %number of FORCE components
input = -((1 + cos(2*pi*8*(1:nt)*dt))); %Septal inputs
%%
IPSC = zeros(N,1); %post synaptic current storage variable 
h = zeros(N,1); %Storage variable for filtered firing rates
r = zeros(N,1); %second storage variable for filtered rates 
hr = zeros(N,1); %Third variable for filtered rates 
ro = zeros(N,1); %storage variables for fast AMPA synapses 
hro = zeros(N,1);  
tdo = 0.005; %decay, AMPA synapse time constants for recurrent connectivity 
tro = 0.002; %rise
JD = 0*IPSC; %storage variable required for each spike time 
tspike = zeros(10*nt,2); %Storage variable for spike times 
ns = 0; %Number of spikes, counts during simulation  
z = zeros(k,1);  %Initialize the approximant 
v = vreset + rand(N,1)*(30-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v;  %v_ is the voltage at previous time steps  
u = zeros(N,1); %adaptation variable 
d = 11;  %adaptation jump amount  
tu = 0.05; %adaptation decay time constant 

current = zeros(nt,k);  %storage variable for output current/approximant 
z1 = z; z2 = z;
i = 1;

%background currents to neurons 
BIAS(1:NE) =  vpeak; 
BIAS(NE+1:N) = vpeak+1;

tcrit = 0; %keep septal inputs off 
XX = 2.0;
tlast = zeros(N,1); %This vector is used to set  the re*fractory times 
%%
ilast = i; 
for i = ilast:1:nt          
 
I = IPSC + BIAS + WIN.*input(i)*(dt*i<tcrit); %Neuronal Current
I(51:NE) = I(51:NE) + gee*mean(ro(1:50)); %initiators to the rest of SHOT-CA3E 
I(1:50) = I(1:50) + geeR*mean(ro(1:50)); %initiators to themselves 
I(NE+1:N) = I(NE+1:N) + gie*mean(ro(1:50)); %initiators to the SHOT-CA3I
I(1:NE) = I(1:NE) - u(1:NE);  %note that only pyramidal neurons have adaptation


dv = (dt*i>tlast + tref).*(-v+I)/tm; %Voltage equation with refractory period 
v = v + dt*(dv) + sqrt(dt/tm)*XX*randn(N,1);
index = find(v>=vpeak);  %Find the neurons that have spiked 
u = u + dt*(-u/tu) + d*(v>=vpeak);

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

%compute filtered spikes 
r = r*exp(-dt/tr) + hr*dt; 
hr = hr*exp(-dt/td) + (v>=vpeak)/(tr*td);
ro = ro*exp(-dt/tro) + hro*dt; 
hro = hro*exp(-dt/tdo) + (v>=vpeak)/(tro*tdo);
end



%Decoded FORCE componenets 
z = BPhi'*r;
v = v + (30 - v).*(v>=vpeak);
%REC(i,:) = v(1:20:N); %Record a random voltage 
v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  
current(i,:) = z;


     if mod(i,round(0.05/dt))==1
     dt*i/T
    figure(1)
    drawnow
    plot(tspike(tspike(1:ns,1)<=NE,2),tspike(tspike(1:ns,1)<=NE,1),'r.'), hold on 
    plot(tspike(tspike(1:ns,1)>NE,2),tspike(tspike(1:ns,1)>NE,1),'b.'), hold off
    xlim([dt*i-2,dt*i])
    xlabel('Time (s)')
    ylabel('Spike Rastor') 
    legend('SHOT-CA3E','SHOT-CA3I') 

     end
end
%%
 