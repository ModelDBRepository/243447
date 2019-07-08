clear all
clc 

%load trained weights, and phase sort indices
load FORCE_trained.mat N NE NI tref tm vreset vpeak td tr BIAS OMEGA EPlus EMinus BPhi1 BPhi2 BPhi E W WIN
load sortingid.mat
%% Simulation Parameters
T = 20;
dt = 0.00005;
nt = round(T/dt);
%% Coupling Weight Matrix, phase sort 
OMEGA0 = OMEGA + EPlus*BPhi1'+EMinus*BPhi2';
OMEGA0(NE+1:N,NE+1:N) = OMEGA0(ix3+NE,ix3+NE);
OMEGA0(1:NE,NE+1:N) = OMEGA0(ix2,NE+ix3);

%% Neuronal and input parameters 
k = min(size(BPhi)); %number of FORCE componenets 
input = -(1 + cos(2*pi*8*(1:nt)*dt));  %septal inputs, INP-MS
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
mq = 10; %used to storage every mq time steps, helps cut down on ram use.  
REC2 = zeros(round(1.1*nt/(mq)),N); %store filtered spikes.  
REC = zeros(nt,10); %store voltage traces

current = zeros(nt,k);  %storage variable for output current/approximant 
i = 1; %initial simulation time point 
kd = 0; %initial storage variable for REC2 


BIAS(1:NE) = -5; %Set bias current to exictatory neurons 
tcrit = 15; %at this time, turn off the septal inputs.   
tlast = zeros(N,1); %This vector is used to set  the re*fractory times 
%%
ilast = i; 

for i = ilast:1:nt          
 
I = IPSC + BIAS + WIN.*input(i)*(dt*i<tcrit); %Neuronal Current
I(1:NE) = I(1:NE) + 20*(dt*i>tcrit); %extra current if MS is off
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



z = BPhi'*r;
v = v + (30 - v).*(v>=vpeak);
REC(i,:) = [v(1:5);v(NE+1:NE+5)]; %Record a random voltage 
v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  
current(i,:) = z; %store FORCE componenets 

if mod(i,mq)==1 
kd = kd + 1;
REC2(kd,:) = r;  %store the filtered spikes 
end


     if mod(i,round(0.05/dt))==1
     dt*i/T
     end
end
%%
time = dt*(1:1:nt);  
timeS = (1:kd)*T/kd; 
REC2 = REC2(1:kd,:);
%%  Plotting Script
close all
figure('DefaultAxesFontSize',24)
cl = 4; 
rw = 3;

cmaps = ones(100,3);
cmaps(:,1) = (99:-1:0)/99; 
cmaps(:,2) = cmaps(:,1);
cmaps(:,3) = cmaps(:,1);
REC3 = REC2(:,1:NE);


figure(1)
h1 = subplot(rw,cl,[1,2,5,6]);
k = 0; 
for i = [3:5,8:10]
    k = k + 1;
    if i <= 5
plot((1:nt)*dt,REC(:,i)/(50-vreset)+k,'r','LineWidth',1), hold on  
    else
plot((1:nt)*dt,REC(:,i)/(50-vreset)+k,'b','LineWidth',1), hold on
    end
end
xlim([10,15.5])
ylim([0,6.5])
title('Neural Voltages')
set(gca,'YTick',[])
ylabel('Neuron Index')
pos = get(h1,'position');
pos(4) = pos(4)*0.95;
set(h1,'position',pos)
set(gca,'XTick',[])


h2 = subplot(rw,cl,[3,4,7,8]);

tspike = tspike(1:ns,:); 
[HE,XE] = hist(tspike(tspike(:,1)<=NE,2),0:0.001:T);
[HI,XI] = hist(tspike(tspike(:,1)>NE,2),0:0.001:T);
[H,X] = hist(tspike(1:ns,2),0:0.001:T);
plot(XE,HE/max(HE(XE>1))+1,'r'), hold on 
plot(XI,HI/max(HI(XI>1))+1.5,'b')

plot(X,H/max(H(X>1))+1.8,'Color',[0.5,0.5,0.5],'LineWidth',1)

plot(X,smooth(H,20)/max(H(X>1))+1.8,'k','LineWidth',2)
title('Population Activity')
legend('E','I','Total','Average','Theta Input')
h4 = plot(XE,smooth(HE,20)/max(HE(XE>1))+1,'k','LineWidth',2,'HandleVisibility','off'), hold on 
h5 = plot(XI,smooth(HI,20)/max(HI(XI>1))+1.5,'k','LineWidth',2,'HandleVisibility','off')

ylim([0.9,2.2])
xlim([10,15.5])

dilspike=tspike(tspike(:,2)<T/2,:);
comspike = tspike(tspike(:,2)>=T/2,:);
 

%ylabel('Population Activities')
set(gca,'YTick',[])
pos = get(h2,'position');
pos(4) = pos(4)*0.95;
%pos(2) = pos(2) + 0.05
set(h2,'position',pos)
set(gca,'XTick',[])


figure(1)
subplot(rw,cl,9:10)
time = (1:kd)*T/kd;
Ner = (1:NE);
imagesc(time,Ner,REC3(1:kd,1:NE)',[0,5]), hold on 
set(gca,'YDir','normal')
plot([15,15],[0,NE],'b','LineWidth',2)
plot([15.5,15.5],[0,NE],'b','LineWidth',2)
plot([15,15.5],[NE,NE],'b','LineWidth',2)
plot([15,15.5],[0,0],'b','LineWidth',2)
set(gca,'XTick',[])
xlim([10,15.5])
title('Time Fields')
ylabel('Neuron Index (E)')
xlim([10,15.5])
subplot(rw,cl,11:12)
imagesc(time,Ner,REC3(1:kd,1:NE)',[0,5])
title('Compressed Bursts')
set(gca,'XTick',[])
ax = gca;
ax.XColor = 'blue';
ax.YColor = 'blue';
set(gca,'linewidth',3)
set(gca,'YDir','normal')
xlim([15.5,16])
colormap(cmaps)
colorbar
set(gca,'XTick',[])
set(gcf,'pos',[20 20 1200 700])


