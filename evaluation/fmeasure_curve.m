% ################# Plot F-measure Figure ################
% X-axis: Threshold
% Y-axis: F-measure
% ###################################################
clc;
clear;
basedir = './result/';
algorithms = {
    'Ours';
    'A2dele';
    'AFNet';
    'CoNet'; 
    'PoolNet';
    'CSNet';
    'EGNet';
    'D3Net';  
    'DANet';  
    'DMRA'; 
    'FRDT'  
};
datasets = {
    'DES';     
    'LFSD';
    'NJU2K';
    'NLPR';
    'SSD';
    'SIP';
    'STERE';
};

index = 7;
dataset = datasets{index};
str=['r', 'g', 'g', 'g', 'b', 'b', 'b', 'y', 'y', 'y', 'k', 'k', 'k'];
% str=['r','r','r','g','g','g','b','b','b','y','y','y','k','k','k','g','g','b','b','m','m','k','k','r','r','b','b','c','c','m','m'];
rr=[];
for i=1:length(algorithms)
  %  load([path num2str(aa(i)) '-filter.mat']);
  load([basedir algorithms{i} '/' dataset '/prec.mat']);
  load([basedir algorithms{i} '/' dataset '/rec.mat']);
  threshold = linspace(0, 255, 256);
  fmeasure = ((1.3.*prec.*rec)./(0.3*prec+rec));
  
  if i == 1
    plot(threshold,fmeasure,[str(i) '-'],'linewidth',5);  
  elseif mod(i,3)==0
    plot(threshold,fmeasure,[str(i) ':'],'linewidth',1.5);
  elseif mod(i,3)==1
    plot(threshold,fmeasure,[str(i) '-'],'linewidth',1.5);
  elseif mod(i,3)==2
    plot(threshold,fmeasure,[str(i) '--'],'linewidth',1.5);
  end
  hold on;
  display([algorithms(i)])% '----' num2str(a)]);

  %  dirpath(i).name
 %  display(num2str(max(pM)));
end
title(dataset, 'fontsize', 30);

xlabel('Threshold', 'fontsize', 30);
ylabel('F-measure', 'fontsize', 30);
axis([0 260 0.6 0.95])
l = legend(algorithms{1}, algorithms{2},algorithms{3}, algorithms{4},...
    algorithms{5}, algorithms{6},algorithms{7},algorithms{8},algorithms{9},algorithms{10},algorithms{11});
set(l, 'Fontsize', 30)
grid
set( gca , 'fontsize', 30 );
set(gca,'LooseInset',get(gca,'TightInset'));
set(gca,'looseInset',[0 0 0 0]);