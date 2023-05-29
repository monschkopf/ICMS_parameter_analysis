
% load('CRS07_sess3freq1_new_stim_motor_info.mat')
% load('CRS07_sess3freq1_new_motor_info.mat')
% load('CRS07_sess3freq1_new_stim_info.mat')
% 



% amplitudes

% fill a variable amp with the respective dummy channels occurances in the
% modulated channels list
amp40 = [
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 2,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 8,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 13,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 20,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 25,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 31,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 37,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 42,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 47,:)
];

amp60 = [
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 1,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 7,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 12,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 19,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 24,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 30,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 36,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 41,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 46,:)
];

amp80 = [
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 3,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 9,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 14,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 21,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 26,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 32,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 38,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 43,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 48,:)
];

amp40histcounts = histcounts(amp40(:,1));
amp40histcounts = amp40histcounts(find(amp40histcounts));
amp60histcounts = histcounts(amp60(:,1));
amp60histcounts = amp60histcounts(find(amp60histcounts));
amp80histcounts = histcounts(amp80(:,1));
amp80histcounts = amp80histcounts(find(amp80histcounts));

amp40histcounts(2,:) = unique(stim_info.dummy_channel_atributes(1,:)); 
amp60histcounts(2,:) = unique(stim_info.dummy_channel_atributes(1,:));
amp80histcounts(2,:) = unique(stim_info.dummy_channel_atributes(1,:));

amp40histcounts(3,:) = 40;
amp60histcounts(3,:) = 60;
amp80histcounts(3,:) = 80;

mean40 = mean(amp40histcounts(1,:));
mean60 = mean(amp60histcounts(1,:));
mean80 = mean(amp80histcounts(1,:));
meanamps(2,:) = [40, 60, 80];

amphistcounts = [amp40histcounts, amp60histcounts, amp80histcounts];

figure

hold on
plot(meanamps(2,:),meanamps(1,:))

gscatter(amphistcounts(3,:),amphistcounts(1,:),amphistcounts(2,:))
xticks([40,60,80])

xlim([30 90])
ylim([30 130])
xlabel('amplitude')
ylabel('# modulated channels')
title('Amplitude vs # of channels activated')
xlim([30 100])
legend('Location','northeast')
title(legend, 'channel')

%% 

% frequencies

freq25 = [
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 1,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 5,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 9,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 13,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 17,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 21,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 25,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 29,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 33,:)
];

freq50 = [
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 2,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 6,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 10,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 14,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 18,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 22,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 26,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 30,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 34,:)
];

freq75 = [
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 3,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 7,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 11,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 15,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 19,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 23,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 27,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 31,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 35,:)
];

freq100 = [
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 4,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 8,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 12,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 16,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 20,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 24,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 28,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 32,:)
stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 36,:)
];


freq25h = histcounts(freq25(:,1));
freq25h = freq25h(find(freq25h));
freq50h = histcounts(freq50(:,1));
freq50h = freq50h(find(freq50h));
freq75h = histcounts(freq75(:,1));
freq75h = freq75h(find(freq75h));
freq100h = histcounts(freq100(:,1));
freq100h = freq100h(find(freq100h));
% freq200h = histcounts(freq200(:,1));
% freq200h = freq200h(find(freq200h));

freq25h(2,:) = unique(stim_info.dummy_channel_atributes(1,:));
freq50h(2,:) = unique(stim_info.dummy_channel_atributes(1,:));
freq75h(2,:) = unique(stim_info.dummy_channel_atributes(1,:));
freq100h(2,:) = unique(stim_info.dummy_channel_atributes(1,:));
% freq200h(2,:) = unique(stim_info.dummy_channel_atributes(1,:));

freq25h(3,:) = 25;
freq50h(3,:) = 50;
freq75h(3,:) = 75;
freq100h(3,:) = 100;
% freq200h(3,:) = 200;

freqhistcounts = [freq25h, freq50h, freq75h, freq100h]; % freq200h];

meanfreq25 = mean(freq25h(1,:));
meanfreq50 = mean(freq50h(1,:));
meanfreq75 = mean(freq75h(1,:));
meanfreq100 = mean(freq100h(1,:));
% meanfreq200 = mean(freq200h(1,:));

meanfreqs = [meanfreq25, meanfreq50, meanfreq75, meanfreq100]; %, meanfreq200];
meanfreqs(2,:) = [25, 50, 75, 100]; %, 200];

figure

hold on
plot(meanfreqs(2,:),meanfreqs(1,:))

gscatter(freqhistcounts(3,:),freqhistcounts(1,:),freqhistcounts(2,:))

% title(legend, 'channel')
title(legend, 'channel')
title('frequency vs # modulated channeld')
xlabel('frequency')
ylabel('# of modulated channels')

xlim([10 120])
% xticks([40, 100, 200])
xticks([25, 50, 75, 100]) %, 200])
title('frequency vs # of modulated channeld')

%% 

% multielectrode

multirefmulti = stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 18,:);
multirefarray = stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 29,:);
multiindex = stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 6,:);
multimiddle = stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 35,:);
multithumb = stim_motor_info.active_channels_w(stim_motor_info.active_channels_w(:,1) == 17,:);

multihists = [length(multirefmulti), length(multirefarray), length(multiindex), length(multimiddle), length(multithumb)];

figure

hold on
scatter([1:5], multihists)
xticks([1,2,3,4,5])
xticklabels({'max across arrays', 'max inside array', 'index', 'middle', 'thumb'})
xlim([0 6])
ylim([0 130])
ylim([0 200])
title('multichannel stimulation vs # of modulated channeld')
xlabel('electrode group')
ylabel('# of channels modulated')


%%

% modulation depth



