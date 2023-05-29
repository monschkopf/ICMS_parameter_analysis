% clear workspace
clear all
close all

% load data:
[data_mean, idata] = prepData();

% extract pre test settings

% stimulated channels in those trials:
stim_info.stim_channels = [];
try
    stim_info.stim_channels = idata.QL.Data.CERESTIM_CONFIG_CHAN_PRESAFETY_ARBITRARY.channel(1:2,:);
end
if ~stim_info.stim_channels
    stim_info.stim_channels = idata.QL.Data.CERESTIM_CONFIG_CHAN_PRESAFETY.channel(1,:);
end

stim_info.real_unique_channels = unique(stim_info.stim_channels', 'rows');
stim_info.real_unique_channels = stim_info.real_unique_channels';
nr_trials = size(stim_info.stim_channels,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters:

nr_combinations_per_channel = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




stim_info.real_unique_combinations = zeros(2,length(stim_info.real_unique_channels)*nr_combinations_per_channel);  % +nnz(~stim_info.real_unique_channels)*4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% this differs per subject -> check stimulated electrodes

% variable_combinations_over_channels = nnz(~stim_info.real_unique_channels)*5; % number of single electrode parameter variations that are not captured by channel variations
position_counter = 1;

for q=1:size(stim_info.real_unique_channels,2)
    if stim_info.real_unique_channels(2,q) == 0
        stim_info.real_unique_combinations(1,position_counter:position_counter+nr_combinations_per_channel-1) = repelem(stim_info.real_unique_channels(1,q), nr_combinations_per_channel); % number of all real combinations of channels including amplitudes and frequencies
        position_counter = position_counter+nr_combinations_per_channel;
    else
        stim_info.real_unique_combinations(:,position_counter) = stim_info.real_unique_channels(:,q);
        position_counter = position_counter+1;
    end
end

% dummy channels to be used in further script, as index for real channels and combinations
stim_info.unique_channels = 1:length(stim_info.real_unique_combinations); 


stim_info.nr_reps = zeros(1,numel(stim_info.unique_channels));
stim_info.nr_reps(:) = 20;



stim_info.dummy_channel_atributes = zeros(4,size(stim_info.real_unique_combinations,2));
stim_info.dummy_channel_atributes(1:2,:) = stim_info.real_unique_combinations;
stim_info.dummy_channel_atributes(3,:) = 60;
stim_info.dummy_channel_atributes(4,:) = 100;
single_channels = find(stim_info.real_unique_combinations(2,:) ==0);
% idx_amp_40 = single_channels(2:nr_combinations_per_channel:end);
% idx_amp_80 = single_channels(3:nr_combinations_per_channel:end);
% idx_freq_50 = single_channels(4:nr_combinations_per_channel:end);
% idx_freq_200 = single_channels(5:nr_combinations_per_channel:end);
%stim_info.dummy_channel_atributes(3,single_channels(2:nr_combinations_per_channel:end)) = 40;
%stim_info.dummy_channel_atributes(3,single_channels(3:nr_combinations_per_channel:end)) = 80;
stim_info.dummy_channel_atributes(4,single_channels(1:nr_combinations_per_channel:end)) = 25;
stim_info.dummy_channel_atributes(4,single_channels(2:nr_combinations_per_channel:end)) = 50;
stim_info.dummy_channel_atributes(4,single_channels(3:nr_combinations_per_channel:end)) = 75;
stim_info.dummy_channel_atributes(4,single_channels(4:nr_combinations_per_channel:end)) = 100;

% get stim pulse times:
%
% There are two NSPs, each STIM_SYNC_EVENT message is duplicated. 
% The clocks should be synchronized, but maybe they drift slightly over time. 
% The STIM_SYNC_EVENT.source_index field specifies which NSP you're looking at. 
% So if you index with source_index == 0 you will only look at events for the first NSP. 
%
% There are also two messages for each pulse, one for the rising edge and 
% one for the falling edge of the stim sync signal. The rising edge should 
% be exactly 60 microseconds before each stim pulse was delivered, and it 
% should be very accurate. The falling edge occurs at 1/f (where f is commanded stim frequency) 
% seconds later. After filtering for source_index, it is probably sufficient 
% to just use every other stim sync message (1:2:end), but you can also use the 
% STIM_SYNC_EVENT.data field to verify the digital value (the most robust way I've used is 
% to use bitget to get the binary value of the stim sync bit (9th bit), 
% convert to double or signed int, and take a diff so that rising edges 
% have value 1, and falling edges have value 0).
%
% All stim info -like stim channels, pulse timestamps, etc.- goes into the stim_info struct.

% 1. extract STIM_SYNC_EVENT timing of NSP1
idx_NSP1 = find(idata.QL.Data.STIM_SYNC_EVENT.source_index == 0); % 0 = NSP1, 1 = NSP2
timestamps_NSP1 = idata.QL.Data.STIM_SYNC_EVENT.source_timestamp(idx_NSP1);
% 2. get start of stim pulse only (there are two STIM_SYNC_EVENTS per stim pulse)
pulse_type_NSP1 = idata.QL.Data.STIM_SYNC_EVENT.data(idx_NSP1);
all_pulse_types = unique(pulse_type_NSP1);
idx_pulse1 = find(pulse_type_NSP1 == all_pulse_types(2)); % only look at the rising edge, (1) would be the falling edge
pulse_times = timestamps_NSP1(idx_pulse1);
% 3. adjust for the 60us offset (0.00006)
% pulse_times = pulse_times+0.00006;
stim_info.pulse_time = pulse_times;

% difference between individual pulses:
diff_pulses = round(diff(stim_info.pulse_time),3);
diff_pulses_markers = find(diff_pulses > 1);
diff_markers = [diff_pulses_markers(1), diff(diff_pulses_markers)];
stim_info.stim_length = 1; % 1s of stimulation

% label each pulse with the respective real channel, amplitude & frequency
pulse_counter = 1;
stim_info.first_pulse = [];
stim_info.last_pulse = [];
stim_info.idx = [];
stim_info.real_channel = [];
stim_info.real_channel_list = [];
stim_info.frequencies = [];
stim_info.amplitudes = [];
amplitudes = idata.QL.Data.CERESTIM_CONFIG_MODULE.amp1(1,:);
frequenceis_train_median = zeros(1,nr_trials);
frequenceis_train_mean = zeros(1,nr_trials);
frequenceis_train_count = zeros(1,nr_trials);

for z=1:nr_trials
    if z==nr_trials
        frequency = round(1/diff_pulses(end)); % frequency of stim trains
%         frequency_mean = round(1/diff_pulses(end)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% those need to be changed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         frequency_count = round(1/diff_pulses(end));
        if frequency < 32.5
            frequency = 25;
        elseif frequency > 37.5 && frequency < 62.5
            frequency = 50;
        elseif frequency > 62.5 && frequency < 87.5
            frequency = 75;
        elseif frequency > 87.5 && frequency < 112.5
            frequency = 100;
        elseif frequency > 112.5 && frequency < 137.5
            frequency = 125;
        elseif frequency > 137.5 && frequency < 162.5
            frequency = 150;
        elseif frequency > 162.5 && frequency < 187.5
            frequency = 175;
        elseif frequency > 187.5 && frequency < 212.5
            frequency = 200;
        elseif frequency > 212.5 
            frequency = 225;
        end
    else
%         current_marker = diff_pulses_markers(z); % frequency of stim trains
%         start_vector = 5:44; 
%         marker_vector = current_marker %- start_vector;
%         current_diff_pulses = diff_pulses(marker_vector);
%         current_frequencies = round(1./current_diff_pulses);
%         frequency_mean = round(mean(current_frequencies),-1);
%         frequency_median = median(current_frequencies);
        frequency = diff_markers(z);
    end
    frequenceis_train_median(z) = frequency;
    frequenceis_train_mean(z) = frequency;
    frequenceis_train_count(z) = frequency;

    current_real_channel = repelem(double(stim_info.stim_channels(:,z)),1,stim_info.stim_length*frequency); % repeat each channel name by stim_length*frequency
    frequencies = repelem(double(frequency),1,stim_info.stim_length*frequency); % repeat each frequency by stim_length*frequency
    current_amplitudes = repelem(double(amplitudes(z)),1,stim_info.stim_length*frequency); % repeat each amplitude by stim_length*frequency
    stim_info.real_channel = [stim_info.real_channel, current_real_channel]; 
    stim_info.real_channel_list = [stim_info.real_channel_list, unique(current_real_channel)];
    stim_info.frequencies = [stim_info.frequencies, frequencies];
    stim_info.amplitudes = [stim_info.amplitudes, current_amplitudes];

    % label each stim pulse with stim train index
    current_idx = repelem(z,stim_info.stim_length*frequency); % label stim pulses with stim train index
    stim_info.idx = [stim_info.idx, current_idx];
    % get stim train start and end times
    current_first_pulse = stim_info.pulse_time(pulse_counter);
    pulse_counter = pulse_counter + frequency;
    current_last_pulse =  stim_info.pulse_time(pulse_counter-1);
    stim_info.first_pulse = [stim_info.first_pulse, current_first_pulse];
    stim_info.last_pulse = [stim_info.last_pulse, current_last_pulse];

end

stim_info.frequencies_train_by_median = frequenceis_train_median;
stim_info.frequencies_train_by_mean = frequenceis_train_mean;
stim_info.frequencies_train_by_count = frequenceis_train_count;
% length of ISI:
% ISI = diff_pulses(diff_pulses > 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% need to change dummy channels to account for frequencies and amplitudes as well

% construct dummy channels that code for respective real channels in the 'stim_info.real_unique_channels' field
for v=1:length(stim_info.real_channel) % for each stim pulse, labeled by channel:
    [lia1, locb1] = ismember(stim_info.real_channel(1,v)',stim_info.real_unique_combinations(1,:)'); % check the intersection of this channel with the unique list and report the location
    [lia2, locb2] = ismember(stim_info.real_channel(2,v)',stim_info.real_unique_combinations(2,:)');
    if stim_info.real_channel(2,v) == 0
        if locb1 < locb2
            col = locb2;
            if stim_info.amplitudes(v) == 40
                col = col+1;
            elseif stim_info.amplitudes(v) == 80
                col = col+2;
            end
            if stim_info.frequencies(v) == 50
                col = col+3;
            elseif stim_info.frequencies(v) == 200
                col = col+4;
            end
        else
            col = locb1;
            if stim_info.amplitudes(v) == 40
                col = col+1;
            elseif stim_info.amplitudes(v) == 80
                col = col+2;
            end
            if stim_info.frequencies(v) == 25
                col = col;
            elseif stim_info.frequencies(v) == 50
                col = col+1;
            elseif stim_info.frequencies(v) == 75
                col = col+2;
            elseif stim_info.frequencies(v) == 100
                col = col+3;
            end
        end
    else
        col = locb2;
    end
    stim_info.channel(v) = col; % dummy channels
end

counter_gg = 1;
total_diff = diff(stim_info.channel);
for gg = 1:length(total_diff)
    if total_diff(gg) ~= 0
        channel_list(counter_gg) = stim_info.channel(gg);
        counter_gg = counter_gg+1;
    end
end
channel_list(counter_gg) = stim_info.channel(gg+1);
% clear gg


% get spike snippets:
%
% idata.QL.Data.SPIKE_SNIPPET (spike timestamps are the beginning of the 
% 48-sample snippet, which includes 12 samples before the threshold crossing).
%
% When you look at spike snippets, those messages will also have a source_index 
% specifying which NSP they came from. The anterior pedestal is source_index == 0 
% and posterior pedestal is source_index == 1 . They both have channels 1-128, 
% with 65-96 being stim channels.
%
% all relevant motor info -like spike times and channel names- is stored in the motor_info struct.
% fprintf('\nsorting spikes')

% channel with spike:
spike_channels_NSP1 = idata.QL.Data.SPIKE_SNIPPET.ss.channel(idata.QL.Data.SPIKE_SNIPPET.ss.source_index == 0);
spike_channels_NSP2 = idata.QL.Data.SPIKE_SNIPPET.ss.channel(idata.QL.Data.SPIKE_SNIPPET.ss.source_index == 1);
% timestamps of spikes:
timestamp_spikes_NSP1 = idata.QL.Data.SPIKE_SNIPPET.ss.source_timestamp(idata.QL.Data.SPIKE_SNIPPET.ss.source_index == 0);
timestamp_spikes_NSP2 = idata.QL.Data.SPIKE_SNIPPET.ss.source_timestamp(idata.QL.Data.SPIKE_SNIPPET.ss.source_index == 1);
% concatenate the data from NSP1 and NSP2 for convenience
motor_info.channel = [spike_channels_NSP1, spike_channels_NSP2 + max(spike_channels_NSP1)]; % relabel spike_channels_NSP2 to 128-256
motor_info.spike_time = [timestamp_spikes_NSP1 timestamp_spikes_NSP2]; % append all spike times
motor_info.unique_channels = unique(motor_info.channel); % unique motor channels
% spike waveform:
motor_info.waveform = [idata.QL.Data.SPIKE_SNIPPET.ss.snippet(:,idata.QL.Data.SPIKE_SNIPPET.ss.source_index == 0) idata.QL.Data.SPIKE_SNIPPET.ss.snippet(:,idata.QL.Data.SPIKE_SNIPPET.ss.source_index == 1)];


% main loop


% pre allocate

% count nr of motor spikes in 0.5ms bins, from 10ms prior to a stim pulse until 15ms after
time_bins_pulse = -0.01:0.0005:0.0145; % define time bins for PTAs
time_bins_train = -1:0.05:0.95; % define time bins for stim windows
sum_spike_counts = zeros(numel(stim_info.unique_channels),numel(motor_info.unique_channels), numel(time_bins_pulse)-1); % predefine datastruct for results
sum_spiketimes = zeros(numel(stim_info.unique_channels),numel(motor_info.unique_channels), numel(time_bins_train)-1); % predefine datastruct for results
bin_spike_counts = sum_spike_counts;
blanked_times = cell(numel(stim_info.unique_channels),numel(motor_info.unique_channels));
stim_info.nr_reps_observed = zeros(1,numel(stim_info.unique_channels));
% count spikes across stim trains + in baseline period
start_baseline = -1; 
end_baseline = 0;
sum_baseline_spike_counts = zeros(numel(stim_info.unique_channels),numel(motor_info.unique_channels),20); %40); % predefine datastruct for results
sum_train_spike_counts = zeros(numel(stim_info.unique_channels),numel(motor_info.unique_channels),20); %40); % predefine datastruct for results, there are max 20 reps of a stim train on a single channel, but some channels may occur twice as much
tt = -0.5:0.001:1.499; % scaffolding for instantaneous firing rate
ifiringrate = zeros(numel(unique(motor_info.channel)),2000,numel(stim_info.unique_channels));
zscores_train = sum_train_spike_counts;
wilcoxintest_h = zeros(numel(stim_info.unique_channels),numel(motor_info.unique_channels));
wilcoxintest_p = zeros(numel(stim_info.unique_channels),numel(motor_info.unique_channels));
blank = [];
current_lfp = [];

for s=1:numel(stim_info.unique_channels) % for each stim channel
    s

    y = find(channel_list == s);
    for iy = 1:length(y)
        current_lfp_trial = data_mean.LFP.raw(data_mean.trial_num == y(iy), :);
        current_lfp = [current_lfp; current_lfp_trial];
    end
    lfp{s} = current_lfp; 


    stim_times = stim_info.pulse_time(stim_info.channel==stim_info.unique_channels(s));
    if numel(stim_times) > stim_info.dummy_channel_atributes(4,s)*unique(stim_info.nr_reps)
        stim_times(stim_info.dummy_channel_atributes(4,s)*unique(stim_info.nr_reps)+1:end) = [];
    end
    start_train = stim_times(1:stim_info.stim_length*stim_info.dummy_channel_atributes(4,s):end); % get start of stim trains within this channel
    if numel(start_train) > 20
        start_train(21:end) = [];
    end
    current_frequency = unique(stim_info.frequencies(stim_info.channel==stim_info.unique_channels(s)));
    current_frequency_2 = stim_info.dummy_channel_atributes(4,s);
%     start_train = start_train(1:15); % this is only to adjust for the number of unequal repetitions. Some channels have more than 15, which would affect the statistical power unequally 
%     stim_info.nr_reps_observed(s) = size(stim_times,2)/(stim_info.stim_length*current_frequency);
    pulses = 1/200;

    for m=1:numel(motor_info.unique_channels) % for each motor channel

         selected_channel_spikes = motor_info.spike_time(motor_info.channel == motor_info.unique_channels(m)); % select all spikes of this motor channel
        for t = 1:numel(stim_times) % for all stim pulses in this stim train, but maximally stim_info.nr_reps numbers of repetitions
            relative_spike_times = selected_channel_spikes - stim_times(t); % spike times relative to current stim onset
            sum_spike_counts(s,m,:) = squeeze(sum_spike_counts(s,m,:))' + histcounts(relative_spike_times,time_bins_pulse); % count spikes in timebins
            % in this last step, the spikes are sorted into timebins & ADDED over all repetitions
        end
        % turn collected SUMS in sum_spike_count into average bin spike counts
        bin_spike_counts(s,m,:) = sum_spike_counts(s,m,:)/numel(current_frequency_2*stim_info.nr_reps); % numel(stim_times);


        % firing rates
        ifr_current = zeros(numel(start_train),2000);

        for t = 1:numel(start_train) % for each stim train 
            relative_spike_times = selected_channel_spikes - start_train(t); % spike times relative to onset stim train
            spiketimes = relative_spike_times(relative_spike_times >= -1 & relative_spike_times < 2); % get all spikes a second before, to a second after the stim window
            
            % account for unequal blanking, by blanking everything @ highest stim frequency
            blanks = zeros(current_frequency_2,10);
            for c = 0:599    % to blank all at 200 hz 
                 y = relative_spike_times(relative_spike_times >= -1+pulses*c & relative_spike_times < -1+pulses*c + 0.0015);
                 blanks(c+1,1:length(y)) = y;
            end
              
            spiketimes(ismember(spiketimes,blanks)) = [];
                        
% histcounts was generally faster than comparing relative spike times and then countig the length of those, but since the actual times are necessary for the instantaneous firing rate, it's not possible to switch to histcounts. Else can skip the comparisons and use these histcounts
%             tic
%             sum_train_spike_counts(s,m,t) = histcounts(relative_spike_times,[0, stim_info.stim_length]); % count spikes in stim train                         
%             sum_baseline_spike_counts(s,m,t) = histcounts(relative_spike_times,[start_baseline, end_baseline]); % count spikes in baseline period
%             toc

            spiketimes_base = spiketimes(spiketimes >= -1 & spiketimes < 0); % spiketimes in baseline 1s before stim window
            spiketimes_train = spiketimes(spiketimes >= 0 & spiketimes < 1);  % spiketimes during stim window
            spiketimes_base_2 = spiketimes(spiketimes >= 1 & spiketimes < 2);  % spiketimes in basline 0.5s after stim window
%             spiketimes_base = relative_spike_times(relative_spike_times >= -1 & relative_spike_times < 0); % spiketimes in baseline 1s before stim window
%             spiketimes_train = relative_spike_times(relative_spike_times >= 0 & relative_spike_times < 1); % spiketimes during stim window
%             spiketimes_base_2 = relative_spike_times(relative_spike_times >= 1 & relative_spike_times < 1.5); % spiketimes in basline 0.5s after stim window
            spiketimes_for_ifr = [spiketimes_base, spiketimes_train, spiketimes_base_2]; % spiketimes from 0.5s before, to 0.5s after stim window for ifiringrate
            sum_train_spike_counts(s,m,t) = length(spiketimes_train); % sum of spikes during stim
            sum_baseline_spike_counts(s,m,t) = length(spiketimes_base); % sum of spikes during baseline
            sum_spiketimes(s,m,:) = squeeze(sum_spiketimes(s,m,:))' + histcounts(spiketimes,time_bins_train);
            blanked_times{s,m} = blanks; 
            % instantaneous firing rates
            [ifr,tt] = instantfr(spiketimes_for_ifr,tt); % turn spike times into instantaneous firing rate
            ifr_current(t,:) = ifr;

            

        end

       
        ifiringrate(m,:,s) = mean(ifr_current,1);

        % turn baselines and train spikings into vectors for significance testing
        vector_baseline_with_zeros = squeeze(sum_baseline_spike_counts(s,m,:)); % do i really want the histcounts? Yes, I do. It's one number per repetition that includes the amount of spikes between the defined edges (spike count)
        vector_train_with_zeros = squeeze(sum_train_spike_counts(s,m,:));
        vector_baseline = vector_baseline_with_zeros(1:numel(start_train));
        vector_train = vector_train_with_zeros(1:numel(start_train));

        % perform a paird samples wilcoxin signed rank test on all channel pairs
        [pw,hw] = signrank(vector_baseline,vector_train,'Alpha',0.05);           
        wilcoxintest_h(s,m) = hw;
        wtest_h = wilcoxintest_h(:)'; % transforms output matrix into row vector, so that the same row vector comes out after correcting for multiple comparisons
        wilcoxintest_p(s,m) = pw;
        wtest_p = wilcoxintest_p(:)';
        
        % modulation values
        train = mean(sum_train_spike_counts(s,m,:),3);
        baseline = mean(sum_baseline_spike_counts(s,m,:),3);
        baseline_sd = std(sum_baseline_spike_counts(s,m,:));
        y = (train-baseline)/baseline_sd;
        zscores_train(s,m,1:numel(y)) = y;

    end
end

lfp_info = lfp;

% save values of spike counts
stim_motor_info.sum_baseline_spike_counts = sum_baseline_spike_counts;
stim_motor_info.sum_train_spike_counts = sum_train_spike_counts;
stim_motor_info.sum_spike_counts = sum_spike_counts;
stim_motor_info.bin_spike_counts = bin_spike_counts;
stim_motor_info.sum_spiketimes = sum_spiketimes;
stim_motor_info.zscores = zscores_train;
stim_motor_info.mean_zscores = mean(zscores_train,3);
stim_motor_info.ifiringrate = ifiringrate;

if size(stim_motor_info.sum_baseline_spike_counts,3) > 20

    % cut all values from doubbled channels
    stim_motor_info.sum_baseline_spike_counts(:,:,21:end) = [];
    stim_motor_info.sum_train_spike_counts(:,:,21:end) = [];
    stim_motor_info.sum_spike_counts(:,:,21:end) = [];
    stim_motor_info.bin_spike_counts(:,:,21:end) = [];
    stim_motor_info.zscores(:,:,21:end) = [];
    stim_motor_info.mean_zscores(:,:,21:end) = [];
    stim_motor_info.ifiringrate(:,:,21:end) = [];

end

% correct for multiple comparisons using fdr procedure with Storey's
% correction
[q, fdr,cor_hw] = fdr_storey(wilcoxintest_p,0.05);
wtest_p_cor = zeros(size(wilcoxintest_p));
wtest_p_cor(:) = q; % turns row vector of adjusted p values back into original matrix with dimesions (s,m)
wtest_h_cor = zeros(size(wilcoxintest_p));
wtest_h_cor(:) = cor_hw;
wtest_fdr = zeros(size(wilcoxintest_p));
wtest_fdr(:) = fdr;
stim_motor_info.wilcoxintest_h = wtest_h_cor; % save values
stim_motor_info.wilcoxintest_p = wtest_p_cor;
stim_motor_info.wilcoxintest_fdr = wtest_fdr;


% summary of channels that are sig modulated by stimulation
[row,col] = find(stim_motor_info.wilcoxintest_h == 1);
stim_motor_info.active_channels_w(:,1) = row;
stim_motor_info.active_channels_w(:,2) = col;
stim_motor_info.active_channels_percent_w = size(stim_motor_info.active_channels_w,1)/((numel(stim_info.unique_channels)*numel(motor_info.unique_channels))/100);


% % modulated bins
% % data_z_auto = zeros(size(sum_spike_counts));
% data_z = zeros(size(sum_spike_counts));
% % data_z_sigidx = zeros(size(sum_spike_counts,1),size(sum_spike_counts,2));
% for s=1:numel(stim_info.unique_channels)
%     for m=1:numel(motor_info.unique_channels)
%         % go through data_mean and turn values into baseline zscores
%         
% %         if any(data_z(s,m,:) > 1.96)
% %             data_z_sigidx(s,m) = 1;
% %         else
% %             data_z_sigidx(s,m) = 0;
% %         end
%     end
% end
% 
% % data_z_p = normcdf(data_z);
% [cor_p_data_z, cor_h_data_z] = bonf_holm(data_z_p);
% proportion_mod_bins = numel(find(data_z_sigidx == 1))/numel(data_z_sigidx);

% data_z_mod = data_z(:,:,13:20);
% data_z_mod_5 = mean(data_z_mod,3);
% data_z_mod_5_max = max(data_z_mod,[],3);
%  
% stim_motor_info.data_z = data_z;
% stim_motor_info.data_z_modulated_4ms = data_z_mod_5;

save('CRS02_sess3freq1_new_stim_motor_info.mat', 'stim_motor_info', '-v7.3')
save('CRS02_sess3freq1_new_stim_info.mat', 'stim_info', '-v7.3')
save('CRS02_sess3freq1_new_motor_info.mat', 'motor_info', '-v7.3')
save('CRS02_sess3freq1_new_lfp_info.mat', 'lfp_info', '-v7.3')