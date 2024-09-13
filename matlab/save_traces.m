%% save_traces.m
% 2024.09.04 CDR
% 
% Restructure KS traces struct so each row contains only one instance of a
% syllable. For better loading into python.

files = [   "/Volumes/users/kswartz/ForCiro/AllBirdsAnlaysis_TrialNormV2_Aug2024bk61wh41_AlignedTraces_iabcdefgh.mat"
            "/Volumes/users/kswartz/ForCiro/AllBirdsAnlaysis_TrialNormV2_Aug2024pu68bk38_AlignedTraces_iabcdefghklmno.mat"
            "/Volumes/users/kswartz/ForCiro/AllBirdsAnlaysis_TrialNormV2_Aug2024pu97bk73_AlignedTraces_abcdefghijr.mat"
];

save_folder = "/Users/cirorandazzo/code/calcium_classifiers";


for i_f=1:length(files)

    f = files(i_f);

    [~, name, ext] = fileparts(f);

    load(f, "traces");
    
    by_syllable = [];
    
    for i_syl = 1:length(traces)

        if isempty(traces(i_syl).dff)  % empty syllable
            continue;
        end

        % preallocate length
        syls = cell( size(traces(i_syl).dff) );
        [syls{:}] = deal( traces(i_syl).syl );
        i = num2cell(1:length(traces(i_syl).dff));       

        syl_data = struct('syl', syls, 'i', i);
        
        for field = string(fields(traces))'
            d = traces(i_syl).(field);

            if any(strcmp(field, {'time_rel', 'syl'}))
                continue;
            elseif iscell(d)
                % already good!
            elseif isa(d, 'double')
                matching_dim = (size(d) == length(syl_data));
                if any(matching_dim)
                    d = num2cell(d, find(~matching_dim));
                    d = cellfun(@squeeze, d, UniformOutput=false);
                else
                    p = cell( size(traces(i_syl).dff) );
                    [p{:}] = deal(d);
                    d = p;
                end
            end

            [syl_data.(field)] = d{:};

        end

        if isempty(by_syllable)
            by_syllable = syl_data;
        else
            by_syllable = [by_syllable syl_data];
        end

    end

    save( ...
        fullfile(save_folder, name+ext), ...
        "by_syllable", ...
        "-v7" ...  % scipy.io.loadmat can't do MAT-v7.3
        );

end

%%

for i_syl = 1:length(traces)
    
    find(traces(i_syl).time_rel ~= traces(1).time_rel)
end