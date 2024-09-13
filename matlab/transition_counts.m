transition_count = [];

for i_syl=1:length(traces)

    transition_count(i_syl).syl = traces(i_syl).syl;

    post_syls = traces(i_syl).postSyls;

    for i_tr = 1:length(post_syls)
        tr = post_syls{i_tr};

        if isscalar(tr)
            char = 'END';
        else
            char = tr(2);
        end
        
        if char == '-'
            char = 'DASH';
        end
    
        if ~isfield(transition_count, char) || isempty(transition_count(i_syl).(char))
            transition_count(i_syl).(char) = 1;
        else
            transition_count(i_syl).(char) = transition_count(i_syl).(char) + 1;
        end
    end
end


%%
G = digraph();