function cfg = parse_inputs(defaultCfg, varargin)
    cfg = defaultCfg;
    nVar = numel(varargin);
    if mod(nVar,2)~=0
        error('parse_inputs:invalidArgs', 'Arguments must be name/value pairs.');
    end
    for ii = 1:2:nVar
        name = varargin{ii};
        val  = varargin{ii+1};
        if isfield(cfg, name)
            cfg.(name) = val;
        else
            error('parse_inputs:unknownParam', ...
                  'Unknown parameter name: %s', name);
        end
    end
end