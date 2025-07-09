function [ options ] = add_default_options(options, defaultOptions)
% ADD_DEFAULT_OPTIONS parses inOptions and add missing entries from
% defaultOptions.
% 
%   [ options ] = add_default_options(options, defaultOptions)
%
% Version:  0.1.0 (2020-11-13)
% Author:   Martin Doskar (MartinDoskar@gmail.com)

for f = fieldnames(defaultOptions)'
  if ~isfield(options, f{1})
      options.(f{1}) = defaultOptions.(f{1});
  end
end

end