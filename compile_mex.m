%% Compile all the *.mex files
% 0 - compile for release, 1 - compile for debugging
debugMode = 0;
rawPathEigen = 'C:\Program Files (x86)\eigen-3.4.0';  % replace with your path to Eigen3
clc;

% Expand ~ to full home directory path if present
if startsWith(rawPathEigen, '~')
    pathEigen = fullfile(getenv('HOME'), rawPathEigen(2:end));
else
    pathEigen = rawPathEigen;
end

pathEigen = fullfile(pathEigen);

if ~isfolder(pathEigen)
    error('Eigen path does not exist: %s', pathEigen);
end


%% Compiling the source codes
disp('Compiling mex files:');
disp('----------------------');
cd mex;
delete('*.mexa64');
delete('*.mexw64');
delete('*.pdb');
string = dir('./src/*/*.cpp');
if ispc
    for i = 1:length(string)
        disp(['compiling ',string(i).name]);
        command_string = append(...
            'mex "', string(i).folder, '\', string(i).name, '"', ...
            ' ./src/myfem/src/myfem.cpp -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"', ...
            ' CXXFLAGS="$CXXFLAGS -std=c++14"', ...
            ' -I./src/myfem/include -I"', pathEigen, '"');
        disp(command_string)
        eval(command_string);
        fprintf('\n');
    end
elseif isunix
    for i = 1:length(string)
        disp(['compiling ',string(i).name]);
        command_string = append(...
            'mex -v "', ...
            string(i).folder, ...
            '/', ...
            string(i).name, ...
            '"', ...
            ' ./src/myfem/src/myfem.cpp -largeArrayDims CXXOPTIMFLAGS="-O3 -fwrapv -DNDEBUG" CXXFLAGS="\$CXXFLAGS -Wall -std=c++20" -I./src/myfem/include -I"', ...
            pathEigen, ...
            '"');
        disp(command_string)
        eval(command_string);
        fprintf('\n');
    end
end

cd ..;
