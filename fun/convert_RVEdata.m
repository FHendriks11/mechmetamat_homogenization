function RVEdata = convert_RVEdata(RVEnew)

    RVEdata.RVEmesh.nodes = [RVEnew.p, ones(size(RVEnew.p,1),1)];
    RVEdata.RVEmesh.nNodes = size(RVEnew.p, 1);
    RVEdata.RVEmesh.elements = double(RVEnew.t + 1);
    RVEdata.RVEmesh.nElems = size(RVEnew.t, 1);
    RVEdata.RVEmesh.elemMats = ones(RVEdata.RVEmesh.nElems, 1);

    if isfield(RVEnew, 'lattice_vectors')
        lv = RVEnew.lattice_vectors;
        RVEdata.RVEmesh.FE2.V = abs(lv(1,1)*lv(2,2)-lv(2,1)*lv(1,2))*4;
    else
        RVEdata.RVEmesh.FE2.V = 1;
        warning('No lattice vectors found');
    end

    % convert the boundary indices to the right type/shape
    % annoying because depending on shape the imported data can end up a cell or an array
    if iscell(RVEnew.boundary_inds)
        if size(RVEnew.boundary_inds, 2) == 6
            b1 = RVEnew.boundary_inds(1);
            b2 = RVEnew.boundary_inds(2);
            b3 = RVEnew.boundary_inds(3);
            b4 = RVEnew.boundary_inds(4);
            b5 = RVEnew.boundary_inds(5);
            b6 = RVEnew.boundary_inds(6);
            source_temp = transpose(double([b1{1,1}, b2{1,1},b3{1,1}] +1));
            image_temp = transpose(double([b4{1,1},b5{1,1},b6{1,1}] +1));
        elseif size(RVEnew.boundary_inds, 2) == 4
            b1 = RVEnew.boundary_inds(1);
            b2 = RVEnew.boundary_inds(2);
            b3 = RVEnew.boundary_inds(3);
            b4 = RVEnew.boundary_inds(4);
            source_temp = transpose(double([b1{1,1}, b2{1,1}] +1));
            image_temp  = transpose(double([b3{1,1}, b4{1,1}] +1));
        else
            throw Exception("cell size is not 4 or 6");
        end
    else
        if size(RVEnew.boundary_inds,1) == 4
            source_temp = transpose(double([RVEnew.boundary_inds(1,:), RVEnew.boundary_inds(2,:)] +1));
            image_temp = transpose(double([RVEnew.boundary_inds(3,:), RVEnew.boundary_inds(4,:)] +1));
        else
            throw Exception("not implemented yet")
        end
    end

    temptemp = [source_temp, image_temp];

    % remove boundary indices overlap
    ok = 0;
    while ~ok
        [C,ia,ib] = intersect(source_temp, image_temp);
    %     C
    %     size(C)
        if size(C, 1) > 0
            source_temp(ia(1)) = source_temp(ib(1));
            temptemp = [source_temp, image_temp];

            % only unique combinations
            temptemp = unique(temptemp, 'rows');

            % no nodes linked to themselves
            temptemp = temptemp(~(temptemp(:, 1) == temptemp(:, 2)), :);

            source_temp = temptemp(:, 1);
            image_temp = temptemp(:, 2);
        else
            ok = 1;
        end
    end
    RVEdata.RVEmesh.FE2.periodicSourceNodes = temptemp(:, 1);
    RVEdata.RVEmesh.FE2.periodicImageNodes = temptemp(:, 2);

    RVEdata.RVEmesh.FE2.fixedNode = RVEdata.RVEmesh.FE2.periodicSourceNodes(1);


end