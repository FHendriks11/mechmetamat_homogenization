classdef transformer

    properties (Access=private)
        F
        U
        R

        mapped = false;
        Urot
        Uref
        Trot
        Tref

        Ufinal
    end


    methods
        function obj = transformer(F, mapping)
            obj.F = F;
            [obj.U, obj.R] = perform_polar_decomposition(F);
            obj.Ufinal = obj.U;

            if nargin == 2 && ~isempty(mapping)
                obj.mapped = true;
                [obj.Urot, obj.Uref, obj.Trot, obj.Tref] = mapping(obj.U);
                obj.Ufinal = obj.Uref;
            end
        end

        function Ufinal = get_U(obj)
            Ufinal = obj.Ufinal;
        end

        function [Pmacro, Cmacro] = upscale(obj, Ppseudo, Cpseudo)
            if obj.mapped
                Pintermediate = obj.Trot' * obj.Tref' * Ppseudo;
                Cintermediate = obj.Trot' * obj.Tref' * Cpseudo * obj.Tref * obj.Trot;
            else
                Pintermediate = Ppseudo;
                Cintermediate = Cpseudo;
            end

            dUdF = compute_dUdF(obj.F);
            d2UdF2 = compute_d2UdF2(obj.F);

            Pd2UdF2 = zeros(4);
            for iii = 1:length(Pintermediate)
                Pd2UdF2 = Pd2UdF2 + Pintermediate(iii) * squeeze(d2UdF2(iii,:,:));
            end

            Pmacro = dUdF' * Pintermediate;
            Cmacro = Pd2UdF2 + dUdF' * Cintermediate * dUdF;
        end

    end
end