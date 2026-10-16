for (twistdual, twist, inplace) in zip(
        (:twistdual!, :twistdual), (:twist!, :twist), (" in place ", " ")
    )
    @eval begin
        """
        $(SIGNATURES)

        Twist the i-th leg of a tensor `t`$($(inplace))if it represents a dual space.
        """
        function $twistdual(t::AbstractTensorMap, i::Int)
            isdual(space(t, i)) || return t
            return $twist(t, i)
        end
        function $twistdual(t::AbstractTensorMap, is)
            is′ = filter(i -> isdual(space(t, i)), is)
            return $twist(t, is′)
        end
    end
end

for (twistnondual, twist, inplace) in zip(
        (:twistnondual!, :twistnondual), (:twist!, :twist), (" in place ", " ")
    )
    @eval begin
        """
        $(SIGNATURES)

        Twist the i-th leg of a tensor `t`$($(inplace))if it represents a non-dual space.
        """
        function $twistnondual(t::AbstractTensorMap, i::Int)
            !isdual(space(t, i)) || return t
            return $twist(t, i)
        end
        function $twistnondual(t::AbstractTensorMap, is)
            is′ = filter(i -> !isdual(space(t, i)), is)
            return $twist(t, is′)
        end
    end
end

"""
$(SIGNATURES)

Apply a twist to domain or codomain indices that correspond to dual spaces.
"""
function twist_linearmap!(t::AbstractTensorMap)
    twistdual!(t, 1:numout(t))
    twistnondual!(t, (numout(t) + 1):numind(t))
    return t
end
