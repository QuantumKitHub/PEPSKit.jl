# Contraction index label helpers
# -------------------------------
#
# These are shared by every contraction expression builder in the package (the CTMRG
# contractions as well as the local-patch ones). Since most of those builders are called from
# the generators of `@generated` functions, the helpers have to be defined *before* any of
# them: a generator can only call methods that already existed when the generated function was
# defined.

function tensorlabel(args...)
    return Symbol(ntuple(i -> iseven(i) ? :_ : args[(i + 1) >> 1], 2 * length(args) - 1)...)
end
envlabel(args...) = tensorlabel(:χ, args...)
virtuallabel(args...) = tensorlabel(:D, args...)
physicallabel(args...) = tensorlabel(:d, args...)
