using Revise, PEPSKit, TensorKit, Zygote, MPSKit

p = InfinitePEPS(fill(ℂ^2,1,1),fill(ℂ^2,1,1));

trans = PEPSKit.InfiniteTransferPEPS(p,1,1);
mps = PEPSKit.initializeMPS(trans,[ℂ^5]);

leading_boundary(mps,trans,VUMPS())

