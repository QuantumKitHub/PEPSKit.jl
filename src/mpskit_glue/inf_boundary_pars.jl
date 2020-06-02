#=
This file defines the necessary functions/structures so that we can simply call leading_boundary(mps,peps,Vumps())
=#

mutable struct Bpars{M,P,T} <: MPSKit.AbstractInfEnv
	peps :: P

	dependency :: M
	tol :: Float64
	maxiter :: Int

	lw::T
	rw::T

	lock::ReentrantLock
end

function MPSKit.params(bmps,peps::InfPEPS;tol=1e-10,maxiter=400)
    @assert length(bmps) == size(peps,1)

	#generate some bogus left fps
	leftfps = PeriodicArray(map(Iterators.product(1:size(bmps,1),1:size(bmps,2))) do (i,j)
		TensorMap(rand,ComplexF64,space(bmps.AL[i+1,j],1)*space(peps[i,j],1)'*space(peps[i,j],1),space(bmps.AL[i,j],1))
	end)

	rightfps = PeriodicArray(map(Iterators.product(1:size(bmps,1),1:size(bmps,2))) do (i,j)
    	TensorMap(rand,ComplexF64,space(bmps.AR[i,j],4)'*space(peps[i,j],3)'*space(peps[i,j],3),space(bmps.AR[i+1,j],4)')
	end)

    pars = Bpars(peps,bmps,tol,maxiter,leftfps,rightfps,ReentrantLock())

	#call recalculate
	MPSKit.recalculate!(pars,bmps;tol=tol,maxiter=maxiter)
end


#=
	we assume that the peps itself doesn't change
=#
function MPSKit.recalculate!(pars,bmps;maxiter = pars.maxiter,tol=pars.tol)
	peps = pars.peps;

	phases = Vector{ComplexF64}(undef,size(bmps,1))

	#recalculate pars.lw[i,1]/pars.rw[i,end]
	jobs = map(1:size(bmps,1)) do i
		#if the bond dimension changed, then we cannot reuse the old solution as initial guess
		if space(pars.lw[i,1],1) != space(bmps.AL[i+1,1],1) || space(pars.lw[i,1],4) != space(bmps.AL[i,1],1)'
			pars.lw[i,1] = TensorMap(rand,ComplexF64,space(bmps.AL[i+1,1],1)*space(peps[i,1],1)'*space(peps[i,1],1),space(bmps.AL[i,1],1))
		end

        lj = @Threads.spawn eigsolve(x->transfer_left(x,peps[i,:],bmps.AL[i,:],bmps.AL[i+1,:]),pars.lw[i,1],1,:LM,Arnoldi());

		if space(pars.rw[i,end],1) != space(bmps.AR[i,end],4)' || space(pars.rw[i,end],4)' != space(bmps.AR[i+1,end],4)'
            pars.rw[i,end] = TensorMap(rand,ComplexF64,space(bmps.AR[i,end],4)'*space(peps[i,end],3)'*space(peps[i,end],3),space(bmps.AR[i+1,end],4)')
        end
        rj = @Threads.spawn eigsolve(x->transfer_right(x,peps[i,:],bmps.AR[i,:],bmps.AR[i+1,:]),pars.rw[i,end],1,:LM,Arnoldi());

		(lj,rj)
	end

	for i in 1:size(bmps,1)
		(vals,vecs,convhist) = fetch(jobs[i][1])
		convhist.converged == 0 && @info "lboundary failed to converge"
		pars.lw[i,1] = vecs[1];
		phases[i] = vals[1]/abs(vals[1])

		(vals,vecs,convhist) = fetch(jobs[i][2])
        convhist.converged == 0 && @info "rboundary failed to converge"
        pars.rw[i,end] = vecs[1];
	end

	# it turns out that these lines are very important - but they're introduced in a weird place
	# also - we change the phase of the boundaries later on again
	# need a better solution in future (change leading boundary code?)
	solutions = similar(phases);
	solutions[1] = 1.0;

	for i in 2:size(bmps,1)
		solutions[i] = solutions[i-1]/phases[i-1]
		rmul!(bmps.AL[i,1],1/solutions[i])
		rmul!(bmps.AR[i,1],1/solutions[i])
		rmul!(bmps.AC[i,1],1/solutions[i])
	end

	#fix the normalization and transfer through
	for i in 1:size(bmps,1)
        val = @tensor pars.lw[i,1][1,2,3,4]*bmps.CR[i,0][4,5]*pars.rw[i,end][5,2,3,6]*conj(bmps.CR[i+1,0][1,6])

		#we can also only divied pars.lw[i,1]; this seemed more symemtric?
		pars.lw[i,1]/=sqrt(val)
        pars.rw[i,end]/=sqrt(val)

        #fill in at other unit sites
        for s in 2:size(peps,2)
            pars.lw[i,s] = transfer_left(		pars.lw[i,s-1],		peps[i,s-1],	bmps.AL[i,s-1],		bmps.AL[i+1,s-1]);
            pars.rw[i,end-s+1] = transfer_right(pars.rw[i,end-s+2],	peps[i,end-s+2],bmps.AR[i,end-s+2],	bmps.AR[i+1,end-s+2]);
        end
    end

	pars.dependency = bmps;

	return pars
end


MPSKit.ac_prime(x,row,col,mps,pars::Bpars) =  @tensor toret[-1 -2 -3;-4]:=MPSKit.leftenv(pars,row,col,mps)[-1,7,8,9]*x[9,3,5,1]*pars.peps[row,col][7,-2,2,3,6]*conj(pars.peps[row,col][8,-3,4,5,6])*MPSKit.rightenv(pars,row,col,mps)[1,2,4,-4]
function MPSKit.ac2_prime(x,row,col,mps,pars::Bpars)
	@tensor toret[-1 -2 -3;-4 -5 -6] := MPSKit.leftenv(pars,row,col,mps)[-1,1,2,3]*x[3,4,5,6,7,8]*pars.peps[row,col][1,-2,9,4,10]*conj(pars.peps[row,col][2,-3,11,5,10])*pars.peps[row,col+1][9,-4,12,6,13]*conj(pars.peps[row,col+1][11,-5,14,7,13])*MPSKit.rightenv(pars,row,col+1,mps)[8,12,14,-6]
end
MPSKit.c_prime(x,row,col,mps,pars::Bpars) = @tensor toret[-1;-2] := MPSKit.leftenv(pars,row,col+1,mps)[-1,2,3,4]*x[4,1]*MPSKit.rightenv(pars,row,col,mps)[1,2,3,-2]
