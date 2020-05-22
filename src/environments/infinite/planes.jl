#leverage mpskit to find the leading boundary
function north_boundary_mps(peps,init;tol=1e-10,verbose=false,maxiter=100,bound_finalize=(iter,state,ham,pars)->(state,pars))
	(init,pars,err) = leading_boundary(init,peps,Vumps(tol_galerkin=tol,verbose=verbose,maxiter=maxiter,finalize=bound_finalize))

	#=
		we have the leading bounary mps in the sense that it's (approximate) eigenvalue after transferring over the entire unit cell is maximal
		however, note that for a 4 site unit cell the solution (1 1 1 1) and (1 -1 -1 1) both would have a maximal dominant eigenvalue
		long story short - (I think) we need to gauge fix the mps
	=#

	for i in 1:size(peps,1)
		phase = dot(init.AC[i+1,1],MPSKit.ac_prime(init.AC[i,1],i,1,init,pars::Bpars))
		phase /= abs(phase);

		for j in 1:size(peps,2)
			rmul!(init.AC[i+1,j],phase)
			rmul!(init.CR[i+1,j],phase)
		end
	end

	return (init,pars,err)
end

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
	#recalculate pars.lw[i,1]
	for i in 1:size(bmps,1)
		#if the bond dimension changed, then we cannot reuse the old solution as initial guess
		if space(pars.lw[i,1],1) != space(bmps.AL[i+1,1],1) || space(pars.lw[i,1],4) != space(bmps.AL[i,1],1)'
			pars.lw[i,1] = TensorMap(rand,ComplexF64,space(bmps.AL[i+1,1],1)*space(peps[i,1],1)'*space(peps[i,1],1),space(bmps.AL[i,1],1))
		end

        (vals,vecs,convhist) = eigsolve(x->transfer_left(x,peps[i,:],bmps.AL[i,:],bmps.AL[i+1,:]),pars.lw[i,1],1,:LM,Arnoldi());
        convhist.converged == 0 && @info "lboundary failed to converge"
        pars.lw[i,1] = vecs[1];

		phases[i] = vals[1]/abs(vals[1])
	end

	# it turns out that these lines are very important - but they're introduced in a weird place
	# also - we change the phase of the boundaries later on again
	# need a better solution in future
	solutions = similar(phases);
	solutions[1] = 1.0;

	for i in 2:size(bmps,1)
		solutions[i] = solutions[i-1]/phases[i-1]
		rmul!(bmps.AL[i,1],1/solutions[i])
		rmul!(bmps.AR[i,1],1/solutions[i])
		rmul!(bmps.AC[i,1],1/solutions[i])
	end

	#recalculate pars.rw[i,end]
	for i in 1:size(bmps,1)
        if space(pars.rw[i,end],1) != space(bmps.AR[i,end],4)' || space(pars.rw[i,end],4)' != space(bmps.AR[i+1,end],4)'
            pars.rw[i,end] = TensorMap(rand,ComplexF64,space(bmps.AR[i,end],4)'*space(peps[i,end],3)'*space(peps[i,end],3),space(bmps.AR[i+1,end],4)')
        end
        (vals,vecs,convhist) = eigsolve(x->transfer_right(x,peps[i,:],bmps.AR[i,:],bmps.AR[i+1,:]),pars.rw[i,end],1,:LM,Arnoldi());
        convhist.converged == 0 && @info "rboundary failed to converge"
        pars.rw[i,end] = vecs[1];

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
