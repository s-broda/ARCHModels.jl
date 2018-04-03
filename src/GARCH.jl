using Base.Cartesian: @nexprs
export GARCH
struct GARCH{p,q} <: VolatilitySpec end

@generated function arch_loglik!{p, q, T1<:FP}(M::Type{GARCH{p,q}}, data::Vector{T1}, ht::Vector{T1}, coefs::T1...)
  r = max(p, q)
  log2pi = T1(1.837877066409345483560659472811235279722794947275566825634303080965531391854519)
  @assert length(coefs) == p+q+1 "Incorrect number of parameters: expected $(p+q+1), got $(length(coefs))."
  quote
    T = length(data)
    T > $r || error("Sample too small.")
    @inbounds begin
      LL = $(zero(T1))#GARCH.jl uses sum(log.(ht[1:$r])+datasq[1:$r]./ht[1:$r]) instead of 0.
      den = $(one(T1))
      @nexprs $(p+q) i -> den -= coefs[i+1]
      h0 = coefs[1]/den
      @nexprs $r i -> ht[i] = h0
      @fastmath for t = $(r+1):T
        ht[t] = coefs[1]
        @nexprs $p i -> ht[t] += coefs[i+1]*ht[t-i]
        @nexprs $q i -> ht[t] += coefs[i+$(1+p)]*data[t-i]^2
        LL += log(ht[t]) + data[t]^2/ht[t]
      end#for
    end#inbounds
    LL = -((T-$r)*$log2pi+LL)/2
  end#quote
end#function


@generated function archsim!{p,q}(M::Type{GARCH{p, q}}, data, ht, coefs...)
  r = max(p,q)
  @assert length(coefs) == p+q+1 "Incorrect number of parameters: expected $(p+q+1), got $(length(coefs))."
  quote
    T=length(data)
    den = one(coefs[1]);
    @nexprs $(p+q) i -> den -= coefs[i+1]
    h0 = coefs[1]/den
    randn!(@view data[1:$r])
    @nexprs $r i -> data[i] *= h0
    @assert T > $r "Sample too small."
    @inbounds begin
      for t = $(r+1):T
        ht[t] = coefs[1]
        @nexprs $p i -> ht[t] += coefs[i+1]*ht[t-i]
        @nexprs $q i -> ht[t] += coefs[i+$(1+p)]*data[t-i]^2
        data[t] = sqrt(ht[t])*randn()
      end
    end
  end#quote
end#function

function fit{p, q}(G::Type{GARCH{p,q}}, data)
  ht = zeros(data)
  obj = x -> -arch_loglik!(G, data, ht, x...)
  #  optimize(DifferentiableFunction(obj), [1., .8, .1], BFGS(), OptimizationOptions(autodiff = true))
end
fit(G::Type{GARCH}, data)="find best GARCH model by AIC"
