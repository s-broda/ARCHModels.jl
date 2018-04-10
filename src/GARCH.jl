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
      LL = $(zero(T1))
      den = $(one(T1))
      @nexprs $(p+q) i -> den -= coefs[i+1]
      h0 = max(coefs[1]/den, 0)
      lh0 = log(h0)
      @nexprs $r i -> ht[i] = h0
      @nexprs $r i -> LL += lh0+data[i]^2/h0
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

function archstart{p, q, T}(G::Type{GARCH{p,q}}, data::Array{T})
  x0 = zeros(T, p+q+1)
  x0[2:p+1] = 0.9/p
  x0[p+2:end] = 0.05/q
  x0[1] = var(data)*(one(T)-sum(x0))
  return x0
end

function fit{p, q, T}(G::Type{GARCH{p,q}}, data::Array{T}, args...; kwargs...)
  q == 0 && return ARCHModel(G, data, Tuple([mean(data.^2); zeros(T, p)]))
  ht = zeros(data)
  obj = x -> -arch_loglik!(G, data, ht, x...)
  x0 = archstart(G, data)
  res = optimize(obj, x0, args...; kwargs...)
  return ARCHModel(G, data, Tuple(res.minimizer))
end

function selectmodel(G::Type{GARCH}, data, maxp=3, maxq=3, args...; criterion=bic, kwargs...)
  res = Array{ARCHModel, 2}(maxp+1, maxq+1)
  for p = 0:maxp, q = 0:maxq
    res[p+1, q+1] = fit(GARCH{p, q}, data, args...; kwargs...)
  end
  crits = criterion.(res)
  println(crits)
  _, ind = findmin(crits)
  return res[ind]
end
