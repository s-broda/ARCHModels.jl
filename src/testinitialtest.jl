using Roots
numsim=1000;
res1=zeros(numsim);
res2=zeros(numsim);
res3=zeros(numsim);
res4=zeros(numsim);
v=5.;
r=1
const cutoff=30.
eabst(v)=2*sqrt(v-2)/(v-1)/beta(v/2, 1/2)
eabst(v, r)=(v-2)^(r/2)*gamma((v-r)/2)*gamma((r+1)/2)/gamma(v/2)/gamma(1/2)
logeabst(v, r)=(r/2)*log(v-2)+lgamma((v-r)/2)+lgamma((r+1)/2)-lgamma(v/2)-lgamma(1/2)
g2(v)=log(v-2)/2-digamma(v/2)/2+digamma(1/2)/2

g(v)=.25*(polygamma(1, v/2)+polygamma(1, .5))
myfit3(data)=(z=mean(log.(abs.(data))); z>g2(cutoff) ? cutoff : find_zero(x->z-g2(x), (2., cutoff)))
#"Tail Index Estimation for Parametric Families Using Log Moments"
myfit2(data)=(z=var(log.(abs.(data))); z<g(cutoff) ? cutoff : find_zero(x->z-g(x), (2., cutoff)))
myfit(data, r)=(z=mean(abs.(data).^r);z>eabst(cutoff, r) ? cutoff : find_zero(x->z-eabst(x, r), (2., cutoff)))
 for i=1:numsim;
     println(i)
     data=rand(StdTDist(v), 4000);#data./=std(data)
     res1[i]=myfit(data, r);
     #res2[i]=myfit2(data)
     res3[i]=myfit3(data)
     res4[i]=min(fit(StdTDist, data).ν, cutoff);
 end
bias1=mean(res1-v)
bias2=mean(res2-v)
bias3=mean(res3-v)
bias4=mean(res4-v)
rmse1=sqrt(mean((res1-v).^2))
rmse2=sqrt(mean((res2-v).^2))
rmse3=sqrt(mean((res3-v).^2))
rmse4=sqrt(mean((res4-v).^2))
println([bias1, bias2, bias3, bias4,  rmse1, rmse2, rmse3, rmse4])
