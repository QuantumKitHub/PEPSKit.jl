#next step - make this j1;j2
const NN{S} = AbstractTensorMap{S,2,2} where S

Base.rotl90(st::NN) = st
Base.rotr90(st::NN) = st


#nn thing
#m1;m2 = left
#cbt = top tensor
#m3;m4 = right
#[t1;t2]
hamtransfer(m1,m2,m3,m4,cbt,t1,t2,nn::NN;bt1=t1,bt2=t2) =
@tensor toret[-1 -2 -3;-4]:=m1[-1,18,19,20]*m2[20,4,2,1]*cbt[1,5,3,6]*m3[6,7,8,11]*m4[11,12,15,-4]*
t1[4,13,7,5,9]*t2[18,-2,12,13,14]*conj(bt1[2,16,8,3,10])*conj(bt2[19,-3,15,16,17])*
nn[10,9,17,14]
hamtransfer(m1,m2,cbt,t1,t2,nn::NN;bt1=t1,bt2=t2) =
@tensor toret[-1 -2 -3 -4 -5;-6]:=m1[-1,14,15,16]*cbt[16,12,9,3,5,1]*m2[1,2,4,-6]*
t1[14,-2,11,12,13]*conj(bt1[15,-3,8,9,10])*t2[11,-4,2,3,6]*conj(bt2[8,-5,4,5,7])*
nn[10,13,7,6]


#plaq thing
#m1;m2 = left
#cbt = top tensor
#m3;m4 = right
#[t1 t2;t3 t4]
#hamtransfer(m1,m2,m3,m4,cbt,t1,t2,t3,t4,plaq) = @tensor m1[-1,30,31,32]*m2[32,8,2,1]*cbt[1,9,3,12,5,6]*m3[6,11,7,22]*m4[22,24,23,-6]*
#tplaq[13,14,29,18,26,21,15,16]*
#t1[8,28,10,9,13]*conj(t1[2,17,4,3,14])*t2[10,25,11,12,15]*conj(t2[4,20,7,5,16])*
#t3[30,-2,27,28,29]*conj(t3[31,-3,19,17,18])*t4[27,-4,24,25,26]*conj(t4[19,-5,23,20,21])
