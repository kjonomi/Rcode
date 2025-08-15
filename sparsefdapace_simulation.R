# Applied Economics Letters  Volume 31, 2024 - Issue 12, Pages 1142-1149.
# https://www.tandfonline.com/doi/full/10.1080/13504851.2023.2176440
# Finding hidden structure of sparse longitudinal data via functional Eigenfunctions
library(sde)
library(fda)
library(fdapace)
library(dygraphs)
library(xts) 
library(funData)

set.seed(1)
# univariate functional data 
full <-simFunData(argvals = seq(0,1, length.out =100), M = 10, eFunType = "Fourier", eValType = "linear", N = 100)$simData 
sparse <-sparsify(full, minObs = 60, maxObs = 80) 
plot(full, ylim=c(-10,20), main="Sparse functional simulated data") 
plot(sparse, type = "p", pch = 20, add = TRUE) 
legend("topright", c("Full", "Sparse"), lty = c(1, NA), pch = c(NA, 20))


data<-as.data.frame(sparse)

summary(data)


s <- seq(0,1,length.out = 100)
Flies<-MakeFPCAInputs(IDs = rep(1:100, each=100), tVec=rep(s,100), t(data$X))
res1 <-FPCA(Flies$Ly, Flies$Lt, list(dataType='Sparse', error=FALSE, kernel='epan', verbose=FALSE)) 
plot(res1)
res1$cumFVE


eigenvalues    <- as.vector(res1$lambda)
eigenfunctions <-res1$phi
eigenvector1=eigenfunctions[,1]
eigenvector2=eigenfunctions[,2]
eigenvector3=eigenfunctions[,3]
eigenvector4=eigenfunctions[,4]
eigenvector5=eigenfunctions[,5]
eigenvector6=eigenfunctions[,6]
eigenvector7=eigenfunctions[,7]

eigen<-as.matrix(cbind(eigenvector1, eigenvector2, eigenvector3, eigenvector4, eigenvector5, eigenvector6, eigenvector7))

data <- data.frame(
  time= seq(from=0, to=1, length.out=51 ), 
  value1=eigen[,1],
  value2=eigen[,2],
  value3=eigen[,3],
  value4=eigen[,4],
  value5=eigen[,5],
  value6=eigen[,6],
  value7=eigen[,7]
)


plot(data[,1], data[,2], frame = FALSE, type="l", lwd=1, lty = 1,
     col = "red", xlab = "", main = "Eigenfunctions", ylab="Sparse Simulated Data", ylim=c(-3,6))
lines(data[,1], data[,3], pch = 18, col = "blue", lty = 2)
lines(data[,1], data[,4], pch = 18, col = "black", lty = 3)
lines(data[,1], data[,5], pch = 18, col = "green", lty = 4)
lines(data[,1], data[,6], pch = 8, col = "orange",  lty = 5)
lines(data[,1], data[,7], pch = 8, col = "brown",  lty = 6)
lines(data[,1], data[,8], pch = 8, col = "pink",  lty = 7)

legend("topright", legend=c("EigenFun1", "EigenFun2", "EigenFun3", "EigenFun4", "EigenFun5", "EigenFun6", "EigenFun7"),
       col=c("red", "blue", "black", "green", "orange", "brown", "pink"), lty = 1:7, cex=0.5)


#-----------------------------------------------------------------------------#
#   Transform data to standard uniform d.f. through its empirical d.f.        #
#-----------------------------------------------------------------------------#

Empiric.df<-function(data,x)
{	data<-sort(data)

if(min(data)>0) a<-0 else a<-floor(min(data)/100)*100
if(max(data)<0) b<-0 else b<-ceiling(max(data)/100)*100

for(j in 1:length(x))
{
  if(x[j]<a) x[j]<-a
  if(x[j]>b) x[j]<-b
}

data<-c(a,data,b)
n<-length(data)
p<-c(rep(0,(n-1)))
q<-c(rep(0,(n-1)))

for(i in 2:(n-2))
{
  p[i]<-(data[i]+data[i+1])/2
  q[i]<-(i-1)/(n-2)
}
p[1]<-a
p[n-1]<-b
q[1]<-0
q[n-1]<-1
approx(p,q,xout=c(x))$y
}


gene<-eigen


#--------------------------- Initial values ---------------------------------#

col1.n <- length(gene[1,]); col1.n
row1.n <- length(gene[,1]); n <- row1.n; n   

#=============================================================================#
#                 Step to transform original data to U(0,1)                   #
#=============================================================================#

Emp1.index <- matrix(rep(0,n*col1.n),n,col1.n)

for(i in 1:col1.n){
  Emp1.index[,i] <- Empiric.df(gene[,i],gene[,i])
}


Emp.index <- data.frame(Emp1.index)


Emp<-Emp.index

# Simulated Data
Emp1.norm<-as.matrix(cbind(Emp[,1], Emp[,2], Emp[,3], Emp[,4], Emp[,5], Emp[,6], Emp[,7]))


library(mvtnorm)
library(scatterplot3d)
library(sn)
library(mnormt)
library(copula)
library(VineCopula)

RVM <-RVineStructureSelect(Emp[,1:7], c(1:6)) # select the C-vine structure, families and parameters 
CVM <-RVineStructureSelect(Emp[,1:7], c(1:6), type = "CVine") # compare the two models based on the data 
vuong <-RVineVuongTest(Emp[,1:7], RVM, CVM) 
vuong$statistic 
vuong$statistic.Schwarz 
vuong$p.value 
vuong$p.value.Schwarz

plot(RVM)
plot(CVM)



