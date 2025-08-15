# Statistics A Journal of Theoretical and Applied Statistics Volume 58, 2024 - Issue 3, Pages 749-769
# https://www.tandfonline.com/doi/full/10.1080/02331888.2024.2364688
# Copula deep learning control chart for multivariate zero inflated count response variables
library(neuralnet)
library(copula)

z<-1000    # number of iterations
y<-13     # number of categorical variables
theta<-8 # copula parameter
n<-1000  # number of data  
p<-13    # total number of variables
#o<-0.5   # missing rate
#conf <- conf2 <- matrix(rep(0,y*z),y,z)

RMSE.zip<-NULL

RMSE.zip<- matrix(rep(0,z*3),z,3)

RMSE.NN<-RMSE.NN1<-RMSE.DL<-RMSE.DL1<-NULL

RMSE.NN<-RMSE.NN1<-RMSE.DL<-RMSE.DL1<-matrix(rep(0,z*3),z,3)

RMSE.gcmr.NN1<-RMSE.gcmr.DL1<-NULL

RMSE.gcmr.NN1<-RMSE.gcmr.DL1<-matrix(rep(0,z*3),z,3)

RMSE.vine.NN1<-RMSE.vine.DL1<-NULL

RMSE.vine.NN1<-RMSE.vine.DL1<-matrix(rep(0,z*3),z,3)

WMAPE.zip<-NULL

WMAPE.zip<- matrix(rep(0,z*3),z,3)

WMAPE.NN<-WMAPE.NN1<-WMAPE.DL<-WMAPE.DL1<-NULL

WMAPE.NN<-WMAPE.NN1<-WMAPE.DL<-WMAPE.DL1<-matrix(rep(0,z*3),z,3)

WMAPE.gcmr.NN1<-WMAPE.gcmr.DL1<-NULL

WMAPE.gcmr.NN1<-WMAPE.gcmr.DL1<-matrix(rep(0,z*3),z,3)

WMAPE.vine.NN1<-WMAPE.vine.DL1<-NULL

WMAPE.vine.NN1<-WMAPE.vine.DL1<-matrix(rep(0,z*3),z,3)


MAD.zip<-NULL

MAD.zip<- matrix(rep(0,z*3),z,3)

MAD.NN<-MAD.NN1<-MAD.DL<-MAD.DL1<-NULL

MAD.NN<-MAD.NN1<-MAD.DL<-MAD.DL1<-matrix(rep(0,z*3),z,3)

MAD.gcmr.NN1<-MAD.gcmr.DL1<-NULL

MAD.gcmr.NN1<-MAD.gcmr.DL1<-matrix(rep(0,z*3),z,3)

MAD.vine.NN1<-MAD.vine.DL1<-NULL

MAD.vine.NN1<-MAD.vine.DL1<-matrix(rep(0,z*3),z,3)



zip.Y1_LCL <-zip.Y2_LCL <-zip.Y3_LCL <- 0
zip.Y1_UCL <- zip.Y2_UCL <- zip.Y3_UCL <- 0
zip.Y1_CL <- zip.Y2_CL <-zip.Y3_CL <- 0
zip.Y1_coverage <-zip.Y2_coverage <-zip.Y3_coverage <- 0



dl.Y1_LCL <-dl.Y2_LCL <-dl.Y3_LCL <- 0
dl.Y1_UCL <- dl.Y2_UCL <- dl.Y3_UCL <- 0
dl.Y1_CL <- dl.Y2_CL <-dl.Y3_CL <- 0
dl.Y1_coverage <-dl.Y2_coverage <-dl.Y3_coverage <- 0

nn.Y1_LCL <- nn.Y2_LCL <- nn.Y3_LCL <- 0
nn.Y1_UCL <- nn.Y2_UCL <- nn.Y3_UCL <- 0
nn.Y1_CL <- nn.Y2_CL <-nn.Y3_CL <- 0
nn.Y1_coverage <-nn.Y2_coverage <-nn.Y3_coverage <- 0

dl1.Y1_LCL <-dl1.Y2_LCL <-dl1.Y3_LCL <- 0
dl1.Y1_UCL <- dl1.Y2_UCL <- dl1.Y3_UCL <- 0
dl1.Y1_CL <- dl1.Y2_CL <-dl1.Y3_CL <- 0
dl1.Y1_coverage <-dl1.Y2_coverage <-dl1.Y3_coverage <- 0

nn1.Y1_LCL <- nn1.Y2_LCL <- nn1.Y3_LCL <- 0
nn1.Y1_UCL <- nn1.Y2_UCL <- nn1.Y3_UCL <- 0
nn1.Y1_CL <- nn1.Y2_CL <-nn1.Y3_CL <- 0
nn1.Y1_coverage <-nn1.Y2_coverage <-nn1.Y3_coverage <- 0



gcdl1.Y1_LCL <-gcdl1.Y2_LCL <-gcdl1.Y3_LCL <- 0
gcdl1.Y1_UCL <- gcdl1.Y2_UCL <- gcdl1.Y3_UCL <- 0
gcdl1.Y1_CL <- gcdl1.Y2_CL <-gcdl1.Y3_CL <- 0
gcdl1.Y1_coverage <-gcdl1.Y2_coverage <-gcdl1.Y3_coverage <- 0

gcnn1.Y1_LCL <- gcnn1.Y2_LCL <- gcnn1.Y3_LCL <- 0
gcnn1.Y1_UCL <- gcnn1.Y2_UCL <- gcnn1.Y3_UCL <- 0
gcnn1.Y1_CL <- gcnn1.Y2_CL <-gcnn1.Y3_CL <- 0
gcnn1.Y1_coverage <-gcnn1.Y2_coverage <-gcnn1.Y3_coverage <- 0


vinedl1.Y1_LCL <-vinedl1.Y2_LCL <-vinedl1.Y3_LCL <- 0
vinedl1.Y1_UCL <- vinedl1.Y2_UCL <- vinedl1.Y3_UCL <- 0
vinedl1.Y1_CL <- vinedl1.Y2_CL <-vinedl1.Y3_CL <- 0
vinedl1.Y1_coverage <-vinedl1.Y2_coverage <-vinedl1.Y3_coverage <- 0

vinenn1.Y1_LCL <- vinenn1.Y2_LCL <- vinenn1.Y3_LCL <- 0
vinenn1.Y1_UCL <- vinenn1.Y2_UCL <- vinenn1.Y3_UCL <- 0
vinenn1.Y1_CL <- vinenn1.Y2_CL <-vinenn1.Y3_CL <- 0
vinenn1.Y1_coverage <-vinenn1.Y2_coverage <-vinenn1.Y3_coverage <- 0


zip.Y1_RL0<-zip.Y2_RL0<-zip.Y3_RL0<- NULL

dl.Y1_RL0<-nn.Y1_RL0 <- NULL
dl.Y2_RL0<-nn.Y2_RL0 <- NULL
dl.Y3_RL0<-nn.Y3_RL0 <- NULL

dl1.Y1_RL0<-nn1.Y1_RL0 <- NULL
dl1.Y2_RL0<-nn1.Y2_RL0 <- NULL
dl1.Y3_RL0<-nn1.Y3_RL0 <- NULL

gcdl1.Y1_RL0<-gcnn1.Y1_RL0 <- NULL
gcdl1.Y2_RL0<-gcnn1.Y2_RL0 <- NULL
gcdl1.Y3_RL0<-gcnn1.Y3_RL0 <- NULL

vinedl1.Y1_RL0<-vinenn1.Y1_RL0 <- NULL
vinedl1.Y2_RL0<-vinenn1.Y2_RL0 <- NULL
vinedl1.Y3_RL0<-vinenn1.Y3_RL0 <- NULL


zip.ARL.0.Y1<-zip.ARL.0.Y2<-zip.ARL.0.Y3 <- NULL
zip.ARL.0.Y1<-zip.ARL.0.Y2<-zip.ARL.0.Y3 <- rep(0,z)

zip.Y1_ARL.0a<-zip.Y2_ARL.0a<-zip.Y3_ARL.0a<- NULL
zip.Y1_ARL.0a<-zip.Y2_ARL.0a<-zip.Y3_ARL.0a<- rep(0,z)


NN.ARL.0.Y1<-DL.ARL.0.Y1<-NN1.ARL.0.Y1<-DL1.ARL.0.Y1 <- NULL
NN.ARL.0.Y2<-DL.ARL.0.Y2<-NN1.ARL.0.Y2<-DL1.ARL.0.Y2 <- NULL
NN.ARL.0.Y3<-DL.ARL.0.Y3<-NN1.ARL.0.Y3<-DL1.ARL.0.Y3 <- NULL

gcNN1.ARL.0.Y1<-gcDL1.ARL.0.Y1 <- NULL
gcNN1.ARL.0.Y2<-gcDL1.ARL.0.Y2 <- NULL
gcNN1.ARL.0.Y3<-gcDL1.ARL.0.Y3 <- NULL

vineNN1.ARL.0.Y1<-vineDL1.ARL.0.Y1 <- NULL
vineNN1.ARL.0.Y2<-vineDL1.ARL.0.Y2 <- NULL
vineNN1.ARL.0.Y3<-vineDL1.ARL.0.Y3 <- NULL

NN.ARL.0.Y1<-DL.ARL.0.Y1<-NN1.ARL.0.Y1<-DL1.ARL.0.Y1 <- rep(0,z)
NN.ARL.0.Y2<-DL.ARL.0.Y2<-NN1.ARL.0.Y2<-DL1.ARL.0.Y2 <- rep(0,z)
NN.ARL.0.Y3<-DL.ARL.0.Y3<-NN1.ARL.0.Y3<-DL1.ARL.0.Y3 <- rep(0,z)

gcNN1.ARL.0.Y1<-gcDL1.ARL.0.Y1 <- rep(0,z)
gcNN1.ARL.0.Y2<-gcDL1.ARL.0.Y2 <- rep(0,z)
gcNN1.ARL.0.Y3<-gcDL1.ARL.0.Y3 <- rep(0,z)

vineNN1.ARL.0.Y1<-vineDL1.ARL.0.Y1 <- rep(0,z)
vineNN1.ARL.0.Y2<-vineDL1.ARL.0.Y2 <- rep(0,z)
vineNN1.ARL.0.Y3<-vineDL1.ARL.0.Y3 <- rep(0,z)


NN.Y1_ARL.0a<-NN.Y2_ARL.0a<-NN.Y3_ARL.0a<- NULL
DL.Y1_ARL.0a<-DL.Y2_ARL.0a<-DL.Y3_ARL.0a<- NULL
NN1.Y1_ARL.0a<-NN1.Y2_ARL.0a<-NN1.Y3_ARL.0a<- NULL
DL1.Y1_ARL.0a<-DL1.Y2_ARL.0a<-DL1.Y3_ARL.0a<- NULL

gcNN1.Y1_ARL.0a<-gcNN1.Y2_ARL.0a<-gcNN1.Y3_ARL.0a<- NULL
gcDL1.Y1_ARL.0a<-gcDL1.Y2_ARL.0a<-gcDL1.Y3_ARL.0a<- NULL

vineNN1.Y1_ARL.0a<-vineNN1.Y2_ARL.0a<-vineNN1.Y3_ARL.0a<- NULL
vineDL1.Y1_ARL.0a<-vineDL1.Y2_ARL.0a<-vineDL1.Y3_ARL.0a<- NULL

NN.Y1_ARL.0a<-NN.Y2_ARL.0a<-NN.Y3_ARL.0a<- rep(0,z)
DL.Y1_ARL.0a<-DL.Y2_ARL.0a<-DL.Y3_ARL.0a<- rep(0,z)
NN1.Y1_ARL.0a<-NN1.Y2_ARL.0a<-NN1.Y3_ARL.0a<- rep(0,z)
DL1.Y1_ARL.0a<-DL1.Y2_ARL.0a<-DL1.Y3_ARL.0a<- rep(0,z)

gcNN1.Y1_ARL.0a<-gcNN1.Y2_ARL.0a<-gcNN1.Y3_ARL.0a<- rep(0,z)
gcDL1.Y1_ARL.0a<-gcDL1.Y2_ARL.0a<-gcDL1.Y3_ARL.0a<- rep(0,z)

vineNN1.Y1_ARL.0a<-vineNN1.Y2_ARL.0a<-vineNN1.Y3_ARL.0a<- rep(0,z)
vineDL1.Y1_ARL.0a<-vineDL1.Y2_ARL.0a<-vineDL1.Y3_ARL.0a<- rep(0,z)

for (q in 1:z){

set.seed(q)
  
  clayton <- claytonCopula(theta, dim=p)
  sim.data <- as.matrix(rCopula(n, clayton))
  
#  normal<-normalCopula(0.8, dim = p)
#  sim.data <- as.matrix(rCopula(n, normal))
 # cor(sim.data)

  
  M <- matrix(rep(0,n*p), nrow = n, ncol=p)
  for (i in 1:1000) {
    for (j in 1:5) {
      if(sim.data[i,j] < 0.5) M[i,j]<-0 else M[i,j]<-1 
    }
    for (k in 6:8) {
      
      if (sim.data[i,k] <= 0.5) {
        M[i,k]<-0  
      } 
      else if (sim.data[i,k] > 0.5  & sim.data[i,k] <= 0.8) {
        M[i,k]<-1
      } else {
        M[i,k]<-2  
      }
    }
    for (m in 9:13) {
      
      if (sim.data[i,m] <= 0.5) {
        M[i,m]<-0  
      } 
      else if (sim.data[i,m] > 0.5  & sim.data[i,m] <= 0.8) {
        M[i,m]<-1
      } else if (sim.data[i,m] > 0.8  & sim.data[i,m] <= 0.9) {
        M[i,m]<-2
      } else {
        M[i,m]<-3  
      }
    }
  }

  
  data1 <- data.frame(M)
  
  
  colnames(data1)<-c("X1", "X2","X3","X4","X5","X6","X7",
                     "X8", "X9","X10", "Y1","Y2","Y3")
  
  library(vioplot)
  vioplot(data1[,c(11:13)], ylab='Frequency', col="blue")
#  title("Clayton Copula Simulated Multivariate Count Data")
  title("Gaussian Copula Simulated Multivariate Count Data")
  
  
  normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }
  
  sdata <- as.data.frame(lapply(data1, normalize))
  
  
  colnames(sdata)<-c("X1", "X2","X3","X4","X5","X6","X7",
                    "X8", "X9","X10", "Y1","Y2","Y3")
  
  
  
  
  samplesize = 0.80 * nrow(sdata)
                             
  index = sample(1:nrow(sdata),samplesize )
  
  pca <- prcomp(data1[,-(11:13)], scale = F)
  #pca <- princomp(data[,-1], cor = F, scores = TRUE)
  y<-data1[,c(11:13)]
  data.pca <- data.frame(y, pca$x)	
  
  
  train.pca = data.pca[index,]
  test.pca = data.pca[-index,]
  data.pca = data.pca[-index,]
  
  train.pca.Y1<- train.pca[,-c(2,3)]
  train.pca.Y2<- train.pca[,-c(1,3)]
  train.pca.Y3<- train.pca[,-c(1,2)]
  
  x_tr.pca <- train.pca[,-c(1:3)]
  x_te.pca <- test.pca[,-c(1:3)]
  y_tr.Y1 <- train.pca[,1]
  y_te.Y1 <- test.pca[,1]
  
  y_tr.Y2 <- train.pca[,2]
  y_te.Y2 <- test.pca[,2]
  
  y_tr.Y3 <- train.pca[,3]
  y_te.Y3 <- test.pca[,3]
  
  ############ zlp ############
  library(pscl)
  # Fit Zero Inflated Poisson model
  zip.Y1<-zeroinfl(Y1 ~., dist = 'poisson', data=train.pca.Y1)
  predict_zero.Y1<-predict(zip.Y1, x_te.pca)
  resultzeroPOI.Y1 <- data.frame(actual = y_te.Y1, prediction = predict_zero.Y1)
  predictedzeroPOI.Y1=resultzeroPOI.Y1$prediction
  actualzeroPOI.Y1=resultzeroPOI.Y1$actual
  
  zip.Y2<-zeroinfl(Y2 ~., dist = 'poisson', data=train.pca.Y2)
  predict_zero.Y2<-predict(zip.Y2, x_te.pca)
  resultzeroPOI.Y2 <- data.frame(actual = y_te.Y2, prediction = predict_zero.Y2)
  predictedzeroPOI.Y2=resultzeroPOI.Y2$prediction
  actualzeroPOI.Y2=resultzeroPOI.Y2$actual
  
  zip.Y3<-zeroinfl(Y3 ~., dist = 'poisson', data=train.pca.Y3)
  predict_zero.Y3<-predict(zip.Y3, x_te.pca)
  resultzeroPOI.Y3 <- data.frame(actual = y_te.Y3, prediction = predict_zero.Y3)
  predictedzeroPOI.Y3=resultzeroPOI.Y3$prediction
  actualzeroPOI.Y3=resultzeroPOI.Y3$actual
  
  
  RMSE.zip[q,1]<- sqrt((sum(actualzeroPOI.Y1-predictedzeroPOI.Y1)^2)/length(actualzeroPOI.Y1))
  RMSE.zip[q,2]<- sqrt((sum(actualzeroPOI.Y2-predictedzeroPOI.Y2)^2)/length(actualzeroPOI.Y2))
  RMSE.zip[q,3]<- sqrt((sum(actualzeroPOI.Y3-predictedzeroPOI.Y3)^2)/length(actualzeroPOI.Y3))
  

  WMAPE.zip[q,1] <- sum(abs(actualzeroPOI.Y1-predictedzeroPOI.Y1))/sum(actualzeroPOI.Y1)
  WMAPE.zip[q,2] <- sum(abs(actualzeroPOI.Y2-predictedzeroPOI.Y2))/sum(actualzeroPOI.Y2)
  WMAPE.zip[q,3] <- sum(abs(actualzeroPOI.Y3-predictedzeroPOI.Y3))/sum(actualzeroPOI.Y3)
  
  MAD.zip[q,1] <- mean(abs(actualzeroPOI.Y1-predictedzeroPOI.Y1))
  MAD.zip[q,2] <- mean(abs(actualzeroPOI.Y2-predictedzeroPOI.Y2))
  MAD.zip[q,3] <- mean(abs(actualzeroPOI.Y3-predictedzeroPOI.Y3))
  
  zip.Y1_r <- actualzeroPOI.Y1-predictedzeroPOI.Y1
  zip.Y1_CL[q] <- mean(zip.Y1_r)
  zip.Y1_LCL[q] <- mean(zip.Y1_r) - 3*sd(zip.Y1_r)
  zip.Y1_UCL[q] <- mean(zip.Y1_r) + 3*sd(zip.Y1_r)
  zip.Y1_coverage[q] <- mean(zip.Y1_r > zip.Y1_LCL[q] & zip.Y1_r < zip.Y1_UCL[q])		
  zip.Y1_check0 <- which(zip.Y1_r < zip.Y1_LCL[q] | zip.Y1_r > zip.Y1_UCL[q])
  if(length(zip.Y1_check0)!=0L)	zip.Y1_RL0 <- c(zip.Y1_RL0, min(zip.Y1_check0))
  
  zip.Y2_r <- actualzeroPOI.Y2-predictedzeroPOI.Y2
  zip.Y2_CL[q] <- mean(zip.Y2_r)
  zip.Y2_LCL[q] <- mean(zip.Y2_r) - 3*sd(zip.Y2_r)
  zip.Y2_UCL[q] <- mean(zip.Y2_r) + 3*sd(zip.Y2_r)
  zip.Y2_coverage[q] <- mean(zip.Y2_r > zip.Y2_LCL[q] & zip.Y2_r < zip.Y2_UCL[q])		
  zip.Y2_check0 <- which(zip.Y2_r < zip.Y2_LCL[q] | zip.Y2_r > zip.Y2_UCL[q])
  if(length(zip.Y2_check0)!=0L)	zip.Y2_RL0 <- c(zip.Y2_RL0, min(zip.Y2_check0))

  zip.Y3_r <- actualzeroPOI.Y3-predictedzeroPOI.Y3
  zip.Y3_CL[q] <- mean(zip.Y3_r)
  zip.Y3_LCL[q] <- mean(zip.Y3_r) - 3*sd(zip.Y3_r)
  zip.Y3_UCL[q] <- mean(zip.Y3_r) + 3*sd(zip.Y3_r)
  zip.Y3_coverage[q] <- mean(zip.Y3_r > zip.Y3_LCL[q] & zip.Y3_r < zip.Y3_UCL[q])		
  zip.Y3_check0 <- which(zip.Y3_r < zip.Y3_LCL[q] | zip.Y3_r > zip.Y3_UCL[q])
  if(length(zip.Y3_check0)!=0L)	zip.Y3_RL0 <- c(zip.Y3_RL0, min(zip.Y3_check0))
  
  
  #### NN and DL Parts ###
                             
  trainNN = sdata[index,]
  testNN = sdata[-index,]
  datatest = sdata[-index,]
                             
                             
  x_tr <- trainNN[,-(11:13)]
  y_tr <- trainNN[,c(11:13)]
  x_te <- testNN[,-c(11:13)]
  y_te <- testNN[,c(11:13)]
  
  
  
  nn <- neuralnet(as.formula(Y1 + Y2 + Y3 ~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10),
                  data=trainNN, hidden=5, act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  
  nn.results <- compute(nn, testNN)
  nnresults <- data.frame(actual = y_te, prediction = nn.results$net.result)
  
  
  predictednn.1=nnresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  actualnn.Y1=nnresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  
  predictednn.2=nnresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
  actualnn.Y2=nnresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
  
  predictednn.3=nnresults$prediction.3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 
  actualnn.Y3=nnresults$actual.Y3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 
  
  
  RMSE.NN[q,1]<- sqrt((sum(actualnn.Y1-predictednn.1)^2)/length(actualnn.Y1))
  RMSE.NN[q,2]<- sqrt((sum(actualnn.Y2-predictednn.2)^2)/length(actualnn.Y2))
  RMSE.NN[q,3]<- sqrt((sum(actualnn.Y3-predictednn.3)^2)/length(actualnn.Y3))

  WMAPE.NN[q,1] <- sum(abs(actualnn.Y1-predictednn.1))/sum(actualnn.Y1)
  WMAPE.NN[q,2] <- sum(abs(actualnn.Y2-predictednn.2))/sum(actualnn.Y2)
  WMAPE.NN[q,3] <- sum(abs(actualnn.Y3-predictednn.3))/sum(actualnn.Y3)
  
  MAD.NN[q,1] <- mean(abs(actualnn.Y1-predictednn.1))
  MAD.NN[q,2] <- mean(abs(actualnn.Y2-predictednn.2))
  MAD.NN[q,3] <- mean(abs(actualnn.Y3-predictednn.3))
  
  nn.Y1_r <- actualnn.Y1-predictednn.1
  nn.Y1_CL[q] <- mean(nn.Y1_r)
  nn.Y1_LCL[q] <- mean(nn.Y1_r) - 3*sd(nn.Y1_r)
  nn.Y1_UCL[q] <- mean(nn.Y1_r) + 3*sd(nn.Y1_r)
  nn.Y1_coverage[q] <- mean(nn.Y1_r > nn.Y1_LCL[q] & nn.Y1_r < nn.Y1_UCL[q])		
  nn.Y1_check0 <- which(nn.Y1_r < nn.Y1_LCL[q] | nn.Y1_r > nn.Y1_UCL[q])
  if(length(nn.Y1_check0)!=0L)	nn.Y1_RL0 <- c(nn.Y1_RL0, min(nn.Y1_check0))
  
  nn.Y2_r <- actualnn.Y2-predictednn.2
  nn.Y2_CL[q] <- mean(nn.Y2_r)
  nn.Y2_LCL[q] <- mean(nn.Y2_r) - 3*sd(nn.Y2_r)
  nn.Y2_UCL[q] <- mean(nn.Y2_r) + 3*sd(nn.Y2_r)
  nn.Y2_coverage[q] <- mean(nn.Y2_r > nn.Y2_LCL[q] & nn.Y2_r < nn.Y2_UCL[q])		
  nn.Y2_check0 <- which(nn.Y2_r < nn.Y2_LCL[q] | nn.Y2_r > nn.Y2_UCL[q])
  if(length(nn.Y2_check0)!=0L)	nn.Y2_RL0 <- c(nn.Y2_RL0, min(nn.Y2_check0))
  
  nn.Y3_r <- actualnn.Y3-predictednn.3
  nn.Y3_CL[q] <- mean(nn.Y3_r)
  nn.Y3_LCL[q] <- mean(nn.Y3_r) - 3*sd(nn.Y3_r)
  nn.Y3_UCL[q] <- mean(nn.Y3_r) + 3*sd(nn.Y3_r)
  nn.Y3_coverage[q] <- mean(nn.Y3_r > nn.Y3_LCL[q] & nn.Y3_r < nn.Y3_UCL[q])		
  nn.Y3_check0 <- which(nn.Y3_r < nn.Y3_LCL[q] | nn.Y3_r > nn.Y3_UCL[q])
  if(length(nn.Y3_check0)!=0L)	nn.Y3_RL0 <- c(nn.Y3_RL0, min(nn.Y3_check0))
  

  dl <- neuralnet(as.formula(Y1 + Y2 + Y3 ~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10),
        data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  
  dl.results <- compute(dl, testNN)
  dlresults <- data.frame(actual = y_te, prediction = dl.results$net.result)
                             
                             
  predicteddl.1=dlresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  actualdl.Y1=dlresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
                             
  predicteddl.2=dlresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
  actualdl.Y2=dlresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
                             
  predicteddl.3=dlresults$prediction.3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 
  actualdl.Y3=dlresults$actual.Y3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 

    
  RMSE.DL[q,1]<- sqrt((sum(actualdl.Y1-predicteddl.1)^2)/length(actualdl.Y1))
  RMSE.DL[q,2]<- sqrt((sum(actualdl.Y2-predicteddl.2)^2)/length(actualdl.Y2))
  RMSE.DL[q,3]<- sqrt((sum(actualdl.Y3-predicteddl.3)^2)/length(actualdl.Y3))
  
  WMAPE.DL[q,1] <- sum(abs(actualdl.Y1-predicteddl.1))/sum(actualdl.Y1)
  WMAPE.DL[q,2] <- sum(abs(actualdl.Y2-predicteddl.2))/sum(actualdl.Y2)
  WMAPE.DL[q,3] <- sum(abs(actualdl.Y3-predicteddl.3))/sum(actualdl.Y3)
  
  MAD.DL[q,1] <- mean(abs(actualdl.Y1-predicteddl.1))
  MAD.DL[q,2] <- mean(abs(actualdl.Y2-predicteddl.2))
  MAD.DL[q,3] <- mean(abs(actualdl.Y3-predicteddl.3))
  
  dl.Y1_r <- actualdl.Y1-predicteddl.1
  dl.Y1_CL[q] <- mean(dl.Y1_r)
  dl.Y1_LCL[q] <- mean(dl.Y1_r) - 3*sd(dl.Y1_r)
  dl.Y1_UCL[q] <- mean(dl.Y1_r) + 3*sd(dl.Y1_r)
  dl.Y1_coverage[q] <- mean(dl.Y1_r > dl.Y1_LCL[q] & dl.Y1_r < dl.Y1_UCL[q])		
  dl.Y1_check0 <- which(dl.Y1_r < dl.Y1_LCL[q] | dl.Y1_r > dl.Y1_UCL[q])
  if(length(dl.Y1_check0)!=0L)	dl.Y1_RL0 <- c(dl.Y1_RL0, min(dl.Y1_check0))

  dl.Y2_r <- actualdl.Y2-predicteddl.2
  dl.Y2_CL[q] <- mean(dl.Y2_r)
  dl.Y2_LCL[q] <- mean(dl.Y2_r) - 3*sd(dl.Y2_r)
  dl.Y2_UCL[q] <- mean(dl.Y2_r) + 3*sd(dl.Y2_r)
  dl.Y2_coverage[q] <- mean(dl.Y2_r > dl.Y2_LCL[q] & dl.Y2_r < dl.Y2_UCL[q])		
  dl.Y2_check0 <- which(dl.Y2_r < dl.Y2_LCL[q] | dl.Y2_r > dl.Y2_UCL[q])
  if(length(dl.Y2_check0)!=0L)	dl.Y2_RL0 <- c(dl.Y2_RL0, min(dl.Y2_check0))
  
  dl.Y3_r <- actualdl.Y3-predicteddl.3
  dl.Y3_CL[q] <- mean(dl.Y3_r)
  dl.Y3_LCL[q] <- mean(dl.Y3_r) - 3*sd(dl.Y3_r)
  dl.Y3_UCL[q] <- mean(dl.Y3_r) + 3*sd(dl.Y3_r)
  dl.Y3_coverage[q] <- mean(dl.Y3_r > dl.Y3_LCL[q] & dl.Y3_r < dl.Y3_UCL[q])		
  dl.Y3_check0 <- which(dl.Y3_r < dl.Y3_LCL[q] | dl.Y3_r > dl.Y3_UCL[q])
  if(length(dl.Y3_check0)!=0L)	dl.Y3_RL0 <- c(dl.Y3_RL0, min(dl.Y3_check0))

  
    
  nn.Y1 <- neuralnet(Y1~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
  nn.Y1.results <- compute(nn.Y1, testNN)
  nnY1.results <- data.frame(actual = testNN$Y1, prediction = nn.Y1.results$net.result)

  nn.Y1.predicted=nnY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  nn.Y1.actual=nnY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  
  nn.Y2 <- neuralnet(Y2~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
  nn.Y2.results <- compute(nn.Y2, testNN)
  nnY2.results <- data.frame(actual = testNN$Y2, prediction = nn.Y2.results$net.result)

  nn.Y2.predicted=nnY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  nn.Y2.actual=nnY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  
  nn.Y3 <- neuralnet(Y3~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
  nn.Y3.results <- compute(nn.Y3, testNN)
  nnY3.results <- data.frame(actual = testNN$Y3, prediction = nn.Y3.results$net.result)
  
  nn.Y3.predicted=nnY3.results$prediction * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)
  nn.Y3.actual=nnY3.results$actual * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)
  
  
  RMSE.NN1[q,1]<- sqrt((sum(nn.Y1.actual-nn.Y1.predicted)^2)/length(nn.Y1.actual))
  RMSE.NN1[q,2]<- sqrt((sum(nn.Y2.actual-nn.Y2.predicted)^2)/length(nn.Y2.actual))
  RMSE.NN1[q,3]<- sqrt((sum(nn.Y3.actual-nn.Y3.predicted)^2)/length(nn.Y3.actual))
  
  WMAPE.NN1[q,1] <- sum(abs(nn.Y1.actual-nn.Y1.predicted))/sum(nn.Y1.actual)
  WMAPE.NN1[q,2] <- sum(abs(nn.Y2.actual-nn.Y2.predicted))/sum(nn.Y2.actual)
  WMAPE.NN1[q,3] <- sum(abs(nn.Y3.actual-nn.Y3.predicted))/sum(nn.Y3.actual)
  
  MAD.NN1[q,1] <- mean(abs(nn.Y1.actual-nn.Y1.predicted))
  MAD.NN1[q,2] <- mean(abs(nn.Y2.actual-nn.Y2.predicted))
  MAD.NN1[q,3] <- mean(abs(nn.Y3.actual-nn.Y3.predicted))
  
  nn1.Y1_r <- nn.Y1.actual-nn.Y1.predicted
  nn1.Y1_CL[q] <- mean(nn1.Y1_r)
  nn1.Y1_LCL[q] <- mean(nn1.Y1_r) - 3*sd(nn1.Y1_r)
  nn1.Y1_UCL[q] <- mean(nn1.Y1_r) + 3*sd(nn1.Y1_r)
  nn1.Y1_coverage[q] <- mean(nn1.Y1_r > nn1.Y1_LCL[q] & nn1.Y1_r < nn1.Y1_UCL[q])		
  nn1.Y1_check0 <- which(nn1.Y1_r < nn1.Y1_LCL[q] | nn1.Y1_r > nn1.Y1_UCL[q])
  if(length(nn1.Y1_check0)!=0L)	nn1.Y1_RL0 <- c(nn1.Y1_RL0, min(nn1.Y1_check0))
  
  nn1.Y2_r <- nn.Y2.actual-nn.Y2.predicted
  nn1.Y2_CL[q] <- mean(nn1.Y2_r)
  nn1.Y2_LCL[q] <- mean(nn1.Y2_r) - 3*sd(nn1.Y2_r)
  nn1.Y2_UCL[q] <- mean(nn1.Y2_r) + 3*sd(nn1.Y2_r)
  nn1.Y2_coverage[q] <- mean(nn1.Y2_r > nn1.Y2_LCL[q] & nn1.Y2_r < nn1.Y2_UCL[q])		
  nn1.Y2_check0 <- which(nn1.Y2_r < nn1.Y2_LCL[q] | nn1.Y2_r > nn1.Y2_UCL[q])
  if(length(nn1.Y2_check0)!=0L)	nn1.Y2_RL0 <- c(nn1.Y2_RL0, min(nn1.Y2_check0))
  
  nn1.Y3_r <- nn.Y3.actual-nn.Y3.predicted
  nn1.Y3_CL[q] <- mean(nn1.Y3_r)
  nn1.Y3_LCL[q] <- mean(nn1.Y3_r) - 3*sd(nn1.Y3_r)
  nn1.Y3_UCL[q] <- mean(nn1.Y3_r) + 3*sd(nn1.Y3_r)
  nn1.Y3_coverage[q] <- mean(nn1.Y3_r > nn1.Y3_LCL[q] & nn1.Y3_r < nn1.Y3_UCL[q])		
  nn1.Y3_check0 <- which(nn1.Y3_r < nn1.Y3_LCL[q] | nn1.Y3_r > nn1.Y3_UCL[q])
  if(length(nn1.Y3_check0)!=0L)	nn1.Y3_RL0 <- c(nn1.Y3_RL0, min(nn1.Y3_check0))
  
  
  dl.Y1 <- neuralnet(Y1~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  dl.Y1.results <- compute(dl.Y1, testNN)
  dlY1.results <- data.frame(actual = testNN$Y1, prediction = dl.Y1.results$net.result)
  
  dl.Y1.predicted=dlY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  dl.Y1.actual=dlY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  
  dl.Y2 <- neuralnet(Y2~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  dl.Y2.results <- compute(dl.Y2, testNN)
  dlY2.results <- data.frame(actual = testNN$Y2, prediction = dl.Y2.results$net.result)
  
  dl.Y2.predicted=dlY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  dl.Y2.actual=dlY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  
  
  dl.Y3 <- neuralnet(Y3~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  dl.Y3.results <- compute(dl.Y3, testNN)
  dlY3.results <- data.frame(actual = testNN$Y3, prediction = dl.Y3.results$net.result)
  
  dl.Y3.predicted=dlY3.results$prediction * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)
  dl.Y3.actual=dlY3.results$actual * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)
  
  
  RMSE.DL1[q,1]<- sqrt((sum(dl.Y1.actual-dl.Y1.predicted)^2)/length(dl.Y1.actual))
  RMSE.DL1[q,2]<- sqrt((sum(dl.Y2.actual-dl.Y2.predicted)^2)/length(dl.Y2.actual))
  RMSE.DL1[q,3]<- sqrt((sum(dl.Y3.actual-dl.Y3.predicted)^2)/length(dl.Y3.actual))

  WMAPE.DL1[q,1] <- sum(abs(dl.Y1.actual-dl.Y1.predicted))/sum(dl.Y1.actual)
  WMAPE.DL1[q,2] <- sum(abs(dl.Y2.actual-dl.Y2.predicted))/sum(dl.Y2.actual)
  WMAPE.DL1[q,3] <- sum(abs(dl.Y3.actual-dl.Y3.predicted))/sum(dl.Y3.actual)
  
  MAD.DL1[q,1] <- mean(abs(dl.Y1.actual-dl.Y1.predicted))
  MAD.DL1[q,2] <- mean(abs(dl.Y2.actual-dl.Y2.predicted))
  MAD.DL1[q,3] <- mean(abs(dl.Y3.actual-dl.Y3.predicted))
  
  dl1.Y1_r <- dl.Y1.actual-dl.Y1.predicted
  dl1.Y1_CL[q] <- mean(dl1.Y1_r)
  dl1.Y1_LCL[q] <- mean(dl1.Y1_r) - 3*sd(dl1.Y1_r)
  dl1.Y1_UCL[q] <- mean(dl1.Y1_r) + 3*sd(dl1.Y1_r)
  dl1.Y1_coverage[q] <- mean(dl1.Y1_r > dl1.Y1_LCL[q] & dl1.Y1_r < dl1.Y1_UCL[q])		
  dl1.Y1_check0 <- which(dl1.Y1_r < dl1.Y1_LCL[q] | dl1.Y1_r > dl1.Y1_UCL[q])
  if(length(dl1.Y1_check0)!=0L)	dl1.Y1_RL0 <- c(dl1.Y1_RL0, min(dl1.Y1_check0))
  
  dl1.Y2_r <- dl.Y2.actual-dl.Y2.predicted
  dl1.Y2_CL[q] <- mean(dl1.Y2_r)
  dl1.Y2_LCL[q] <- mean(dl1.Y2_r) - 3*sd(dl1.Y2_r)
  dl1.Y2_UCL[q] <- mean(dl1.Y2_r) + 3*sd(dl1.Y2_r)
  dl1.Y2_coverage[q] <- mean(dl1.Y2_r > dl1.Y2_LCL[q] & dl1.Y2_r < dl1.Y2_UCL[q])		
  dl1.Y2_check0 <- which(dl1.Y2_r < dl1.Y2_LCL[q] | dl1.Y2_r > dl1.Y2_UCL[q])
  if(length(dl1.Y2_check0)!=0L)	dl1.Y2_RL0 <- c(dl1.Y2_RL0, min(dl1.Y2_check0))
  
  dl1.Y3_r <- dl.Y3.actual-dl.Y3.predicted
  dl1.Y3_CL[q] <- mean(dl1.Y3_r)
  dl1.Y3_LCL[q] <- mean(dl1.Y3_r) - 3*sd(dl1.Y3_r)
  dl1.Y3_UCL[q] <- mean(dl1.Y3_r) + 3*sd(dl1.Y3_r)
  dl1.Y3_coverage[q] <- mean(dl1.Y3_r > dl1.Y3_LCL[q] & dl1.Y3_r < dl1.Y3_UCL[q])		
  dl1.Y3_check0 <- which(dl1.Y3_r < dl1.Y3_LCL[q] | dl1.Y3_r > dl1.Y3_UCL[q])
  if(length(dl1.Y3_check0)!=0L)	dl1.Y3_RL0 <- c(dl1.Y3_RL0, min(dl1.Y3_check0))
  

  
  
  # The data:----
  
  nn1_data<-cbind(nn.Y1.predicted, nn.Y2.predicted, nn.Y3.predicted)
  y1 = nn1_data[,1]
  y2 = nn1_data[,2]
  y3 = nn1_data[,3]
  datann<-data.frame(cbind(y1, y2, y3))
  
  library(gcmr)
  
  y1.23.00<-gcmr( y1~y2+y3, data = datann, marginal = gaussian.marg(link="identity"),
                  cormat = arma.cormat(0, 0) )
  
  y2.13.00<-gcmr( y2~y1+y3, data = datann, marginal = gaussian.marg(link="identity"),
                  cormat = arma.cormat(0, 0) )
  
  y3.12.00<-gcmr( y3~y1+y2, data = datann, marginal = gaussian.marg(link="identity"),
                  cormat = arma.cormat(0, 0) )
  

#  summary(y1.23.00)
  predictions1nn <-y1.23.00$estimate[1]+
    y1.23.00$estimate[2]*datann$y2+y1.23.00$estimate[3]*datann$y3
  
  
#  summary(y2.13.00)
  predictions2nn <-y2.13.00$estimate[1]+
    y2.13.00$estimate[2]*datann$y1+y2.13.00$estimate[3]*datann$y3
  
  
#  summary(y3.12.00)
  predictions3nn <-y3.12.00$estimate[1]+
    y3.12.00$estimate[2]*datann$y1+y3.12.00$estimate[3]*datann$y2
  
  
  RMSE.gcmr.NN1[q,1]<- sqrt(mean((nn.Y1.actual - predictions1nn)^2))
  RMSE.gcmr.NN1[q,2]<- sqrt(mean((nn.Y2.actual - predictions2nn)^2))
  RMSE.gcmr.NN1[q,3]<- sqrt(mean((nn.Y3.actual - predictions3nn)^2))
  
  WMAPE.gcmr.NN1[q,1] <- sum(abs(nn.Y1.actual - predictions1nn))/sum(nn.Y1.actual)
  WMAPE.gcmr.NN1[q,2] <- sum(abs(nn.Y2.actual - predictions2nn))/sum(nn.Y2.actual)
  WMAPE.gcmr.NN1[q,3] <- sum(abs(nn.Y3.actual - predictions3nn))/sum(nn.Y3.actual)
  
  MAD.gcmr.NN1[q,1] <- mean(abs(nn.Y1.actual - predictions1nn))
  MAD.gcmr.NN1[q,2] <- mean(abs(nn.Y2.actual - predictions2nn))
  MAD.gcmr.NN1[q,3] <- mean(abs(nn.Y3.actual - predictions3nn))
  
  gcnn1.Y1_r <- nn.Y1.actual-predictions1nn
  gcnn1.Y1_CL[q] <- mean(gcnn1.Y1_r)
  gcnn1.Y1_LCL[q] <- mean(gcnn1.Y1_r) - 3*sd(gcnn1.Y1_r)
  gcnn1.Y1_UCL[q] <- mean(gcnn1.Y1_r) + 3*sd(gcnn1.Y1_r)
  gcnn1.Y1_coverage[q] <- mean(gcnn1.Y1_r > gcnn1.Y1_LCL[q] & gcnn1.Y1_r < gcnn1.Y1_UCL[q])		
  gcnn1.Y1_check0 <- which(gcnn1.Y1_r < gcnn1.Y1_LCL[q] | gcnn1.Y1_r > gcnn1.Y1_UCL[q])
  if(length(gcnn1.Y1_check0)!=0L)	gcnn1.Y1_RL0 <- c(gcnn1.Y1_RL0, min(gcnn1.Y1_check0))
  
  gcnn1.Y2_r <- nn.Y2.actual-predictions2nn
  gcnn1.Y2_CL[q] <- mean(gcnn1.Y2_r)
  gcnn1.Y2_LCL[q] <- mean(gcnn1.Y2_r) - 3*sd(gcnn1.Y2_r)
  gcnn1.Y2_UCL[q] <- mean(gcnn1.Y2_r) + 3*sd(gcnn1.Y2_r)
  gcnn1.Y2_coverage[q] <- mean(gcnn1.Y2_r > gcnn1.Y2_LCL[q] & gcnn1.Y2_r < gcnn1.Y2_UCL[q])		
  gcnn1.Y2_check0 <- which(gcnn1.Y2_r < gcnn1.Y2_LCL[q] | gcnn1.Y2_r > gcnn1.Y2_UCL[q])
  if(length(gcnn1.Y2_check0)!=0L)	gcnn1.Y2_RL0 <- c(gcnn1.Y2_RL0, min(gcnn1.Y2_check0))
  
  gcnn1.Y3_r <- nn.Y3.actual-predictions3nn
  gcnn1.Y3_CL[q] <- mean(gcnn1.Y3_r)
  gcnn1.Y3_LCL[q] <- mean(gcnn1.Y3_r) - 3*sd(gcnn1.Y3_r)
  gcnn1.Y3_UCL[q] <- mean(gcnn1.Y3_r) + 3*sd(gcnn1.Y3_r)
  gcnn1.Y3_coverage[q] <- mean(gcnn1.Y3_r > gcnn1.Y3_LCL[q] & gcnn1.Y3_r < gcnn1.Y3_UCL[q])		
  gcnn1.Y3_check0 <- which(gcnn1.Y3_r < gcnn1.Y3_LCL[q] | gcnn1.Y3_r > gcnn1.Y3_UCL[q])
  if(length(gcnn1.Y3_check0)!=0L)	gcnn1.Y3_RL0 <- c(gcnn1.Y3_RL0, min(gcnn1.Y3_check0))
  
  # fit vine regression model 
  
  library(vinereg)
  
  vy1.23.00<-vinereg( y1~y2+y3, data = datann)
  vy2.13.00<-vinereg( y2~y1+y3, data = datann)
  vy3.12.00<-vinereg( y3~y1+y2, data = datann)
  
  vinepredictions1nn <-predict(vy1.23.00, newdata = datann, alpha = NA)
  vinepredictions2nn <-predict(vy2.13.00, newdata = datann, alpha = NA)
  vinepredictions3nn <-predict(vy3.12.00, newdata = datann, alpha = NA)
  
  
  RMSE.vine.NN1[q,1]<- sqrt(mean((nn.Y1.actual - vinepredictions1nn$mean)^2))
  RMSE.vine.NN1[q,2]<- sqrt(mean((nn.Y2.actual - vinepredictions2nn$mean)^2))
  RMSE.vine.NN1[q,3]<- sqrt(mean((nn.Y3.actual - vinepredictions3nn$mean)^2))
  
  WMAPE.vine.NN1[q,1] <- sum(abs(nn.Y1.actual - vinepredictions1nn$mean))/sum(nn.Y1.actual)
  WMAPE.vine.NN1[q,2] <- sum(abs(nn.Y2.actual - vinepredictions2nn$mean))/sum(nn.Y2.actual)
  WMAPE.vine.NN1[q,3] <- sum(abs(nn.Y3.actual - vinepredictions3nn$mean))/sum(nn.Y3.actual)
  
  MAD.vine.NN1[q,1] <- mean(abs(nn.Y1.actual - vinepredictions1nn$mean))
  MAD.vine.NN1[q,2] <- mean(abs(nn.Y2.actual - vinepredictions2nn$mean))
  MAD.vine.NN1[q,3] <- mean(abs(nn.Y3.actual - vinepredictions3nn$mean))
  

  vinenn1.Y1_r <- nn.Y1.actual-vinepredictions1nn$mean
  vinenn1.Y1_CL[q] <- mean(vinenn1.Y1_r)
  vinenn1.Y1_LCL[q] <- mean(vinenn1.Y1_r) - 3*sd(vinenn1.Y1_r)
  vinenn1.Y1_UCL[q] <- mean(vinenn1.Y1_r) + 3*sd(vinenn1.Y1_r)
  vinenn1.Y1_coverage[q] <- mean(vinenn1.Y1_r > vinenn1.Y1_LCL[q] & vinenn1.Y1_r < vinenn1.Y1_UCL[q])		
  vinenn1.Y1_check0 <- which(vinenn1.Y1_r < vinenn1.Y1_LCL[q] | vinenn1.Y1_r > vinenn1.Y1_UCL[q])
  if(length(vinenn1.Y1_check0)!=0L)	vinenn1.Y1_RL0 <- c(vinenn1.Y1_RL0, min(vinenn1.Y1_check0))
  
  vinenn1.Y2_r <- nn.Y2.actual-vinepredictions2nn$mean
  vinenn1.Y2_CL[q] <- mean(vinenn1.Y2_r)
  vinenn1.Y2_LCL[q] <- mean(vinenn1.Y2_r) - 3*sd(vinenn1.Y2_r)
  vinenn1.Y2_UCL[q] <- mean(vinenn1.Y2_r) + 3*sd(vinenn1.Y2_r)
  vinenn1.Y2_coverage[q] <- mean(vinenn1.Y2_r > vinenn1.Y2_LCL[q] & vinenn1.Y2_r < vinenn1.Y2_UCL[q])		
  vinenn1.Y2_check0 <- which(vinenn1.Y2_r < vinenn1.Y2_LCL[q] | vinenn1.Y2_r > vinenn1.Y2_UCL[q])
  if(length(vinenn1.Y2_check0)!=0L)	vinenn1.Y2_RL0 <- c(vinenn1.Y2_RL0, min(vinenn1.Y2_check0))
  
  vinenn1.Y3_r <- nn.Y3.actual-vinepredictions3nn$mean
  vinenn1.Y3_CL[q] <- mean(vinenn1.Y3_r)
  vinenn1.Y3_LCL[q] <- mean(vinenn1.Y3_r) - 3*sd(vinenn1.Y3_r)
  vinenn1.Y3_UCL[q] <- mean(vinenn1.Y3_r) + 3*sd(vinenn1.Y3_r)
  vinenn1.Y3_coverage[q] <- mean(vinenn1.Y3_r > vinenn1.Y3_LCL[q] & vinenn1.Y3_r < vinenn1.Y3_UCL[q])		
  vinenn1.Y3_check0 <- which(vinenn1.Y3_r < vinenn1.Y3_LCL[q] | vinenn1.Y3_r > vinenn1.Y3_UCL[q])
  if(length(vinenn1.Y3_check0)!=0L)	vinenn1.Y3_RL0 <- c(vinenn1.Y3_RL0, min(vinenn1.Y3_check0))
  
  
  
  dl1_data<-cbind(dl.Y1.predicted, dl.Y2.predicted, dl.Y3.predicted)
  y1 = dl1_data[,1]
  y2 = dl1_data[,2]
  y3 = dl1_data[,3]
  data<-data.frame(cbind(y1, y2, y3))
  
  library(gcmr)
  
  y1.23.00<-gcmr( y1~y2+y3, data = data, marginal = gaussian.marg(link="identity"),
                  cormat = arma.cormat(0, 0) )
  
  y2.13.00<-gcmr( y2~y1+y3, data = data, marginal = gaussian.marg(link="identity"),
                  cormat = arma.cormat(0, 0) )
  
  y3.12.00<-gcmr( y3~y1+y2, data = data, marginal = gaussian.marg(link="identity"),
                  cormat = arma.cormat(0, 0) )

#  summary(y1.23.00)
  predictions1 <-y1.23.00$estimate[1]+
    y1.23.00$estimate[2]*data$y2+y1.23.00$estimate[3]*data$y3
  
#  summary(y2.13.00)
  predictions2 <-y2.13.00$estimate[1]+
    y2.13.00$estimate[2]*data$y1+y2.13.00$estimate[3]*data$y3
  
#  summary(y3.12.00)
  predictions3 <-y3.12.00$estimate[1]+
    y3.12.00$estimate[2]*data$y1+y3.12.00$estimate[3]*data$y2
  
  
  RMSE.gcmr.DL1[q,1]<- sqrt(mean((dl.Y1.actual - predictions1)^2))
  RMSE.gcmr.DL1[q,2]<- sqrt(mean((dl.Y2.actual - predictions2)^2))
  RMSE.gcmr.DL1[q,3]<- sqrt(mean((dl.Y3.actual - predictions3)^2))
  
  WMAPE.gcmr.DL1[q,1] <- sum(abs(dl.Y1.actual - predictions1))/sum(dl.Y1.actual)
  WMAPE.gcmr.DL1[q,2] <- sum(abs(dl.Y2.actual - predictions2))/sum(dl.Y2.actual)
  WMAPE.gcmr.DL1[q,3] <- sum(abs(dl.Y3.actual - predictions3))/sum(dl.Y3.actual)
  
  MAD.gcmr.DL1[q,1] <- mean(abs(dl.Y1.actual - predictions1))
  MAD.gcmr.DL1[q,2] <- mean(abs(dl.Y2.actual - predictions2))
  MAD.gcmr.DL1[q,3] <- mean(abs(dl.Y3.actual - predictions3))
  
  
  gcdl1.Y1_r <- dl.Y1.actual-predictions1
  gcdl1.Y1_CL[q] <- mean(gcdl1.Y1_r)
  gcdl1.Y1_LCL[q] <- mean(gcdl1.Y1_r) - 3*sd(gcdl1.Y1_r)
  gcdl1.Y1_UCL[q] <- mean(gcdl1.Y1_r) + 3*sd(gcdl1.Y1_r)
  gcdl1.Y1_coverage[q] <- mean(gcdl1.Y1_r > gcdl1.Y1_LCL[q] & gcdl1.Y1_r < gcdl1.Y1_UCL[q])		
  gcdl1.Y1_check0 <- which(gcdl1.Y1_r < gcdl1.Y1_LCL[q] | gcdl1.Y1_r > gcdl1.Y1_UCL[q])
  if(length(gcdl1.Y1_check0)!=0L)	gcdl1.Y1_RL0 <- c(gcdl1.Y1_RL0, min(gcdl1.Y1_check0))
  
  gcdl1.Y2_r <- dl.Y2.actual-predictions2
  gcdl1.Y2_CL[q] <- mean(gcdl1.Y2_r)
  gcdl1.Y2_LCL[q] <- mean(gcdl1.Y2_r) - 3*sd(gcdl1.Y2_r)
  gcdl1.Y2_UCL[q] <- mean(gcdl1.Y2_r) + 3*sd(gcdl1.Y2_r)
  gcdl1.Y2_coverage[q] <- mean(gcdl1.Y2_r > gcdl1.Y2_LCL[q] & gcdl1.Y2_r < gcdl1.Y2_UCL[q])		
  gcdl1.Y2_check0 <- which(gcdl1.Y2_r < gcdl1.Y2_LCL[q] | gcdl1.Y2_r > gcdl1.Y2_UCL[q])
  if(length(gcdl1.Y2_check0)!=0L)	gcdl1.Y2_RL0 <- c(gcdl1.Y2_RL0, min(gcdl1.Y2_check0))
  
  gcdl1.Y3_r <- dl.Y3.actual-predictions3
  gcdl1.Y3_CL[q] <- mean(gcdl1.Y3_r)
  gcdl1.Y3_LCL[q] <- mean(gcdl1.Y3_r) - 3*sd(gcdl1.Y3_r)
  gcdl1.Y3_UCL[q] <- mean(gcdl1.Y3_r) + 3*sd(gcdl1.Y3_r)
  gcdl1.Y3_coverage[q] <- mean(gcdl1.Y3_r > gcdl1.Y3_LCL[q] & gcdl1.Y3_r < gcdl1.Y3_UCL[q])		
  gcdl1.Y3_check0 <- which(gcdl1.Y3_r < gcdl1.Y3_LCL[q] | gcdl1.Y3_r > gcdl1.Y3_UCL[q])
  if(length(gcdl1.Y3_check0)!=0L)	gcdl1.Y3_RL0 <- c(gcdl1.Y3_RL0, min(gcdl1.Y3_check0))
  
  
  # fit vine regression model 
  
  library(vinereg)
  
  vy1.23.00<-vinereg( y1~y2+y3, data = data)
  vy2.13.00<-vinereg( y2~y1+y3, data = data)
  vy3.12.00<-vinereg( y3~y1+y2, data = data)
  
  vinepredictions1dl <-predict(vy1.23.00, newdata = data, alpha = NA)
  vinepredictions2dl <-predict(vy2.13.00, newdata = data, alpha = NA)
  vinepredictions3dl <-predict(vy3.12.00, newdata = data, alpha = NA)
  
  
  RMSE.vine.DL1[q,1]<- sqrt(mean((dl.Y1.actual - vinepredictions1dl$mean)^2))
  RMSE.vine.DL1[q,2]<- sqrt(mean((dl.Y2.actual - vinepredictions2dl$mean)^2))
  RMSE.vine.DL1[q,3]<- sqrt(mean((dl.Y3.actual - vinepredictions3dl$mean)^2))
  
  WMAPE.vine.DL1[q,1] <- sum(abs(dl.Y1.actual - vinepredictions1dl$mean))/sum(dl.Y1.actual)
  WMAPE.vine.DL1[q,2] <- sum(abs(dl.Y2.actual - vinepredictions2dl$mean))/sum(dl.Y2.actual)
  WMAPE.vine.DL1[q,3] <- sum(abs(dl.Y3.actual - vinepredictions3dl$mean))/sum(dl.Y3.actual)
  
  MAD.vine.DL1[q,1] <- mean(abs(dl.Y1.actual - vinepredictions1dl$mean))
  MAD.vine.DL1[q,2] <- mean(abs(dl.Y2.actual - vinepredictions2dl$mean))
  MAD.vine.DL1[q,3] <- mean(abs(dl.Y3.actual - vinepredictions3dl$mean))
  
  
  vinedl1.Y1_r <- dl.Y1.actual-vinepredictions1dl$mean
  vinedl1.Y1_CL[q] <- mean(vinedl1.Y1_r)
  vinedl1.Y1_LCL[q] <- mean(vinedl1.Y1_r) - 3*sd(vinedl1.Y1_r)
  vinedl1.Y1_UCL[q] <- mean(vinedl1.Y1_r) + 3*sd(vinedl1.Y1_r)
  vinedl1.Y1_coverage[q] <- mean(vinedl1.Y1_r > vinedl1.Y1_LCL[q] & vinedl1.Y1_r < vinedl1.Y1_UCL[q])		
  vinedl1.Y1_check0 <- which(vinedl1.Y1_r < vinedl1.Y1_LCL[q] | vinedl1.Y1_r > vinedl1.Y1_UCL[q])
  if(length(vinedl1.Y1_check0)!=0L)	vinedl1.Y1_RL0 <- c(vinedl1.Y1_RL0, min(vinedl1.Y1_check0))
  
  vinedl1.Y2_r <- dl.Y2.actual-vinepredictions2dl$mean
  vinedl1.Y2_CL[q] <- mean(vinedl1.Y2_r)
  vinedl1.Y2_LCL[q] <- mean(vinedl1.Y2_r) - 3*sd(vinedl1.Y2_r)
  vinedl1.Y2_UCL[q] <- mean(vinedl1.Y2_r) + 3*sd(vinedl1.Y2_r)
  vinedl1.Y2_coverage[q] <- mean(vinedl1.Y2_r > vinedl1.Y2_LCL[q] & vinedl1.Y2_r < vinedl1.Y2_UCL[q])		
  vinedl1.Y2_check0 <- which(vinedl1.Y2_r < vinedl1.Y2_LCL[q] | vinedl1.Y2_r > vinedl1.Y2_UCL[q])
  if(length(vinedl1.Y2_check0)!=0L)	vinedl1.Y2_RL0 <- c(vinedl1.Y2_RL0, min(vinedl1.Y2_check0))
  
  vinedl1.Y3_r <- dl.Y3.actual-vinepredictions3dl$mean
  vinedl1.Y3_CL[q] <- mean(vinedl1.Y3_r)
  vinedl1.Y3_LCL[q] <- mean(vinedl1.Y3_r) - 3*sd(vinedl1.Y3_r)
  vinedl1.Y3_UCL[q] <- mean(vinedl1.Y3_r) + 3*sd(vinedl1.Y3_r)
  vinedl1.Y3_coverage[q] <- mean(vinedl1.Y3_r > vinedl1.Y3_LCL[q] & vinedl1.Y3_r < vinedl1.Y3_UCL[q])		
  vinedl1.Y3_check0 <- which(vinedl1.Y3_r < vinedl1.Y3_LCL[q] | vinedl1.Y3_r > vinedl1.Y3_UCL[q])
  if(length(vinedl1.Y3_check0)!=0L)	vinedl1.Y3_RL0 <- c(vinedl1.Y3_RL0, min(vinedl1.Y3_check0))
  
  
  
  print(q)
  }



x1 <- RMSE.zip[,1]
x2 <- RMSE.DL[,1]
x3 <- RMSE.NN[,1]
x4 <- RMSE.DL1[,1]
x5 <- RMSE.NN1[,1]
x6<-RMSE.gcmr.DL1[,1]
x7<-RMSE.gcmr.NN1[,1]
x8<-RMSE.vine.DL1[,1]
x9<-RMSE.vine.NN1[,1]

boxplot(x1, x2, x3, x4, x5, x6, x7, x8, x9, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "RMSE",
        col="gold")
title("RMSE with Simulated Data (Y1)")

w1 <- WMAPE.zip[,1]
w2 <- WMAPE.DL[,1]
w3 <- WMAPE.NN[,1]
w4 <- WMAPE.DL1[,1]
w5 <- WMAPE.NN1[,1]
w6<-WMAPE.gcmr.DL1[,1]
w7<-WMAPE.gcmr.NN1[,1]
w8<-WMAPE.vine.DL1[,1]
w9<-WMAPE.vine.NN1[,1]

boxplot(w1, w2, w3, w4, w5, w6, w7, w8, w9, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "WMAPE",
        col="gold")
title("WMAPE with Simulated Data (Y1)")


z1 <- MAD.zip[,1]
z2 <- MAD.DL[,1]
z3 <- MAD.NN[,1]
z4 <- MAD.DL1[,1]
z5 <- MAD.NN1[,1]
z6<-MAD.gcmr.DL1[,1]
z7<-MAD.gcmr.NN1[,1]
z8<-MAD.vine.DL1[,1]
z9<-MAD.vine.NN1[,1]

boxplot(z1, z2, z3, z4, z5, z6, z7, z8, z9, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "MAD",
        col="gold")
title("MAD with Simulated Data (Y1)")



x11 <- RMSE.zip[,2]
x12 <- RMSE.DL[,2]
x13 <- RMSE.NN[,2]
x14 <- RMSE.DL1[,2]
x15 <- RMSE.NN1[,2]
x16<-RMSE.gcmr.DL1[,2]
x17<-RMSE.gcmr.NN1[,2]
x18<-RMSE.vine.DL1[,2]
x19<-RMSE.vine.NN1[,2]

boxplot(x11, x12, x13, x14, x15, x16, x17, x18, x19, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "RMSE",
        col="gold")
title("RMSE with Simulated Data (Y2)")

w11 <- WMAPE.zip[,2]
w12 <- WMAPE.DL[,2]
w13 <- WMAPE.NN[,2]
w14 <- WMAPE.DL1[,2]
w15 <- WMAPE.NN1[,2]
w16<-WMAPE.gcmr.DL1[,2]
w17<-WMAPE.gcmr.NN1[,2]
w18<-WMAPE.vine.DL1[,2]
w19<-WMAPE.vine.NN1[,2]

boxplot(w11, w12, w13, w14, w15, w16, w17, w18, w19, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "WMAPE",
        col="gold")
title("WMAPE with Simulated Data (Y2)")


z11 <- MAD.zip[,2]
z12 <- MAD.DL[,2]
z13 <- MAD.NN[,2]
z14 <- MAD.DL1[,2]
z15 <- MAD.NN1[,2]
z16<-MAD.gcmr.DL1[,2]
z17<-MAD.gcmr.NN1[,2]
z18<-MAD.vine.DL1[,2]
z19<-MAD.vine.NN1[,2]

boxplot(z11, z12, z13, z14, z15, z16, z17, z18, z19, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "MAD",
        col="gold")
title("MAD with Simulated Data (Y2)")



x21 <- RMSE.zip[,3]
x22 <- RMSE.DL[,3]
x23 <- RMSE.NN[,3]
x24 <- RMSE.DL1[,3]
x25 <- RMSE.NN1[,3]
x26<-RMSE.gcmr.DL1[,3]
x27<-RMSE.gcmr.NN1[,3]
x28<-RMSE.vine.DL1[,3]
x29<-RMSE.vine.NN1[,3]

boxplot(x21, x22, x23, x24, x25, x26, x27, x28, x29, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "RMSE",
        col="gold")
title("RMSE with Simulated Data (Y3)")

w21 <- WMAPE.zip[,3]
w22 <- WMAPE.DL[,3]
w23 <- WMAPE.NN[,3]
w24 <- WMAPE.DL1[,3]
w25 <- WMAPE.NN1[,3]
w26<-WMAPE.gcmr.DL1[,3]
w27<-WMAPE.gcmr.NN1[,3]
w28<-WMAPE.vine.DL1[,3]
w29<-WMAPE.vine.NN1[,3]

boxplot(w21, w22, w23, w24, w25, w26, w27, w28, w29, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "WMAPE",
        col="gold")
title("WMAPE with Simulated Data (Y3)")


z21 <- MAD.zip[,3]
z22 <- MAD.DL[,3]
z23 <- MAD.NN[,3]
z24 <- MAD.DL1[,3]
z25 <- MAD.NN1[,3]
z26<-MAD.gcmr.DL1[,3]
z27<-MAD.gcmr.NN1[,3]
z28<-MAD.vine.DL1[,3]
z29<-MAD.vine.NN1[,3]

boxplot(z21, z22, z23, z24, z25, z26, z27, z28, z29, 
        names=c("ZIP","DL", "NN", "DL1", "NN1", "GCDL1", "GCNN1", "VINEDL1", "VINENN1"), ylab = "MAD",
        col="gold")
title("MAD with Simulated Data (Y3)")

summary(RMSE.zip)
IQR(RMSE.zip[,1])
IQR(RMSE.zip[,2])
IQR(RMSE.zip[,3])

summary(RMSE.DL)
IQR(RMSE.DL[,1])
IQR(RMSE.DL[,2])
IQR(RMSE.DL[,3])

summary(RMSE.NN)
IQR(RMSE.NN[,1])
IQR(RMSE.NN[,2])
IQR(RMSE.NN[,3])

summary(RMSE.DL1)
IQR(RMSE.DL1[,1])
IQR(RMSE.DL1[,2])
IQR(RMSE.DL1[,3])

summary(RMSE.NN1)
IQR(RMSE.NN1[,1])
IQR(RMSE.NN1[,2])
IQR(RMSE.NN1[,3])


summary(RMSE.gcmr.DL1)
IQR(RMSE.gcmr.DL1[,1])
IQR(RMSE.gcmr.DL1[,2])
IQR(RMSE.gcmr.DL1[,3])

summary(RMSE.gcmr.NN1)
IQR(RMSE.gcmr.NN1[,1])
IQR(RMSE.gcmr.NN1[,2])
IQR(RMSE.gcmr.NN1[,3])


summary(RMSE.vine.DL1)
IQR(RMSE.vine.DL1[,1])
IQR(RMSE.vine.DL1[,2])
IQR(RMSE.vine.DL1[,3])

summary(RMSE.vine.NN1)
IQR(RMSE.vine.NN1[,1])
IQR(RMSE.vine.NN1[,2])
IQR(RMSE.vine.NN1[,3])


summary(WMAPE.zip)
IQR(WMAPE.zip[,1])
IQR(WMAPE.zip[,2])
IQR(WMAPE.zip[,3])

summary(WMAPE.DL)
IQR(WMAPE.DL[,1])
IQR(WMAPE.DL[,2])
IQR(WMAPE.DL[,3])

summary(WMAPE.NN)
IQR(WMAPE.NN[,1])
IQR(WMAPE.NN[,2])
IQR(WMAPE.NN[,3])

summary(WMAPE.DL1)
IQR(WMAPE.DL1[,1])
IQR(WMAPE.DL1[,2])
IQR(WMAPE.DL1[,3])

summary(WMAPE.NN1)
IQR(WMAPE.NN1[,1])
IQR(WMAPE.NN1[,2])
IQR(WMAPE.NN1[,3])


summary(WMAPE.gcmr.DL1)
IQR(WMAPE.gcmr.DL1[,1])
IQR(WMAPE.gcmr.DL1[,2])
IQR(WMAPE.gcmr.DL1[,3])

summary(WMAPE.gcmr.NN1)
IQR(WMAPE.gcmr.NN1[,1])
IQR(WMAPE.gcmr.NN1[,2])
IQR(WMAPE.gcmr.NN1[,3])


summary(WMAPE.vine.DL1)
IQR(WMAPE.vine.DL1[,1])
IQR(WMAPE.vine.DL1[,2])
IQR(WMAPE.vine.DL1[,3])

summary(WMAPE.vine.NN1)
IQR(WMAPE.vine.NN1[,1])
IQR(WMAPE.vine.NN1[,2])
IQR(WMAPE.vine.NN1[,3])


summary(MAD.zip)
IQR(MAD.zip[,1])
IQR(MAD.zip[,2])
IQR(MAD.zip[,3])

summary(MAD.DL)
IQR(MAD.DL[,1])
IQR(MAD.DL[,2])
IQR(MAD.DL[,3])

summary(MAD.NN)
IQR(MAD.NN[,1])
IQR(MAD.NN[,2])
IQR(MAD.NN[,3])

summary(MAD.DL1)
IQR(MAD.DL1[,1])
IQR(MAD.DL1[,2])
IQR(MAD.DL1[,3])

summary(MAD.NN1)
IQR(MAD.NN1[,1])
IQR(MAD.NN1[,2])
IQR(MAD.NN1[,3])


summary(MAD.gcmr.DL1)
IQR(MAD.gcmr.DL1[,1])
IQR(MAD.gcmr.DL1[,2])
IQR(MAD.gcmr.DL1[,3])

summary(MAD.gcmr.NN1)
IQR(MAD.gcmr.NN1[,1])
IQR(MAD.gcmr.NN1[,2])
IQR(MAD.gcmr.NN1[,3])


summary(MAD.vine.DL1)
IQR(MAD.vine.DL1[,1])
IQR(MAD.vine.DL1[,2])
IQR(MAD.vine.DL1[,3])

summary(MAD.vine.NN1)
IQR(MAD.vine.NN1[,1])
IQR(MAD.vine.NN1[,2])
IQR(MAD.vine.NN1[,3])




#ARL0

mean(zip.Y1_LCL)
mean(zip.Y1_UCL)
mean(zip.Y1_CL)
mean(zip.Y1_coverage)

mean(zip.Y2_LCL)
mean(zip.Y2_UCL)
mean(zip.Y2_CL)
mean(zip.Y2_coverage)

mean(zip.Y3_LCL)
mean(zip.Y3_UCL)
mean(zip.Y3_CL)
mean(zip.Y3_coverage)


mean(dl.Y1_LCL)
mean(dl.Y1_UCL)
mean(dl.Y1_CL)
mean(dl.Y1_coverage)

mean(dl.Y2_LCL)
mean(dl.Y2_UCL)
mean(dl.Y2_CL)
mean(dl.Y2_coverage)

mean(dl.Y3_LCL)
mean(dl.Y3_UCL)
mean(dl.Y3_CL)
mean(dl.Y3_coverage)

mean(nn.Y1_LCL)
mean(nn.Y1_UCL)
mean(nn.Y1_CL)
mean(nn.Y1_coverage)

mean(nn.Y2_LCL)
mean(nn.Y2_UCL)
mean(nn.Y2_CL)
mean(nn.Y2_coverage)

mean(nn.Y3_LCL)
mean(nn.Y3_UCL)
mean(nn.Y3_CL)
mean(nn.Y3_coverage)

mean(dl1.Y1_LCL)
mean(dl1.Y1_UCL)
mean(dl1.Y1_CL)
mean(dl1.Y1_coverage)

mean(dl1.Y2_LCL)
mean(dl1.Y2_UCL)
mean(dl1.Y2_CL)
mean(dl1.Y2_coverage)

mean(dl1.Y3_LCL)
mean(dl1.Y3_UCL)
mean(dl1.Y3_CL)
mean(dl1.Y3_coverage)

mean(nn1.Y1_LCL)
mean(nn1.Y1_UCL)
mean(nn1.Y1_CL)
mean(nn1.Y1_coverage)

mean(nn1.Y2_LCL)
mean(nn1.Y2_UCL)
mean(nn1.Y2_CL)
mean(nn1.Y2_coverage)

mean(nn1.Y3_LCL)
mean(nn1.Y3_UCL)
mean(nn1.Y3_CL)
mean(nn1.Y3_coverage)


mean(gcdl1.Y1_LCL)
mean(gcdl1.Y1_UCL)
mean(gcdl1.Y1_CL)
mean(gcdl1.Y1_coverage)

mean(gcdl1.Y2_LCL)
mean(gcdl1.Y2_UCL)
mean(gcdl1.Y2_CL)
mean(gcdl1.Y2_coverage)

mean(gcdl1.Y3_LCL)
mean(gcdl1.Y3_UCL)
mean(gcdl1.Y3_CL)
mean(gcdl1.Y3_coverage)

mean(gcnn1.Y1_LCL)
mean(gcnn1.Y1_UCL)
mean(gcnn1.Y1_CL)
mean(gcnn1.Y1_coverage)

mean(gcnn1.Y2_LCL)
mean(gcnn1.Y2_UCL)
mean(gcnn1.Y2_CL)
mean(gcnn1.Y2_coverage)

mean(gcnn1.Y3_LCL)
mean(gcnn1.Y3_UCL)
mean(gcnn1.Y3_CL)
mean(gcnn1.Y3_coverage)


mean(vinedl1.Y1_LCL)
mean(vinedl1.Y1_UCL)
mean(vinedl1.Y1_CL)
mean(vinedl1.Y1_coverage)

mean(vinedl1.Y2_LCL)
mean(vinedl1.Y2_UCL)
mean(vinedl1.Y2_CL)
mean(vinedl1.Y2_coverage)

mean(vinedl1.Y3_LCL)
mean(vinedl1.Y3_UCL)
mean(vinedl1.Y3_CL)
mean(vinedl1.Y3_coverage)

mean(vinenn1.Y1_LCL)
mean(vinenn1.Y1_UCL)
mean(vinenn1.Y1_CL)
mean(vinenn1.Y1_coverage)

mean(vinenn1.Y2_LCL)
mean(vinenn1.Y2_UCL)
mean(vinenn1.Y2_CL)
mean(vinenn1.Y2_coverage)

mean(vinenn1.Y3_LCL)
mean(vinenn1.Y3_UCL)
mean(vinenn1.Y3_CL)
mean(vinenn1.Y3_coverage)



setwd("C:/Users/kjono/Dropbox/Documents/My Paper/Deep_regression")



#### ARL for Simulation Data



library(neuralnet)
library(copula)

z<-1000    # number of iterations
y<-13     # number of categorical variables
theta<-8 # copula parameter
n<-1000  # number of data  
p<-13    # total number of variables

set.seed(0205)
clayton <- claytonCopula(theta, dim=p)
sim.data <- as.matrix(rCopula(n, clayton))

#normal<-normalCopula(0.8, dim = p)
#sim.data <- as.matrix(rCopula(n, normal))
# cor(sim.data)


M <- matrix(rep(0,n*p), nrow = n, ncol=p)
for (i in 1:1000) {
  for (j in 1:5) {
    if(sim.data[i,j] < 0.5) M[i,j]<-0 else M[i,j]<-1 
  }
  for (k in 6:8) {
    
    if (sim.data[i,k] <= 0.5) {
      M[i,k]<-0  
    } 
    else if (sim.data[i,k] > 0.5  & sim.data[i,k] <= 0.8) {
      M[i,k]<-1
    } else {
      M[i,k]<-2  
    }
  }
  for (m in 9:13) {
    
    if (sim.data[i,m] <= 0.5) {
      M[i,m]<-0  
    } 
    else if (sim.data[i,m] > 0.5  & sim.data[i,m] <= 0.8) {
      M[i,m]<-1
    } else if (sim.data[i,m] > 0.8  & sim.data[i,m] <= 0.9) {
      M[i,m]<-2
    } else {
      M[i,m]<-3  
    }
  }
}


data1 <- data.frame(M)


colnames(data1)<-c("X1", "X2","X3","X4","X5","X6","X7",
                   "X8", "X9","X10", "Y1","Y2","Y3")

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

sdata <- as.data.frame(lapply(data1, normalize))


colnames(sdata)<-c("X1", "X2","X3","X4","X5","X6","X7",
                   "X8", "X9","X10", "Y1","Y2","Y3")




samplesize = 0.80 * nrow(sdata)

index = sample(1:nrow(sdata),samplesize )

pca <- prcomp(data1[,-(11:13)], scale = F)
#pca <- princomp(data[,-1], cor = F, scores = TRUE)
y<-data1[,c(11:13)]
data.pca <- data.frame(y, pca$x)	


train.pca = data.pca[index,]
test.pca = data.pca[-index,]
data.pca = data.pca[-index,]

train.pca.Y1<- train.pca[,-c(2,3)]
train.pca.Y2<- train.pca[,-c(1,3)]
train.pca.Y3<- train.pca[,-c(1,2)]

x_tr.pca <- train.pca[,-c(1:3)]
x_te.pca <- test.pca[,-c(1:3)]
y_tr.Y1 <- train.pca[,1]
y_te.Y1 <- test.pca[,1]

y_tr.Y2 <- train.pca[,2]
y_te.Y2 <- test.pca[,2]

y_tr.Y3 <- train.pca[,3]
y_te.Y3 <- test.pca[,3]

############ zlp ############
library(pscl)
# Fit Zero Inflated Poisson model
zip.Y1<-zeroinfl(Y1 ~., dist = 'poisson', data=train.pca.Y1)
predict_zero.Y1<-predict(zip.Y1, x_te.pca)
resultzeroPOI.Y1 <- data.frame(actual = y_te.Y1, prediction = predict_zero.Y1)
predictedzeroPOI.Y1=resultzeroPOI.Y1$prediction
actualzeroPOI.Y1=resultzeroPOI.Y1$actual

zip.Y2<-zeroinfl(Y2 ~., dist = 'poisson', data=train.pca.Y2)
predict_zero.Y2<-predict(zip.Y2, x_te.pca)
resultzeroPOI.Y2 <- data.frame(actual = y_te.Y2, prediction = predict_zero.Y2)
predictedzeroPOI.Y2=resultzeroPOI.Y2$prediction
actualzeroPOI.Y2=resultzeroPOI.Y2$actual

zip.Y3<-zeroinfl(Y3 ~., dist = 'poisson', data=train.pca.Y3)
predict_zero.Y3<-predict(zip.Y3, x_te.pca)
resultzeroPOI.Y3 <- data.frame(actual = y_te.Y3, prediction = predict_zero.Y3)
predictedzeroPOI.Y3=resultzeroPOI.Y3$prediction
actualzeroPOI.Y3=resultzeroPOI.Y3$actual


zip.Y1_r <-predictedzeroPOI.Y1-actualzeroPOI.Y1

zip.Y2_r <-predictedzeroPOI.Y2-actualzeroPOI.Y2

zip.Y3_r <-predictedzeroPOI.Y3-actualzeroPOI.Y3


zip.Y1_r_out=zip.Y1_r
zip.Y2_r_out=zip.Y2_r
zip.Y3_r_out=zip.Y3_r




# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(zip.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)


###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(zip.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)


###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(zip.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)



library(qcc)

length(zip.Y1_r_out)
#q1 = cusum(nn.Y1_r_out)
q1 = cusum(zip.Y1_r_out[1:150], newdata=zip.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Zero Inflated Poisson of Y1", xlab="", ylab="")

q2 = cusum(zip.Y2_r_out[1:150], newdata=zip.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Zero Inflated Poisson of Y2", xlab="", ylab="")

q3 = cusum(zip.Y3_r_out[1:150], newdata=zip.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Zero Inflated Poisson of Y3", xlab="", ylab="")




#### NN and DL Parts ###

trainNN = sdata[index,]
testNN = sdata[-index,]
datatest = sdata[-index,]


x_tr <- trainNN[,-(11:13)]
y_tr <- trainNN[,c(11:13)]
x_te <- testNN[,-c(11:13)]
y_te <- testNN[,c(11:13)]



nn <- neuralnet(as.formula(Y1 + Y2 + Y3 ~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10),
                data=trainNN, hidden=5, act.fct = "logistic", linear.output=FALSE, threshold=0.1)

nn.results <- compute(nn, testNN)
nnresults <- data.frame(actual = y_te, prediction = nn.results$net.result)


predictednn.1=nnresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
actualnn.Y1=nnresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

predictednn.2=nnresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
actualnn.Y2=nnresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 

predictednn.3=nnresults$prediction.3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 
actualnn.Y3=nnresults$actual.Y3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 




nn.Y1_r <- actualnn.Y1-predictednn.1

nn.Y2_r <- actualnn.Y2-predictednn.2

nn.Y3_r <- actualnn.Y3-predictednn.3

nn.Y1_r_out=nn.Y1_r
nn.Y2_r_out=nn.Y2_r
nn.Y3_r_out=nn.Y3_r




# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)


###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)


###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)



library(qcc)

length(nn.Y1_r_out)
#q1 = cusum(nn.Y1_r_out)
q1 = cusum(nn.Y1_r_out[1:150], newdata=nn.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Multivariate Neural Network of Y1", xlab="", ylab="")

q2 = cusum(nn.Y2_r_out[1:150], newdata=nn.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Multivariate Neural Network of Y2", xlab="", ylab="")

q3 = cusum(nn.Y3_r_out[1:150], newdata=nn.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Multivariate Neural Network of Y3", xlab="", ylab="")





##


dl <- neuralnet(as.formula(Y1 + Y2 + Y3 ~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10),
                data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)

dl.results <- compute(dl, testNN)
dlresults <- data.frame(actual = y_te, prediction = dl.results$net.result)


predicteddl.1=dlresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
actualdl.Y1=dlresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

predicteddl.2=dlresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
actualdl.Y2=dlresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 

predicteddl.3=dlresults$prediction.3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 
actualdl.Y3=dlresults$actual.Y3 * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3) 


dl.Y1_r <- actualdl.Y1-predicteddl.1
dl.Y2_r <- actualdl.Y2-predicteddl.2
dl.Y3_r <- actualdl.Y3-predicteddl.3

dl.Y1_r_out=dl.Y1_r
dl.Y2_r_out=dl.Y2_r
dl.Y3_r_out=dl.Y3_r


# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)



library(qcc)

length(dl.Y1_r_out)

q1 = cusum(dl.Y1_r_out[1:150], newdata=dl.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Multivariate Deep Learning of Y1", xlab="", ylab="")

q2 = cusum(dl.Y2_r_out[1:150], newdata=dl.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Multivariate Deep Learning of Y2", xlab="", ylab="")

q3 = cusum(dl.Y3_r_out[1:150], newdata=dl.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Multivariate Deep Learning of Y3", xlab="", ylab="")




####

nn.Y1 <- neuralnet(Y1~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
nn.Y1.results <- compute(nn.Y1, testNN)
nnY1.results <- data.frame(actual = testNN$Y1, prediction = nn.Y1.results$net.result)

nn.Y1.predicted=nnY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
nn.Y1.actual=nnY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

nn.Y2 <- neuralnet(Y2~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=5,act.fct = "logistic", linear.output=TRUE, threshold=0.3)
nn.Y2.results <- compute(nn.Y2, testNN)
nnY2.results <- data.frame(actual = testNN$Y2, prediction = nn.Y2.results$net.result)

nn.Y2.predicted=nnY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
nn.Y2.actual=nnY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)

nn.Y3 <- neuralnet(Y3~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=5,act.fct = "logistic", linear.output=TRUE, threshold=0.3)
nn.Y3.results <- compute(nn.Y3, testNN)
nnY3.results <- data.frame(actual = testNN$Y3, prediction = nn.Y3.results$net.result)

nn.Y3.predicted=nnY3.results$prediction * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)
nn.Y3.actual=nnY3.results$actual * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)

nn1.Y1_r <- nn.Y1.actual-nn.Y1.predicted
nn1.Y2_r <- nn.Y2.actual-nn.Y2.predicted
nn1.Y3_r <- nn.Y3.actual-nn.Y3.predicted

nn1.Y1_r_out=nn1.Y1_r
nn1.Y2_r_out=nn1.Y2_r
nn1.Y3_r_out=nn1.Y3_r



# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn1.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)


library(qcc)

length(nn1.Y1_r_out)
#q1 = cusum(nn1.Y1_r_out)
q1 = cusum(nn1.Y1_r_out[1:150], newdata=nn1.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Univariate Neural Network of Y1", xlab="", ylab="")

q2 = cusum(nn1.Y2_r_out[1:150], newdata=nn1.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Univariate Neural Network of Y2", xlab="", ylab="")

q3 = cusum(nn1.Y3_r_out[1:150], newdata=nn1.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Univariate Neural Network of Y3", xlab="", ylab="")




###

dl.Y1 <- neuralnet(Y1~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
dl.Y1.results <- compute(dl.Y1, testNN)
dlY1.results <- data.frame(actual = testNN$Y1, prediction = dl.Y1.results$net.result)

dl.Y1.predicted=dlY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
dl.Y1.actual=dlY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

dl.Y2 <- neuralnet(Y2~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
dl.Y2.results <- compute(dl.Y2, testNN)
dlY2.results <- data.frame(actual = testNN$Y2, prediction = dl.Y2.results$net.result)

dl.Y2.predicted=dlY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
dl.Y2.actual=dlY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)


dl.Y3 <- neuralnet(Y3~ X1 + X2 + X3+ X4 + X5 + X6+X7 + X8 + X9+ X10, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
dl.Y3.results <- compute(dl.Y3, testNN)
dlY3.results <- data.frame(actual = testNN$Y3, prediction = dl.Y3.results$net.result)

dl.Y3.predicted=dlY3.results$prediction * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)
dl.Y3.actual=dlY3.results$actual * (max(data1$Y3)-min(data1$Y3)) + min(data1$Y3)

dl1.Y1_r <- dl.Y1.actual-dl.Y1.predicted
dl1.Y2_r <- dl.Y2.actual-dl.Y2.predicted
dl1.Y3_r <- dl.Y3.actual-dl.Y3.predicted

dl1.Y1_r_out=dl1.Y1_r 
dl1.Y2_r_out=dl1.Y2_r
dl1.Y3_r_out=dl1.Y3_r



# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl1.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

library(qcc)

length(dl1.Y1_r_out)

q1 = cusum(dl1.Y1_r_out[1:150], newdata=dl1.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Univariate Deep Learning of Y1", xlab="", ylab="")

q2 = cusum(dl1.Y2_r_out[1:150], newdata=dl1.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Univariate Deep Learning of Y2", xlab="", ylab="")

q3 = cusum(dl1.Y3_r_out[1:150], newdata=dl1.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Univariate Deep Learning of Y3", xlab="", ylab="")



#####

####




dl1_data<-cbind(dl.Y1.predicted, dl.Y2.predicted, dl.Y3.predicted)
y1 = dl1_data[,1]
y2 = dl1_data[,2]
y3 = dl1_data[,3]
data<-data.frame(cbind(y1, y2, y3))

library(gcmr)

y1.23.00<-gcmr( y1~y2+y3, data = data, marginal = gaussian.marg(link="identity"),
                cormat = arma.cormat(0, 0))
y2.13.00<-gcmr( y2~y1+y3, data = data, marginal = gaussian.marg(link="identity"),
                cormat = arma.cormat(0, 0))
y3.12.00<-gcmr( y3~y1+y2, data = data, marginal = gaussian.marg(link="identity"),
                cormat = arma.cormat(0, 0))

# summary(y1.23.00)
predictions1 <-y1.23.00$estimate[1]+
  y1.23.00$estimate[2]*data$y2+y1.23.00$estimate[3]*data$y3

# summary(y2.13.00)
predictions2 <-y2.13.00$estimate[1]+
  y2.13.00$estimate[2]*data$y1+y2.13.00$estimate[3]*data$y3

# summary(y3.12.00)
predictions3 <-y3.12.00$estimate[1]+
  y3.12.00$estimate[2]*data$y1+y3.12.00$estimate[3]*data$y2



gcdl1.Y1_r <- dl.Y1.actual-predictions1

gcdl1.Y2_r <- dl.Y2.actual-predictions2

gcdl1.Y3_r <- dl.Y3.actual-predictions3

gcdl1.Y1_r_out<-gcdl1.Y1_r

gcdl1.Y2_r_out<-gcdl1.Y2_r

gcdl1.Y3_r_out<-gcdl1.Y3_r



# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(gcdl1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(gcdl1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(gcdl1.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)



length(gcdl1.Y1_r_out)

q1 = cusum(gcdl1.Y1_r_out[1:150], newdata=gcdl1.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Gaussian Copula Deep Learning of Y1", xlab="", ylab="")

q2 = cusum(gcdl1.Y2_r_out[1:150], newdata=gcdl1.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Gaussian Copula Deep Learning of Y2", xlab="", ylab="")

q3 = cusum(gcdl1.Y3_r_out[1:150], newdata=gcdl1.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Gaussian Copula Deep Learning of Y3", xlab="", ylab="")


#####

# fit vine regression model 

library(vinereg)

vy1.23.00<-vinereg( y1~y2+y3, data = data)
vy2.13.00<-vinereg( y2~y1+y3, data = data)
vy3.12.00<-vinereg( y3~y1+y2, data = data)

vinepredictions1dl <-predict(vy1.23.00, newdata = data, alpha = NA)
vinepredictions2dl <-predict(vy2.13.00, newdata = data, alpha = NA)
vinepredictions3dl <-predict(vy3.12.00, newdata = data, alpha = NA)


vinedl1.Y1_r <- dl.Y1.actual - vinepredictions1dl$mean
vinedl1.Y2_r <- dl.Y2.actual - vinepredictions2dl$mean
vinedl1.Y3_r <- dl.Y3.actual - vinepredictions3dl$mean


vinedl1.Y1_r_out<-vinedl1.Y1_r

vinedl1.Y2_r_out<-vinedl1.Y2_r

vinedl1.Y3_r_out<-vinedl1.Y3_r





# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinedl1.Y1_r_out
p2 = vinedl1.Y2_r_out

p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)


rhoyx1<-mat1[150:200,1]

matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(1,1)-sGARCH(1,1) of Vine Copula Deep Learning of Y1 and Y2")



comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)


# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinedl1.Y1_r_out
p2 = vinedl1.Y3_r_out

p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)


rhoyx1<-mat1[150:200,1]

matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(1,1)-sGARCH(1,1) of Vine Copula Deep Learning of Y1 and Y3")

comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)




# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinedl1.Y2_r_out
p2 = vinedl1.Y3_r_out
p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(0,0)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)


rhoyx1<-mat1[150:200,1]

matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(0,0)-sGARCH(1,1) of Vine Copula Deep Learning of Y2 and Y3")

comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)




# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinedl1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinedl1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinedl1.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)



length(vinedl1.Y1_r_out)

q1 = cusum(vinedl1.Y1_r_out[1:150], newdata=vinedl1.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Vine Copula Deep Learning of Y1", xlab="", ylab="")

q2 = cusum(vinedl1.Y2_r_out[1:150], newdata=vinedl1.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Vine Copula Deep Learning of Y2", xlab="", ylab="")

q3 = cusum(vinedl1.Y3_r_out[1:150], newdata=vinedl1.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Vine Copula Deep Learning of Y3", xlab="", ylab="")




####



nn1_data<-cbind(nn.Y1.predicted, nn.Y2.predicted, nn.Y3.predicted)
y1 = nn1_data[,1]
y2 = nn1_data[,2]
y3 = nn1_data[,3]
data<-data.frame(cbind(y1, y2, y3))

library(gcmr)

y1.23.00<-gcmr( y1~y2+y3, data = data, marginal = gaussian.marg(link="identity"),
                cormat = arma.cormat(0, 0))
y2.13.00<-gcmr( y2~y1+y3, data = data, marginal = gaussian.marg(link="identity"),
                cormat = arma.cormat(0, 0))
y3.12.00<-gcmr( y3~y1+y2, data = data, marginal = gaussian.marg(link="identity"),
                cormat = arma.cormat(0, 0))

# summary(y1.23.00)
predictions1nn <-y1.23.00$estimate[1]+
  y1.23.00$estimate[2]*data$y2+y1.23.00$estimate[3]*data$y3

# summary(y2.13.00)
predictions2nn <-y2.13.00$estimate[1]+
  y2.13.00$estimate[2]*data$y1+y2.13.00$estimate[3]*data$y3

# summary(y3.12.00)
predictions3nn <-y3.12.00$estimate[1]+
  y3.12.00$estimate[2]*data$y1+y3.12.00$estimate[3]*data$y2



gcnn1.Y1_r <- nn.Y1.actual-predictions1nn

gcnn1.Y2_r <- nn.Y2.actual-predictions2nn

gcnn1.Y3_r <- nn.Y3.actual-predictions3nn

gcnn1.Y1_r_out<-gcnn1.Y1_r

gcnn1.Y2_r_out<-gcnn1.Y2_r

gcnn1.Y3_r_out<-gcnn1.Y3_r



# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(gcnn1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(gcnn1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(gcnn1.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)



library(qcc)

length(gcnn1.Y1_r_out)

q1 = cusum(gcnn1.Y1_r_out[1:150], newdata=gcnn1.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Gaussian Copula Neural Network of Y1", xlab="", ylab="")

q2 = cusum(gcnn1.Y2_r_out[1:150], newdata=gcnn1.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Gaussian Copula Neural Network of Y2", xlab="", ylab="")

q3 = cusum(gcnn1.Y3_r_out[1:150], newdata=gcnn1.Y3_r_out[151:40],
           add.stats=FALSE,
           title="Gaussian Copula Neural Network of Y3", xlab="", ylab="")


# fit vine regression model 

library(vinereg)

vy1.23.00<-vinereg( y1~y2+y3, data = data)
vy2.13.00<-vinereg( y2~y1+y3, data = data)
vy3.12.00<-vinereg( y3~y1+y2, data = data)

vinepredictions1nn <-predict(vy1.23.00, newdata = data, alpha = NA)
vinepredictions2nn <-predict(vy2.13.00, newdata = data, alpha = NA)
vinepredictions3nn <-predict(vy3.12.00, newdata = data, alpha = NA)


vinenn1.Y1_r <- nn.Y1.actual - vinepredictions1nn$mean

vinenn1.Y2_r <- nn.Y2.actual - vinepredictions2nn$mean

vinenn1.Y3_r <- nn.Y3.actual - vinepredictions3nn$mean


vinenn1.Y1_r_out<-vinenn1.Y1_r

vinenn1.Y2_r_out<-vinenn1.Y2_r

vinenn1.Y3_r_out<-vinenn1.Y3_r

#par(mfrow = c(1,1))


# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinenn1.Y1_r_out
p2 = vinenn1.Y2_r_out
p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(2,2)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)


rhoyx1<-mat1[150:200,1]

matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(2,2)-sGARCH(1,1) of Vine Copula Neural Network of Y1 and Y2")

comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)


# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinenn1.Y1_r_out
p2 = vinenn1.Y3_r_out
p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(2,2)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)


rhoyx1<-mat1[150:200,1]

matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(2,2)-sGARCH(1,1) of Vine Copula Neural Network of Y1 and Y3")

comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)




# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinenn1.Y2_r_out
p2 = vinenn1.Y3_r_out
p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(4,3)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)


rhoyx1<-mat1[150:200,1]

matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(4,3)-sGARCH(2,2) of Vine Copula Neural Network of Y2 and Y3")

comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)



# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinenn1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinenn1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinenn1.Y3_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=50)
arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=150)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=50, r=0.2)
arl.ewma(S=10000, n=50, r=0.6)
arl.ewma(S=10000, n=50, r=0.9)
arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=150, r=0.2)
arl.ewma(S=10000, n=150, r=0.6)
arl.ewma(S=10000, n=150, r=0.9)



library(qcc)

length(vinenn1.Y1_r_out)

q1 = cusum(vinenn1.Y1_r_out[1:150], newdata=vinenn1.Y1_r_out[151:200],
           add.stats=FALSE,
           title="Vine Copula Neural Network of Y1", xlab="", ylab="")

q2 = cusum(vinenn1.Y2_r_out[1:150], newdata=vinenn1.Y2_r_out[151:200],
           add.stats=FALSE,
           title="Vine Copula Neural Network of Y2", xlab="", ylab="")

q3 = cusum(vinenn1.Y3_r_out[1:150], newdata=vinenn1.Y3_r_out[151:200],
           add.stats=FALSE,
           title="Vine Copula Neural Network of Y3", xlab="", ylab="")



### Real Data



library(neuralnet)
library(copula)
library(CASdatasets)
data(ausprivauto0405)

z<-1000    # number of iterations

RMSE.zip<-NULL

RMSE.zip<- matrix(rep(0,z*2),z,2)

RMSE.NN<-RMSE.NN1<-RMSE.DL<-RMSE.DL1<-NULL

RMSE.NN<-RMSE.NN1<-RMSE.DL<-RMSE.DL1<-matrix(rep(0,z*2),z,2)

RMSE.gcmr.NN1<-RMSE.gcmr.DL1<-NULL

RMSE.gcmr.NN1<-RMSE.gcmr.DL1<-matrix(rep(0,z*2),z,2)

RMSE.vine.NN1<-RMSE.vine.DL1<-NULL

RMSE.vine.NN1<-RMSE.vine.DL1<-matrix(rep(0,z*2),z,2)

WMAPE.zip<-NULL

WMAPE.zip<- matrix(rep(0,z*2),z,2)

WMAPE.NN<-WMAPE.NN1<-WMAPE.DL<-WMAPE.DL1<-NULL

WMAPE.NN<-WMAPE.NN1<-WMAPE.DL<-WMAPE.DL1<-matrix(rep(0,z*2),z,2)

WMAPE.gcmr.NN1<-WMAPE.gcmr.DL1<-NULL

WMAPE.gcmr.NN1<-WMAPE.gcmr.DL1<-matrix(rep(0,z*2),z,2)

WMAPE.vine.NN1<-WMAPE.vine.DL1<-NULL

WMAPE.vine.NN1<-WMAPE.vine.DL1<-matrix(rep(0,z*2),z,2)

MAD.zip<-NULL

MAD.zip<- matrix(rep(0,z*2),z,2)

MAD.NN<-MAD.NN1<-MAD.DL<-MAD.DL1<-NULL

MAD.NN<-MAD.NN1<-MAD.DL<-MAD.DL1<-matrix(rep(0,z*2),z,2)

MAD.gcmr.NN1<-MAD.gcmr.DL1<-NULL

MAD.gcmr.NN1<-MAD.gcmr.DL1<-matrix(rep(0,z*2),z,2)

MAD.vine.NN1<-MAD.vine.DL1<-NULL

MAD.vine.NN1<-MAD.vine.DL1<-matrix(rep(0,z*2),z,2)


zip.Y1_LCL <-zip.Y2_LCL  <- 0
zip.Y1_UCL <- zip.Y2_UCL  <- 0
zip.Y1_CL <- zip.Y2_CL <- 0
zip.Y1_coverage <-zip.Y2_coverage <- 0



dl.Y1_LCL <-dl.Y2_LCL <- 0
dl.Y1_UCL <- dl.Y2_UCL <- 0
dl.Y1_CL <- dl.Y2_CL <- 0
dl.Y1_coverage <-dl.Y2_coverage <- 0

nn.Y1_LCL <- nn.Y2_LCL <- 0
nn.Y1_UCL <- nn.Y2_UCL <- 0
nn.Y1_CL <- nn.Y2_CL <- 0
nn.Y1_coverage <-nn.Y2_coverage <- 0

dl1.Y1_LCL <-dl1.Y2_LCL <- 0
dl1.Y1_UCL <- dl1.Y2_UCL <- 0
dl1.Y1_CL <- dl1.Y2_CL <- 0
dl1.Y1_coverage <-dl1.Y2_coverage <- 0

nn1.Y1_LCL <- nn1.Y2_LCL <- 0
nn1.Y1_UCL <- nn1.Y2_UCL <- 0
nn1.Y1_CL <- nn1.Y2_CL <- 0
nn1.Y1_coverage <-nn1.Y2_coverage <- 0

gcdl1.Y1_LCL <-gcdl1.Y2_LCL <- 0
gcdl1.Y1_UCL <- gcdl1.Y2_UCL <- 0
gcdl1.Y1_CL <- gcdl1.Y2_CL <- 0
gcdl1.Y1_coverage <-gcdl1.Y2_coverage <- 0

gcnn1.Y1_LCL <- gcnn1.Y2_LCL <- 0
gcnn1.Y1_UCL <- gcnn1.Y2_UCL <- 0
gcnn1.Y1_CL <- gcnn1.Y2_CL <- 0
gcnn1.Y1_coverage <-gcnn1.Y2_coverage <- 0


vinedl1.Y1_LCL <-vinedl1.Y2_LCL <- 0
vinedl1.Y1_UCL <- vinedl1.Y2_UCL <- 0
vinedl1.Y1_CL <- vinedl1.Y2_CL <- 0
vinedl1.Y1_coverage <-vinedl1.Y2_coverage <- 0

vinenn1.Y1_LCL <- vinenn1.Y2_LCL <- 0
vinenn1.Y1_UCL <- vinenn1.Y2_UCL <- 0
vinenn1.Y1_CL <- vinenn1.Y2_CL <- 0
vinenn1.Y1_coverage <-vinenn1.Y2_coverage <- 0


zip.Y1_RL0<-zip.Y2_RL0<- NULL

dl.Y1_RL0<-nn.Y1_RL0 <- NULL
dl.Y2_RL0<-nn.Y2_RL0 <- NULL

dl1.Y1_RL0<-nn1.Y1_RL0 <- NULL
dl1.Y2_RL0<-nn1.Y2_RL0 <- NULL

gcdl1.Y1_RL0<-gcnn1.Y1_RL0 <- NULL
gcdl1.Y2_RL0<-gcnn1.Y2_RL0 <- NULL

vinedl1.Y1_RL0<-vinenn1.Y1_RL0 <- NULL
vinedl1.Y2_RL0<-vinenn1.Y2_RL0 <- NULL


zip.ARL.0.Y1<-zip.ARL.0.Y2 <- NULL
zip.ARL.0.Y1<-zip.ARL.0.Y2 <- rep(0,z)

zip.Y1_ARL.0a<-zip.Y2_ARL.0a<- NULL
zip.Y1_ARL.0a<-zip.Y2_ARL.0a<- rep(0,z)


NN.ARL.0.Y1<-DL.ARL.0.Y1<-NN1.ARL.0.Y1<-DL1.ARL.0.Y1 <- NULL
NN.ARL.0.Y2<-DL.ARL.0.Y2<-NN1.ARL.0.Y2<-DL1.ARL.0.Y2 <- NULL

gcNN1.ARL.0.Y1<-gcDL1.ARL.0.Y1 <- NULL
gcNN1.ARL.0.Y2<-gcDL1.ARL.0.Y2 <- NULL

vineNN1.ARL.0.Y1<-vineDL1.ARL.0.Y1 <- NULL
vineNN1.ARL.0.Y2<-vineDL1.ARL.0.Y2 <- NULL

NN.ARL.0.Y1<-DL.ARL.0.Y1<-NN1.ARL.0.Y1<-DL1.ARL.0.Y1 <- rep(0,z)
NN.ARL.0.Y2<-DL.ARL.0.Y2<-NN1.ARL.0.Y2<-DL1.ARL.0.Y2 <- rep(0,z)

gcNN1.ARL.0.Y1<-gcDL1.ARL.0.Y1 <- rep(0,z)
gcNN1.ARL.0.Y2<-gcDL1.ARL.0.Y2 <- rep(0,z)

vineNN1.ARL.0.Y1<-vineDL1.ARL.0.Y1 <- rep(0,z)
vineNN1.ARL.0.Y2<-vineDL1.ARL.0.Y2 <- rep(0,z)


NN.Y1_ARL.0a<-NN.Y2_ARL.0a<- NULL
DL.Y1_ARL.0a<-DL.Y2_ARL.0a<- NULL
NN1.Y1_ARL.0a<-NN1.Y2_ARL.0a<- NULL
DL1.Y1_ARL.0a<-DL1.Y2_ARL.0a<- NULL

gcNN1.Y1_ARL.0a<-gcNN1.Y2_ARL.0a<- NULL
gcDL1.Y1_ARL.0a<-gcDL1.Y2_ARL.0a<- NULL

vineNN1.Y1_ARL.0a<-vineNN1.Y2_ARL.0a<- NULL
vineDL1.Y1_ARL.0a<-vineDL1.Y2_ARL.0a<- NULL

NN.Y1_ARL.0a<-NN.Y2_ARL.0a<- rep(0,z)
DL.Y1_ARL.0a<-DL.Y2_ARL.0a<- rep(0,z)
NN1.Y1_ARL.0a<-NN1.Y2_ARL.0a<- rep(0,z)
DL1.Y1_ARL.0a<-DL1.Y2_ARL.0a<- rep(0,z)

gcNN1.Y1_ARL.0a<-gcNN1.Y2_ARL.0a<- rep(0,z)
gcDL1.Y1_ARL.0a<-gcDL1.Y2_ARL.0a<- rep(0,z)

vineNN1.Y1_ARL.0a<-vineNN1.Y2_ARL.0a<- rep(0,z)
vineDL1.Y1_ARL.0a<-vineDL1.Y2_ARL.0a<- rep(0,z)





for (q in 1:z){
  
  set.seed(q)
  
  library(CASdatasets)
  data(ausprivauto0405)
  #make factor variables levels(ausprivauto0405$VehAge)
  ausprivauto0405$VehAge <- as.numeric(ordered(ausprivauto0405$VehAge, levels=c("old cars", "oldest cars", "young cars", "youngest cars")))
  ausprivauto0405$VehBody<- as.numeric(ordered(ausprivauto0405$VehBody, levels=  c("Bus", "Convertible", "Coupe","Hardtop", "Hatchback", "Minibus", "Motorized caravan", "Panel van","Roadster", "Sedan","Station wagon","Truck", "Utility")))        
  ausprivauto0405$Gender <- as.numeric(ordered(ausprivauto0405$Gender, levels=c("Female", "Male")))
  ausprivauto0405$DrivAge<- as.numeric(ordered(ausprivauto0405$DrivAge, levels=c("old people", "older work. people", "oldest people", "working people", "young people", "youngest people")))
  
     
  
  data1 <- data.frame(ausprivauto0405[,c(7,8,1,2,3,4,5,6,9)])
  #samplesize<-600
  #data1<-data[sample(nrow(data), samplesize), ]
  colnames(data1)<-c("Y1", "Y2", "Exposure", "VehValue", "VehAge", "VehBody", "Gender", "DrivAge", "ClaimAmount")
  
  library(vioplot)
  vioplot(data1[,c(1:2)], ylab='Frequency', col="blue")
  title("Y1=ClaimNb and Y2=ClaimAmount")
  
#  cor(data1)
  
  normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }
  
  sdata <- as.data.frame(lapply(data1, normalize))
  colnames(sdata)<-c("Y1", "Y2", "Exposure", "VehValue", "VehAge", "VehBody", "Gender", "DrivAge", "ClaimAmount")
  head(sdata)
  
  samplesize = 0.80 * nrow(sdata)
  
  index = sample(1:nrow(sdata),samplesize )
  
  pca <- prcomp(data1[,-(1:2)], scale = F)
  y<-data1[,c(1:2)]
  data.pca <- data.frame(y, pca$x)	
  
    
    train.pca = data.pca[index,]
    test.pca = data.pca[-index,]
    data.pca = data.pca[-index,]
    
    train.pca.Y1<- train.pca[,-c(2)]
    train.pca.Y2<- train.pca[,-c(1)]
  
    x_tr.pca <- train.pca[,-c(1:2)]
    x_te.pca <- test.pca[,-c(1:2)]
    y_tr.Y1 <- train.pca[,1]
    y_te.Y1 <- test.pca[,1]
    
    y_tr.Y2 <- train.pca[,2]
    y_te.Y2 <- test.pca[,2]
    


  
  
  #### NN and DL Parts ###
  
  trainNN = sdata[index,]
  testNN = sdata[-index,]
  datatest = sdata[-index,]
  
  
  
  x_tr <- trainNN[,-c(1:2)]
  y_tr <- trainNN[,c(1:2)]
  x_te <- testNN[,-c(1:2)]
  y_te <- testNN[,c(1:2)]

  

  
  nn <- neuralnet(as.formula(Y1 + Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge),
                  data=trainNN, hidden=5, act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  
  nn.results <- compute(nn, testNN)
  nnresults <- data.frame(actual = y_te, prediction = nn.results$net.result)
  
  
  predictednn.1=nnresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  actualnn.Y1=nnresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  
  predictednn.2=nnresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
  actualnn.Y2=nnresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
  

  
  RMSE.NN[q,1]<- sqrt((sum(actualnn.Y1-predictednn.1)^2)/length(actualnn.Y1))
  RMSE.NN[q,2]<- sqrt((sum(actualnn.Y2-predictednn.2)^2)/length(actualnn.Y2))

  WMAPE.NN[q,1] <- sum(abs(actualnn.Y1-predictednn.1))/sum(actualnn.Y1)
  WMAPE.NN[q,2] <- sum(abs(actualnn.Y2-predictednn.2))/sum(actualnn.Y2)

  MAD.NN[q,1] <- mean(abs(actualnn.Y1-predictednn.1))
  MAD.NN[q,2] <- mean(abs(actualnn.Y2-predictednn.2))

  nn.Y1_r <- actualnn.Y1-predictednn.1
  nn.Y1_CL[q] <- mean(nn.Y1_r)
  nn.Y1_LCL[q] <- mean(nn.Y1_r) - 3*sd(nn.Y1_r)
  nn.Y1_UCL[q] <- mean(nn.Y1_r) + 3*sd(nn.Y1_r)
  nn.Y1_coverage[q] <- mean(nn.Y1_r > nn.Y1_LCL[q] & nn.Y1_r < nn.Y1_UCL[q])		
  nn.Y1_check0 <- which(nn.Y1_r < nn.Y1_LCL[q] | nn.Y1_r > nn.Y1_UCL[q])
  if(length(nn.Y1_check0)!=0L)	nn.Y1_RL0 <- c(nn.Y1_RL0, min(nn.Y1_check0))
  
  nn.Y2_r <- actualnn.Y2-predictednn.2
  nn.Y2_CL[q] <- mean(nn.Y2_r)
  nn.Y2_LCL[q] <- mean(nn.Y2_r) - 3*sd(nn.Y2_r)
  nn.Y2_UCL[q] <- mean(nn.Y2_r) + 3*sd(nn.Y2_r)
  nn.Y2_coverage[q] <- mean(nn.Y2_r > nn.Y2_LCL[q] & nn.Y2_r < nn.Y2_UCL[q])		
  nn.Y2_check0 <- which(nn.Y2_r < nn.Y2_LCL[q] | nn.Y2_r > nn.Y2_UCL[q])
  if(length(nn.Y2_check0)!=0L)	nn.Y2_RL0 <- c(nn.Y2_RL0, min(nn.Y2_check0))
  

  
  
  
  dl <- neuralnet(as.formula(Y1 + Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge),
                  data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  
  dl.results <- compute(dl, testNN)
  dlresults <- data.frame(actual = y_te, prediction = dl.results$net.result)
  
  
  predicteddl.1=dlresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  actualdl.Y1=dlresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  
  predicteddl.2=dlresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
  actualdl.Y2=dlresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
  
  
  
  RMSE.DL[q,1]<- sqrt((sum(actualdl.Y1-predicteddl.1)^2)/length(actualdl.Y1))
  RMSE.DL[q,2]<- sqrt((sum(actualdl.Y2-predicteddl.2)^2)/length(actualdl.Y2))
  
  WMAPE.DL[q,1] <- sum(abs(actualdl.Y1-predicteddl.1))/sum(actualdl.Y1)
  WMAPE.DL[q,2] <- sum(abs(actualdl.Y2-predicteddl.2))/sum(actualdl.Y2)
  
  MAD.DL[q,1] <- mean(abs(actualdl.Y1-predicteddl.1))
  MAD.DL[q,2] <- mean(abs(actualdl.Y2-predicteddl.2))
  
  
  dl.Y1_r <- actualdl.Y1-predicteddl.1
  dl.Y1_CL[q] <- mean(dl.Y1_r)
  dl.Y1_LCL[q] <- mean(dl.Y1_r) - 3*sd(dl.Y1_r)
  dl.Y1_UCL[q] <- mean(dl.Y1_r) + 3*sd(dl.Y1_r)
  dl.Y1_coverage[q] <- mean(dl.Y1_r > dl.Y1_LCL[q] & dl.Y1_r < dl.Y1_UCL[q])		
  dl.Y1_check0 <- which(dl.Y1_r < dl.Y1_LCL[q] | dl.Y1_r > dl.Y1_UCL[q])
  if(length(dl.Y1_check0)!=0L)	dl.Y1_RL0 <- c(dl.Y1_RL0, min(dl.Y1_check0))
  
  dl.Y2_r <- actualdl.Y2-predicteddl.2
  dl.Y2_CL[q] <- mean(dl.Y2_r)
  dl.Y2_LCL[q] <- mean(dl.Y2_r) - 3*sd(dl.Y2_r)
  dl.Y2_UCL[q] <- mean(dl.Y2_r) + 3*sd(dl.Y2_r)
  dl.Y2_coverage[q] <- mean(dl.Y2_r > dl.Y2_LCL[q] & dl.Y2_r < dl.Y2_UCL[q])		
  dl.Y2_check0 <- which(dl.Y2_r < dl.Y2_LCL[q] | dl.Y2_r > dl.Y2_UCL[q])
  if(length(dl.Y2_check0)!=0L)	dl.Y2_RL0 <- c(dl.Y2_RL0, min(dl.Y2_check0))
  
  
  
  nn.Y1 <- neuralnet(Y1 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
  nn.Y1.results <- compute(nn.Y1, testNN)
  nnY1.results <- data.frame(actual = testNN$Y1, prediction = nn.Y1.results$net.result)
  
  nn.Y1.predicted=nnY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  nn.Y1.actual=nnY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  
  nn.Y2 <- neuralnet(Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
  nn.Y2.results <- compute(nn.Y2, testNN)
  nnY2.results <- data.frame(actual = testNN$Y2, prediction = nn.Y2.results$net.result)
  
  nn.Y2.predicted=nnY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  nn.Y2.actual=nnY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  

  RMSE.NN1[q,1]<- sqrt((sum(nn.Y1.actual-nn.Y1.predicted)^2)/length(nn.Y1.actual))
  RMSE.NN1[q,2]<- sqrt((sum(nn.Y2.actual-nn.Y2.predicted)^2)/length(nn.Y2.actual))

  WMAPE.NN1[q,1] <- sum(abs(nn.Y1.actual-nn.Y1.predicted))/sum(nn.Y1.actual)
  WMAPE.NN1[q,2] <- sum(abs(nn.Y2.actual-nn.Y2.predicted))/sum(nn.Y2.actual)

  MAD.NN1[q,1] <- mean(abs(nn.Y1.actual-nn.Y1.predicted))
  MAD.NN1[q,2] <- mean(abs(nn.Y2.actual-nn.Y2.predicted))

  
  nn1.Y1_r <- nn.Y1.actual-nn.Y1.predicted
  nn1.Y1_CL[q] <- mean(nn1.Y1_r)
  nn1.Y1_LCL[q] <- mean(nn1.Y1_r) - 3*sd(nn1.Y1_r)
  nn1.Y1_UCL[q] <- mean(nn1.Y1_r) + 3*sd(nn1.Y1_r)
  nn1.Y1_coverage[q] <- mean(nn1.Y1_r > nn1.Y1_LCL[q] & nn1.Y1_r < nn1.Y1_UCL[q])		
  nn1.Y1_check0 <- which(nn1.Y1_r < nn1.Y1_LCL[q] | nn1.Y1_r > nn1.Y1_UCL[q])
  if(length(nn1.Y1_check0)!=0L)	nn1.Y1_RL0 <- c(nn1.Y1_RL0, min(nn1.Y1_check0))
  
  nn1.Y2_r <- nn.Y2.actual-nn.Y2.predicted
  nn1.Y2_CL[q] <- mean(nn1.Y2_r)
  nn1.Y2_LCL[q] <- mean(nn1.Y2_r) - 3*sd(nn1.Y2_r)
  nn1.Y2_UCL[q] <- mean(nn1.Y2_r) + 3*sd(nn1.Y2_r)
  nn1.Y2_coverage[q] <- mean(nn1.Y2_r > nn1.Y2_LCL[q] & nn1.Y2_r < nn1.Y2_UCL[q])		
  nn1.Y2_check0 <- which(nn1.Y2_r < nn1.Y2_LCL[q] | nn1.Y2_r > nn1.Y2_UCL[q])
  if(length(nn1.Y2_check0)!=0L)	nn1.Y2_RL0 <- c(nn1.Y2_RL0, min(nn1.Y2_check0))
  
  
  
  
  # The data:----
  
  nn1_data<-cbind(nn.Y1.predicted, nn.Y2.predicted)
  y1 = nn1_data[,1]
  y2 = nn1_data[,2]
  datann<-data.frame(cbind(y1, y2))
  

  
  # fit vine regression model 
  
  library(vinereg)
  
  vy1.2.00<-vinereg( y1~y2, data = datann)
  vy2.1.00<-vinereg( y2~y1, data = datann)
  
  vinepredictions1nn <-predict(vy1.2.00, newdata = datann, alpha = NA)
  vinepredictions2nn <-predict(vy2.1.00, newdata = datann, alpha = NA)
  
  
  RMSE.vine.NN1[q,1]<- sqrt(mean((nn.Y1.actual - vinepredictions1nn$mean)^2))
  RMSE.vine.NN1[q,2]<- sqrt(mean((nn.Y2.actual - vinepredictions2nn$mean)^2))
  
  WMAPE.vine.NN1[q,1] <- sum(abs(nn.Y1.actual - vinepredictions1nn$mean))/sum(nn.Y1.actual)
  WMAPE.vine.NN1[q,2] <- sum(abs(nn.Y2.actual - vinepredictions2nn$mean))/sum(nn.Y2.actual)
  
  MAD.vine.NN1[q,1] <- mean(abs(nn.Y1.actual - vinepredictions1nn$mean))
  MAD.vine.NN1[q,2] <- mean(abs(nn.Y2.actual - vinepredictions2nn$mean))
  
  
  vinenn1.Y1_r <- nn.Y1.actual-vinepredictions1nn$mean
  vinenn1.Y1_CL[q] <- mean(vinenn1.Y1_r)
  vinenn1.Y1_LCL[q] <- mean(vinenn1.Y1_r) - 3*sd(vinenn1.Y1_r)
  vinenn1.Y1_UCL[q] <- mean(vinenn1.Y1_r) + 3*sd(vinenn1.Y1_r)
  vinenn1.Y1_coverage[q] <- mean(vinenn1.Y1_r > vinenn1.Y1_LCL[q] & vinenn1.Y1_r < vinenn1.Y1_UCL[q])		
  vinenn1.Y1_check0 <- which(vinenn1.Y1_r < vinenn1.Y1_LCL[q] | vinenn1.Y1_r > vinenn1.Y1_UCL[q])
  if(length(vinenn1.Y1_check0)!=0L)	vinenn1.Y1_RL0 <- c(vinenn1.Y1_RL0, min(vinenn1.Y1_check0))
  
  vinenn1.Y2_r <- nn.Y2.actual-vinepredictions2nn$mean
  vinenn1.Y2_CL[q] <- mean(vinenn1.Y2_r)
  vinenn1.Y2_LCL[q] <- mean(vinenn1.Y2_r) - 3*sd(vinenn1.Y2_r)
  vinenn1.Y2_UCL[q] <- mean(vinenn1.Y2_r) + 3*sd(vinenn1.Y2_r)
  vinenn1.Y2_coverage[q] <- mean(vinenn1.Y2_r > vinenn1.Y2_LCL[q] & vinenn1.Y2_r < vinenn1.Y2_UCL[q])		
  vinenn1.Y2_check0 <- which(vinenn1.Y2_r < vinenn1.Y2_LCL[q] | vinenn1.Y2_r > vinenn1.Y2_UCL[q])
  if(length(vinenn1.Y2_check0)!=0L)	vinenn1.Y2_RL0 <- c(vinenn1.Y2_RL0, min(vinenn1.Y2_check0))
  
  
  dl.Y1 <- neuralnet(Y1 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  dl.Y1.results <- compute(dl.Y1, testNN)
  dlY1.results <- data.frame(actual = testNN$Y1, prediction = dl.Y1.results$net.result)
  
  dl.Y1.predicted=dlY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  dl.Y1.actual=dlY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
  
  dl.Y2 <- neuralnet(Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
  dl.Y2.results <- compute(dl.Y2, testNN)
  dlY2.results <- data.frame(actual = testNN$Y2, prediction = dl.Y2.results$net.result)
  
  dl.Y2.predicted=dlY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  dl.Y2.actual=dlY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
  
  
  RMSE.DL1[q,1]<- sqrt((sum(dl.Y1.actual-dl.Y1.predicted)^2)/length(dl.Y1.actual))
  RMSE.DL1[q,2]<- sqrt((sum(dl.Y2.actual-dl.Y2.predicted)^2)/length(dl.Y2.actual))
  
  WMAPE.DL1[q,1] <- sum(abs(dl.Y1.actual-dl.Y1.predicted))/sum(dl.Y1.actual)
  WMAPE.DL1[q,2] <- sum(abs(dl.Y2.actual-dl.Y2.predicted))/sum(dl.Y2.actual)
  
  MAD.DL1[q,1] <- mean(abs(dl.Y1.actual-dl.Y1.predicted))
  MAD.DL1[q,2] <- mean(abs(dl.Y2.actual-dl.Y2.predicted))
  
  
  dl1.Y1_r <- dl.Y1.actual-dl.Y1.predicted
  dl1.Y1_CL[q] <- mean(dl1.Y1_r)
  dl1.Y1_LCL[q] <- mean(dl1.Y1_r) - 3*sd(dl1.Y1_r)
  dl1.Y1_UCL[q] <- mean(dl1.Y1_r) + 3*sd(dl1.Y1_r)
  dl1.Y1_coverage[q] <- mean(dl1.Y1_r > dl1.Y1_LCL[q] & dl1.Y1_r < dl1.Y1_UCL[q])		
  dl1.Y1_check0 <- which(dl1.Y1_r < dl1.Y1_LCL[q] | dl1.Y1_r > dl1.Y1_UCL[q])
  if(length(dl1.Y1_check0)!=0L)	dl1.Y1_RL0 <- c(dl1.Y1_RL0, min(dl1.Y1_check0))
  
  dl1.Y2_r <- dl.Y2.actual-dl.Y2.predicted
  dl1.Y2_CL[q] <- mean(dl1.Y2_r)
  dl1.Y2_LCL[q] <- mean(dl1.Y2_r) - 3*sd(dl1.Y2_r)
  dl1.Y2_UCL[q] <- mean(dl1.Y2_r) + 3*sd(dl1.Y2_r)
  dl1.Y2_coverage[q] <- mean(dl1.Y2_r > dl1.Y2_LCL[q] & dl1.Y2_r < dl1.Y2_UCL[q])		
  dl1.Y2_check0 <- which(dl1.Y2_r < dl1.Y2_LCL[q] | dl1.Y2_r > dl1.Y2_UCL[q])
  if(length(dl1.Y2_check0)!=0L)	dl1.Y2_RL0 <- c(dl1.Y2_RL0, min(dl1.Y2_check0))
  
  
  ####
  
  
  
  dl1_data<-cbind(dl.Y1.predicted, dl.Y2.predicted)
  y1 = dl1_data[,1]
  y2 = dl1_data[,2]
  data<-data.frame(cbind(y1, y2))
  

  
  # fit vine regression model 
  
  library(vinereg)
  
  vy1.2.00<-vinereg( y1~y2, data = data)
  vy2.1.00<-vinereg( y2~y1, data = data)
  
  vinepredictions1dl <-predict(vy1.2.00, newdata = data, alpha = NA)
  vinepredictions2dl <-predict(vy2.1.00, newdata = data, alpha = NA)
  
  
  RMSE.vine.DL1[q,1]<- sqrt(mean((dl.Y1.actual - vinepredictions1dl$mean)^2))
  RMSE.vine.DL1[q,2]<- sqrt(mean((dl.Y2.actual - vinepredictions2dl$mean)^2))
  
  WMAPE.vine.DL1[q,1] <- sum(abs(dl.Y1.actual - vinepredictions1dl$mean))/sum(dl.Y1.actual)
  WMAPE.vine.DL1[q,2] <- sum(abs(dl.Y2.actual - vinepredictions2dl$mean))/sum(dl.Y2.actual)
  
  MAD.vine.DL1[q,1] <- mean(abs(dl.Y1.actual - vinepredictions1dl$mean))
  MAD.vine.DL1[q,2] <- mean(abs(dl.Y2.actual - vinepredictions2dl$mean))
  
  
  vinedl1.Y1_r <- dl.Y1.actual-vinepredictions1dl$mean
  vinedl1.Y1_CL[q] <- mean(vinedl1.Y1_r)
  vinedl1.Y1_LCL[q] <- mean(vinedl1.Y1_r) - 3*sd(vinedl1.Y1_r)
  vinedl1.Y1_UCL[q] <- mean(vinedl1.Y1_r) + 3*sd(vinedl1.Y1_r)
  vinedl1.Y1_coverage[q] <- mean(vinedl1.Y1_r > vinedl1.Y1_LCL[q] & vinedl1.Y1_r < vinedl1.Y1_UCL[q])		
  vinedl1.Y1_check0 <- which(vinedl1.Y1_r < vinedl1.Y1_LCL[q] | vinedl1.Y1_r > vinedl1.Y1_UCL[q])
  if(length(vinedl1.Y1_check0)!=0L)	vinedl1.Y1_RL0 <- c(vinedl1.Y1_RL0, min(vinedl1.Y1_check0))
  
  vinedl1.Y2_r <- dl.Y2.actual-vinepredictions2dl$mean
  vinedl1.Y2_CL[q] <- mean(vinedl1.Y2_r)
  vinedl1.Y2_LCL[q] <- mean(vinedl1.Y2_r) - 3*sd(vinedl1.Y2_r)
  vinedl1.Y2_UCL[q] <- mean(vinedl1.Y2_r) + 3*sd(vinedl1.Y2_r)
  vinedl1.Y2_coverage[q] <- mean(vinedl1.Y2_r > vinedl1.Y2_LCL[q] & vinedl1.Y2_r < vinedl1.Y2_UCL[q])		
  vinedl1.Y2_check0 <- which(vinedl1.Y2_r < vinedl1.Y2_LCL[q] | vinedl1.Y2_r > vinedl1.Y2_UCL[q])
  if(length(vinedl1.Y2_check0)!=0L)	vinedl1.Y2_RL0 <- c(vinedl1.Y2_RL0, min(vinedl1.Y2_check0))
  
  
  print(q)
}





#x1 <- RMSE.zip[,1]
x2 <- RMSE.DL[,1]
x3 <- RMSE.NN[,1]
x4 <- RMSE.DL1[,1]
x5 <- RMSE.NN1[,1]
x8<-RMSE.vine.DL1[,1]
x9<-RMSE.vine.NN1[,1]

boxplot(x2, x3, x4, x5, x8, x9, 
        names=c("DL", "NN", "DL1", "NN1", "VINEDL1", "VINENN1"), ylab = "RMSE",
        col="gold")
title("RMSE with Real Data (Y1)")

#w1 <- WMAPE.zip[,1]
w2 <- WMAPE.DL[,1]
w3 <- WMAPE.NN[,1]
w4 <- WMAPE.DL1[,1]
w5 <- WMAPE.NN1[,1]
w8<-WMAPE.vine.DL1[,1]
w9<-WMAPE.vine.NN1[,1]

boxplot(w2, w3, w4, w5, w8, w9, 
        names=c("DL", "NN", "DL1", "NN1", "VINEDL1", "VINENN1"), ylab = "WMAPE",
        col="gold")
title("WMAPE with Real Data (Y1)")


#z1 <- MAD.zip[,1]
z2 <- MAD.DL[,1]
z3 <- MAD.NN[,1]
z4 <- MAD.DL1[,1]
z5 <- MAD.NN1[,1]
z8<-MAD.vine.DL1[,1]
z9<-MAD.vine.NN1[,1]

boxplot(z2, z3, z4, z5, z8, z9, 
        names=c("DL", "NN", "DL1", "NN1", "VINEDL1", "VINENN1"), ylab = "MAD",
        col="gold")
title("MAD with Real Data (Y1)")



#x11 <- RMSE.zip[,2]
x12 <- RMSE.DL[,2]
x13 <- RMSE.NN[,2]
x14 <- RMSE.DL1[,2]
x15 <- RMSE.NN1[,2]
x18<-RMSE.vine.DL1[,2]
x19<-RMSE.vine.NN1[,2]

boxplot(x12, x13, x14, x15, x18, x19, 
        names=c("DL", "NN", "DL1", "NN1", "VINEDL1", "VINENN1"), ylab = "RMSE",
        col="gold")
title("RMSE with Real Data (Y2)")

#w11 <- WMAPE.zip[,2]
w12 <- WMAPE.DL[,2]
w13 <- WMAPE.NN[,2]
w14 <- WMAPE.DL1[,2]
w15 <- WMAPE.NN1[,2]
w18<-WMAPE.vine.DL1[,2]
w19<-WMAPE.vine.NN1[,2]

boxplot(w12, w13, w14, w15, w18, w19, 
        names=c("DL", "NN", "DL1", "NN1", "VINEDL1", "VINENN1"), ylab = "WMAPE",
        col="gold")
title("WMAPE with Real Data (Y2)")


#z11 <- MAD.zip[,2]
z12 <- MAD.DL[,2]
z13 <- MAD.NN[,2]
z14 <- MAD.DL1[,2]
z15 <- MAD.NN1[,2]
z18<-MAD.vine.DL1[,2]
z19<-MAD.vine.NN1[,2]

boxplot(z12, z13, z14, z15, z18, z19, 
        names=c("DL", "NN", "DL1", "NN1","VINEDL1", "VINENN1"), ylab = "MAD",
        col="gold")
title("MAD with Real Data (Y2)")


summary(RMSE.DL)
IQR(RMSE.DL[,1])
IQR(RMSE.DL[,2])

summary(RMSE.NN)
IQR(RMSE.NN[,1])
IQR(RMSE.NN[,2])

summary(RMSE.DL1)
IQR(RMSE.DL1[,1])
IQR(RMSE.DL1[,2])

summary(RMSE.NN1)
IQR(RMSE.NN1[,1])
IQR(RMSE.NN1[,2])


summary(RMSE.gcmr.DL1)
IQR(RMSE.gcmr.DL1[,1])
IQR(RMSE.gcmr.DL1[,2])

summary(RMSE.gcmr.NN1)
IQR(RMSE.gcmr.NN1[,1])
IQR(RMSE.gcmr.NN1[,2])


summary(RMSE.vine.DL1)
IQR(RMSE.vine.DL1[,1])
IQR(RMSE.vine.DL1[,2])

summary(RMSE.vine.NN1)
IQR(RMSE.vine.NN1[,1])
IQR(RMSE.vine.NN1[,2])


summary(WMAPE.DL)
IQR(WMAPE.DL[,1])
IQR(WMAPE.DL[,2])

summary(WMAPE.NN)
IQR(WMAPE.NN[,1])
IQR(WMAPE.NN[,2])

summary(WMAPE.DL1)
IQR(WMAPE.DL1[,1])
IQR(WMAPE.DL1[,2])

summary(WMAPE.NN1)
IQR(WMAPE.NN1[,1])
IQR(WMAPE.NN1[,2])


summary(WMAPE.gcmr.DL1)
IQR(WMAPE.gcmr.DL1[,1])
IQR(WMAPE.gcmr.DL1[,2])

summary(WMAPE.gcmr.NN1)
IQR(WMAPE.gcmr.NN1[,1])
IQR(WMAPE.gcmr.NN1[,2])


summary(WMAPE.vine.DL1)
IQR(WMAPE.vine.DL1[,1])
IQR(WMAPE.vine.DL1[,2])

summary(WMAPE.vine.NN1)
IQR(WMAPE.vine.NN1[,1])
IQR(WMAPE.vine.NN1[,2])


summary(MAD.DL)
IQR(MAD.DL[,1])
IQR(MAD.DL[,2])

summary(MAD.NN)
IQR(MAD.NN[,1])
IQR(MAD.NN[,2])

summary(MAD.DL1)
IQR(MAD.DL1[,1])
IQR(MAD.DL1[,2])

summary(MAD.NN1)
IQR(MAD.NN1[,1])
IQR(MAD.NN1[,2])


summary(MAD.gcmr.DL1)
IQR(MAD.gcmr.DL1[,1])
IQR(MAD.gcmr.DL1[,2])

summary(MAD.gcmr.NN1)
IQR(MAD.gcmr.NN1[,1])
IQR(MAD.gcmr.NN1[,2])


summary(MAD.vine.DL1)
IQR(MAD.vine.DL1[,1])
IQR(MAD.vine.DL1[,2])

summary(MAD.vine.NN1)
IQR(MAD.vine.NN1[,1])
IQR(MAD.vine.NN1[,2])




#ARL0

mean(dl.Y1_LCL)
mean(dl.Y1_UCL)
mean(dl.Y1_CL)
mean(dl.Y1_coverage)

mean(dl.Y2_LCL)
mean(dl.Y2_UCL)
mean(dl.Y2_CL)
mean(dl.Y2_coverage)


mean(nn.Y1_LCL)
mean(nn.Y1_UCL)
mean(nn.Y1_CL)
mean(nn.Y1_coverage)

mean(nn.Y2_LCL)
mean(nn.Y2_UCL)
mean(nn.Y2_CL)
mean(nn.Y2_coverage)


mean(dl1.Y1_LCL)
mean(dl1.Y1_UCL)
mean(dl1.Y1_CL)
mean(dl1.Y1_coverage)

mean(dl1.Y2_LCL)
mean(dl1.Y2_UCL)
mean(dl1.Y2_CL)
mean(dl1.Y2_coverage)


mean(nn1.Y1_LCL)
mean(nn1.Y1_UCL)
mean(nn1.Y1_CL)
mean(nn1.Y1_coverage)

mean(nn1.Y2_LCL)
mean(nn1.Y2_UCL)
mean(nn1.Y2_CL)
mean(nn1.Y2_coverage)



mean(gcdl1.Y1_LCL)
mean(gcdl1.Y1_UCL)
mean(gcdl1.Y1_CL)
mean(gcdl1.Y1_coverage)

mean(gcdl1.Y2_LCL)
mean(gcdl1.Y2_UCL)
mean(gcdl1.Y2_CL)
mean(gcdl1.Y2_coverage)


mean(gcnn1.Y1_LCL)
mean(gcnn1.Y1_UCL)
mean(gcnn1.Y1_CL)
mean(gcnn1.Y1_coverage)

mean(gcnn1.Y2_LCL)
mean(gcnn1.Y2_UCL)
mean(gcnn1.Y2_CL)
mean(gcnn1.Y2_coverage)


mean(vinedl1.Y1_LCL)
mean(vinedl1.Y1_UCL)
mean(vinedl1.Y1_CL)
mean(vinedl1.Y1_coverage)

mean(vinedl1.Y2_LCL)
mean(vinedl1.Y2_UCL)
mean(vinedl1.Y2_CL)
mean(vinedl1.Y2_coverage)


mean(vinenn1.Y1_LCL)
mean(vinenn1.Y1_UCL)
mean(vinenn1.Y1_CL)
mean(vinenn1.Y1_coverage)

mean(vinenn1.Y2_LCL)
mean(vinenn1.Y2_UCL)
mean(vinenn1.Y2_CL)
mean(vinenn1.Y2_coverage)





### Real data cusum figures

library(neuralnet)
library(copula)
set.seed(1)
library(CASdatasets)
data(ausprivauto0405)
#make factor variables levels(ausprivauto0405$VehAge)
ausprivauto0405$VehAge <- as.numeric(ordered(ausprivauto0405$VehAge, levels=c("old cars", "oldest cars", "young cars", "youngest cars")))
ausprivauto0405$VehBody<- as.numeric(ordered(ausprivauto0405$VehBody, levels=  c("Bus", "Convertible", "Coupe","Hardtop", "Hatchback", "Minibus", "Motorized caravan", "Panel van","Roadster", "Sedan","Station wagon","Truck", "Utility")))        
ausprivauto0405$Gender <- as.numeric(ordered(ausprivauto0405$Gender, levels=c("Female", "Male")))
ausprivauto0405$DrivAge<- as.numeric(ordered(ausprivauto0405$DrivAge, levels=c("old people", "older work. people", "oldest people", "working people", "young people", "youngest people")))



data1 <- data.frame(ausprivauto0405[,c(7,8,1,2,3,4,5,6,9)])
#samplesize<-600
#data1<-data[sample(nrow(data), samplesize), ]
colnames(data1)<-c("Y1", "Y2", "Exposure", "VehValue", "VehAge", "VehBody", "Gender", "DrivAge", "ClaimAmount")


#  cor(data1)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

sdata <- as.data.frame(lapply(data1, normalize))
colnames(sdata)<-c("Y1", "Y2", "Exposure", "VehValue", "VehAge", "VehBody", "Gender", "DrivAge", "ClaimAmount")
head(sdata)

samplesize = 0.80 * nrow(sdata)

# nrow(sdata) [1] 67856
# 0.80 * nrow(sdata) [1] 54284.8=54285

index = sample(1:nrow(sdata),samplesize )



#### NN and DL Parts ###

trainNN = sdata[index,]
testNN = sdata[-index,]
datatest = sdata[-index,]



x_tr <- trainNN[,-c(1:2)]
y_tr <- trainNN[,c(1:2)]
x_te <- testNN[,-c(1:2)]
y_te <- testNN[,c(1:2)]




nn <- neuralnet(as.formula(Y1 + Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge),
                data=trainNN, hidden=5, act.fct = "logistic", linear.output=FALSE, threshold=0.1)

nn.results <- compute(nn, testNN)
nnresults <- data.frame(actual = y_te, prediction = nn.results$net.result)


predictednn.1=nnresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
actualnn.Y1=nnresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

predictednn.2=nnresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
actualnn.Y2=nnresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 


nn.Y1_r <- actualnn.Y1-predictednn.1

nn.Y2_r <- actualnn.Y2-predictednn.2



nn.Y1_r_out=nn.Y1_r
nn.Y2_r_out=nn.Y2_r



# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###



library(qcc)

length(nn.Y1_r_out)
#q1 = cusum(nn.Y1_r_out)
q1 = cusum(nn.Y1_r_out[1:300], newdata=nn.Y1_r_out[301:400],
           add.stats=FALSE,
           title="Multivariate Neural Network of Y1", xlab="", ylab="")

q2 = cusum(nn.Y2_r_out[1:300], newdata=nn.Y2_r_out[301:400],
           add.stats=FALSE,
           title="Multivariate Neural Network of Y2", xlab="", ylab="")



###



dl <- neuralnet(as.formula(Y1 + Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge),
                data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)

dl.results <- compute(dl, testNN)
dlresults <- data.frame(actual = y_te, prediction = dl.results$net.result)


predicteddl.1=dlresults$prediction.1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
actualdl.Y1=dlresults$actual.Y1 * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

predicteddl.2=dlresults$prediction.2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 
actualdl.Y2=dlresults$actual.Y2 * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2) 



dl.Y1_r <- actualdl.Y1-predicteddl.1
dl.Y2_r <- actualdl.Y2-predicteddl.2


dl.Y1_r_out=dl.Y1_r
dl.Y2_r_out=dl.Y2_r


# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###


library(qcc)

length(dl.Y1_r_out)

q1 = cusum(dl.Y1_r_out[1:300], newdata=dl.Y1_r_out[301:400],
           add.stats=FALSE,
           title="Multivariate Deep Learning of Y1", xlab="", ylab="")

q2 = cusum(dl.Y2_r_out[1:300], newdata=dl.Y2_r_out[301:400],
           add.stats=FALSE,
           title="Multivariate Deep Learning of Y2", xlab="", ylab="")


#####


nn.Y1 <- neuralnet(Y1 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
nn.Y1.results <- compute(nn.Y1, testNN)
nnY1.results <- data.frame(actual = testNN$Y1, prediction = nn.Y1.results$net.result)

nn.Y1.predicted=nnY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
nn.Y1.actual=nnY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

nn.Y2 <- neuralnet(Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=5, act.fct = "logistic", linear.output=TRUE, threshold=0.3)
nn.Y2.results <- compute(nn.Y2, testNN)
nnY2.results <- data.frame(actual = testNN$Y2, prediction = nn.Y2.results$net.result)

nn.Y2.predicted=nnY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
nn.Y2.actual=nnY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)


nn1.Y1_r <- nn.Y1.actual-nn.Y1.predicted
nn1.Y2_r <- nn.Y2.actual-nn.Y2.predicted


nn1.Y1_r_out=nn1.Y1_r
nn1.Y2_r_out=nn1.Y2_r


# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(nn1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###


library(qcc)

length(nn1.Y1_r_out)
#q1 = cusum(nn1.Y1_r_out)
q1 = cusum(nn1.Y1_r_out[1:300], newdata=nn1.Y1_r_out[301:400],
           add.stats=FALSE,
           title="Univariate Neural Network of Y1", xlab="", ylab="")

q2 = cusum(nn1.Y2_r_out[1:300], newdata=nn1.Y2_r_out[301:400],
           add.stats=FALSE,
           title="Univariate Neural Network of Y2", xlab="", ylab="")


###



dl.Y1 <- neuralnet(Y1 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
dl.Y1.results <- compute(dl.Y1, testNN)
dlY1.results <- data.frame(actual = testNN$Y1, prediction = dl.Y1.results$net.result)

dl.Y1.predicted=dlY1.results$prediction * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)
dl.Y1.actual=dlY1.results$actual * (max(data1$Y1)-min(data1$Y1)) + min(data1$Y1)

dl.Y2 <- neuralnet(Y2 ~ Exposure+VehValue+VehAge+VehBody+Gender+DrivAge, data=trainNN, hidden=c(5,5), act.fct = "logistic", linear.output=FALSE, threshold=0.1)
dl.Y2.results <- compute(dl.Y2, testNN)
dlY2.results <- data.frame(actual = testNN$Y2, prediction = dl.Y2.results$net.result)

dl.Y2.predicted=dlY2.results$prediction * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)
dl.Y2.actual=dlY2.results$actual * (max(data1$Y2)-min(data1$Y2)) + min(data1$Y2)


dl1.Y1_r <- dl.Y1.actual-dl.Y1.predicted
dl1.Y2_r <- dl.Y2.actual-dl.Y2.predicted

dl1.Y1_r_out=dl1.Y1_r 
dl1.Y2_r_out=dl1.Y2_r




# Time-Varying t-Copula

library(rmgarch) 

p1 = dl1.Y1_r_out
p2 = dl1.Y2_r_out
p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)

rhoyx1<-mat1[,1]


matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(1,1)-sGARCH(1,1) of Univariate Deep Learning of Y1 and Y2")



# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)

###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(dl1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}


arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###



library(qcc)

length(dl1.Y1_r_out)

q1 = cusum(dl1.Y1_r_out[1:300], newdata=dl1.Y1_r_out[301:400],
           add.stats=FALSE,
           title="Univariate Deep Learning of Y1", xlab="", ylab="")

q2 = cusum(dl1.Y2_r_out[1:300], newdata=dl1.Y2_r_out[301:400],
           add.stats=FALSE,
           title="Univariate Deep Learning of Y2", xlab="", ylab="")



#####

####



dl1_data<-cbind(dl.Y1.predicted, dl.Y2.predicted)
y1 = dl1_data[,1]
y2 = dl1_data[,2]
data<-data.frame(cbind(y1, y2))

#####

#####

# fit vine regression model 

library(vinereg)

vy1.2.00<-vinereg( y1~y2, data = data)
vy2.1.00<-vinereg( y2~y1, data = data)

vinepredictions1dl <-predict(vy1.2.00, newdata = data, alpha = NA)
vinepredictions2dl <-predict(vy2.1.00, newdata = data, alpha = NA)


vinedl1.Y1_r <- dl.Y1.actual - vinepredictions1dl$mean
vinedl1.Y2_r <- dl.Y2.actual - vinepredictions2dl$mean


vinedl1.Y1_r_out<-vinedl1.Y1_r

vinedl1.Y2_r_out<-vinedl1.Y2_r


# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinedl1.Y1_r_out
p2 = vinedl1.Y2_r_out
p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)


rhoyx1<-mat1[3950:4000,1]

matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(1,1)-sGARCH(1,1) of Vine Copula Deep Learning of Y1 and Y2")

comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)





# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinedl1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinedl1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###



length(vinedl1.Y1_r_out)

q1 = cusum(vinedl1.Y1_r_out[1:300], newdata=vinedl1.Y1_r_out[301:400],
           add.stats=FALSE,
           title="Vine Copula Deep Learning of Y1", xlab="", ylab="")

q2 = cusum(vinedl1.Y2_r_out[1:300], newdata=vinedl1.Y2_r_out[301:400],
           add.stats=FALSE,
           title="Vine Copula Deep Learning of Y2", xlab="", ylab="")


####



nn1_data<-cbind(nn.Y1.predicted, nn.Y2.predicted)
y1 = nn1_data[,1]
y2 = nn1_data[,2]
datann<-data.frame(cbind(y1, y2))

#####

# fit vine regression model 

library(vinereg)

vy1.2.00<-vinereg( y1~y2, data = datann)
vy2.1.00<-vinereg( y2~y1, data = datann)

vinepredictions1nn <-predict(vy1.2.00, newdata = datann, alpha = NA)
vinepredictions2nn <-predict(vy2.1.00, newdata = datann, alpha = NA)


vinenn1.Y1_r <- nn.Y1.actual - vinepredictions1nn$mean
vinenn1.Y2_r <- nn.Y2.actual - vinepredictions2nn$mean


vinenn1.Y1_r_out<-vinenn1.Y1_r

vinenn1.Y2_r_out<-vinenn1.Y2_r


# Time-Varying Vine Copula

library(rmgarch) 

p1 = vinenn1.Y1_r_out
p2 = vinenn1.Y2_r_out
p = cbind(p1,p2)
Dat = p

uspec = ugarchspec(mean.model = list(armaOrder = c(1,1)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"),  distribution.model = "jsu") 

spec1 = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,  lag.criterion = c("AIC", "HQ", "SC", "FPE"), external.regressors = NULL, robust.control = list("gamma" = 0.25, "delta" = 0.01, "nc" = 10, "ns" = 500), dccOrder = c(1,1), asymmetric = FALSE, distribution.model = list(copula = c("mvnorm", "mvt")[2],  method = c("Kendall", "ML")[2], time.varying = TRUE,  transformation = c("parametric", "empirical", "spd")[1])) 

fit1 = cgarchfit(spec1, data = Dat, parallel = parallel, parallel.control = parallel.control,  fit.control = list(eval.se=TRUE)) 
show(fit1) 
A<-data.matrix(rcor(fit1))
head(A)
B<-subset(A, A[,1]!="1")
head(B)
n<-length(B)/2
mat1 <- matrix(B, nrow = n, ncol = 2, byrow=T)

rhoyx1<-mat1[1001:1050,1]


matplot(rhoyx1, type='l', ylab="Correlation",main="Time Varying t-copula Correlation-ARMA(1,1)-sGARCH(1,1) of Vine Copula Neural Network of Y1 and Y2")

comp_r<-rhoyx1
comp_CL <- mean(comp_r)
comp_LCL <- mean(comp_r) - sd(comp_r)
comp_UCL <- mean(comp_r) + sd(comp_r)

#plot(comp_r, ylim=c(-5,5), ylab="Residual", main="DL based r-Control Chart with 1*sigma")
abline(a = comp_LCL, b=0, col = "red")
abline(a = comp_CL, b=0, col = "blue")
abline(a = comp_UCL, b=0, col = "red")

nsim<-length(comp_r)
axis(side=4, at =c(comp_CL, comp_LCL, comp_UCL), labels = c("CL", "LCL", "UCL"), cex=0.5, cex.axis =1.5)
text(c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL],
     comp_r[comp_r > comp_UCL | comp_r < comp_LCL],
     c(1:nsim)[comp_r > comp_UCL | comp_r < comp_LCL], cex=1.5)

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinenn1.Y1_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###

# calculate the ARL of CUSUM (S=1000,300,500, n=500,100,200)
DERA_Avg<-as.matrix(vinenn1.Y2_r_out)
arl.cusum = function(S,n){ 
  z=rep(0,n)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/c[n]-j/n;}
  z[k]=1;
  for(i in 1:( n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(ARL=mean(z))
}



arl.cusum(S=10000 ,n=100)
arl.cusum(S=10000 ,n=200)
arl.cusum(S=10000 ,n=300)


# calculate the ARL of EWMA (S=1000,300,500, n=500,100,200, r=0.2, 0.6, 0.9)
arl.ewma = function(S,n,r){
  z=rep(0,S)
  for(k in 1:S)
  {a=sample(DERA_Avg, n, replace = FALSE);
  c=a;d=a;c[1]=a[1]^2;
  for(i in 1:(n-1))
  {c[i+1]=c[i]+a[i+1]^2;}
  for(j in 1:n)
  {d[j]=c[j]/j;}
  g=a;h=a;h[1]=sqrt(2-r)/sqrt(r*(1-(1-r)^2))*r*log(d[1]^2);
  for(i in 2:n){g[1]=r*(1-r)^(i-1)*log(d[1]^2);
  for(j in 1:(i-1)) g[j+1]=r*(1-r)^(i-j-1)*log(d[j+1]^2)+g[j];
  h[i]=sqrt(2-r)/sqrt(r*(1-(1-r)^(2*i)))*g[i]}
  z[k]=round(n/3);
  for(i in round(n/3):(n-1))
  { if (abs(d[i+1])>=abs(d[i]))
  {z[k]=i+1;}
    else
    {d[i+1]=d[i]}
  }}
  list(r=r,ARL=mean(z))
}

arl.ewma(S=10000, n=100, r=0.2)
arl.ewma(S=10000, n=100, r=0.6)
arl.ewma(S=10000, n=100, r=0.9)
arl.ewma(S=10000, n=200, r=0.2)
arl.ewma(S=10000, n=200, r=0.6)
arl.ewma(S=10000, n=200, r=0.9)
arl.ewma(S=10000, n=300, r=0.2)
arl.ewma(S=10000, n=300, r=0.6)
arl.ewma(S=10000, n=300, r=0.9)


###


length(vinenn1.Y1_r_out)

q1 = cusum(vinenn1.Y1_r_out[1:300], newdata=vinenn1.Y1_r_out[301:400],
           add.stats=FALSE,
           title="Vine Copula Neural Network of Y1", xlab="", ylab="")

q2 = cusum(vinedl1.Y2_r_out[1:300], newdata=vinedl1.Y2_r_out[301:400],
           add.stats=FALSE,
           title="Vine Copula Neural Network of Y2", xlab="", ylab="")



