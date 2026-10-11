# Week 3 Statistical Analysis and Predictive Modeling
# Install once: install.packages(c("dplyr","caret"))
library(dplyr)
library(caret)
set.seed(123)
url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
raw <- read.csv(url, na.strings=c("", "NA"))
data <- raw %>% select(Survived,Pclass,Sex,Age,Fare) %>%
  mutate(Sex=factor(Sex), Pclass=factor(Pclass))
data$Age[is.na(data$Age)] <- median(data$Age,na.rm=TRUE)
data <- na.omit(data)
data$SurvivedFactor <- factor(data$Survived, levels=c(0,1),
                              labels=c("Not survived","Survived"))
print(summary(data)); str(data)

# H0: sex and survival are independent.
sex_table <- table(data$Sex,data$SurvivedFactor)
print(sex_table); print(chisq.test(sex_table))
print(wilcox.test(Age ~ SurvivedFactor,data=data,exact=FALSE))
print(cor.test(data$Age,data$Fare,method="spearman",exact=FALSE))

idx <- createDataPartition(data$SurvivedFactor,p=.8,list=FALSE)
train_data <- data[idx,]; test_data <- data[-idx,]
model <- glm(Survived ~ Pclass+Sex+Age+Fare,data=train_data,family=binomial)
print(summary(model))
prob <- predict(model,newdata=test_data,type="response")
pred <- factor(ifelse(prob>=.5,"Survived","Not survived"),
               levels=c("Not survived","Survived"))
print(confusionMatrix(pred,test_data$SurvivedFactor,positive="Survived"))

# Five-fold cross-validation using ROC AUC.
cv_data <- train_data
cv_data$SurvivedFactor <- relevel(cv_data$SurvivedFactor,ref="Survived")
ctrl <- trainControl(method="cv",number=5,classProbs=TRUE,
                     summaryFunction=twoClassSummary)
cv_model <- train(SurvivedFactor ~ Pclass+Sex+Age+Fare,data=cv_data,
                  method="glm",family=binomial,metric="ROC",trControl=ctrl)
print(cv_model); print(cv_model$results)

png("week3_model_diagnostics.png",width=1200,height=900,res=150)
par(mfrow=c(2,2)); plot(model); dev.off()
png("week3_confusion_matrix.png",width=1000,height=800,res=150)
fourfoldplot(confusionMatrix(pred,test_data$SurvivedFactor)$table,
             main="Test-set confusion matrix"); dev.off()
