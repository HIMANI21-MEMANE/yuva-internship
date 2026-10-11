# Week 4 Comprehensive Integrated Analysis
# Install once: install.packages(c("dplyr","ggplot2"))
library(dplyr)
library(ggplot2)
url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
raw <- read.csv(url,na.strings=c("", "NA"))
print(dim(raw)); str(raw); print(summary(raw))
missing_summary <- data.frame(Variable=names(raw),
  MissingCount=sapply(raw,function(x) sum(is.na(x))))
print(missing_summary)
write.csv(missing_summary,"week4_missing_value_summary.csv",row.names=FALSE)

data <- raw %>% select(Survived,Pclass,Sex,Age,Fare) %>%
  mutate(Pclass=factor(Pclass,levels=c(1,2,3),
                       labels=c("First","Second","Third")),
         Sex=factor(Sex))
data$Age[is.na(data$Age)] <- median(data$Age,na.rm=TRUE)
rates <- data %>% group_by(Pclass) %>%
  summarise(Passengers=n(),SurvivalRate=mean(Survived==1)*100,.groups="drop")
print(rates)
p <- ggplot(rates,aes(Pclass,SurvivalRate,group=1)) +
  geom_line()+geom_point(size=3)+
  geom_text(aes(label=paste0(round(SurvivalRate,1),"%")),vjust=-.8)+
  labs(title="Survival Rate by Passenger Class",x="Passenger class",
       y="Survival rate (%)")+theme_minimal()
ggsave("week4_survival_rate_by_class.png",p,width=7,height=4.5,dpi=180)
print(chisq.test(table(data$Sex,data$Survived)))
integrated_model <- glm(Survived ~ Pclass+Sex+Age+Fare,
                        data=data,family=binomial)
print(summary(integrated_model))
