---
title: "NeuNet Census"
author: "Enrique Otanez"
date: "4/8/2021"
output: word_document
---

```{r}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_knit$set(root.dir = normalizePath("D:/Templates/UW Stuff/Classes/MSBA/Classes/PM Stuff for me/NeuralNet"))
```

```{r}
library(DBI)
library(sqldf)
library(tidyverse)
```


```{r}
#Connecting R to PostgreSQL
setwd("D:/Templates/UW Stuff/Classes/MSBA/Classes/Q4 Models")
df <- read.csv("Config_File.csv")

con <- DBI::dbConnect(odbc::odbc(),
  driver = "PostgreSQL Unicode(x64)",
  database = "TEST",
  UID      = df$UID,
  PWD      = df$PWD,
  server = df$server,
  port = 5432)
```

```{r}
import.census <- dbGetQuery(con, 'select
     bg.bg_geo_id "Block Group ID"
     ,(select avg(bgs.score) from "BG_Score" as bgs where bgs.bg_geo_id=bg.bg_geo_id) "Average Block Group Score"
     ,max(case when dv.sid=\'pop\' then bgd.value Else 0 END) "Population"
     ,max(case when dv.sid=\'pop_MF_3MS\' then bgd.value Else 0 END) "Population: 3 Mile"
     ,max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) "Households: 3 Mile"
    ,max(case when dv.sid=\'M_0_5\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_0_5\' then bgd.value Else 0 END) "Kids under 5"
     ,round(cast((max(case when dv.sid=\'M_0_5\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_0_5\' then bgd.value Else 0 END)) /
        max(case when dv.sid=\'pop\' then bgd.value Else null END) as numeric),3) "Percent Kids under 5"
    ,max(case when dv.sid=\'M_0_5_3MS\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_0_5_3MS\' then bgd.value Else 0 END) "Kids under 5: 3 Mile"
     ,round(cast((max(case when dv.sid=\'M_0_5_3MS\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_0_5_3MS\' then bgd.value Else 0 END)) /
        max(case when dv.sid=\'pop_MF_3MS\' then bgd.value Else null END) as numeric),3) "Percent Kids under 5: 3 Mile"
    ,max(case when dv.sid=\'M_5_9\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_5_9\' then bgd.value Else 0 END) "Kids 5 to 9"
     ,round(cast((max(case when dv.sid=\'M_5_9\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_5_9\' then bgd.value Else 0 END)) /
        max(case when dv.sid=\'pop\' then bgd.value Else null END) as numeric),3) "Percent Kids 5 to 9"
    ,max(case when dv.sid=\'M_5_9_3MS\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_5_9_3MS\' then bgd.value Else 0 END) "Kids 5 to 9: 3 Mile"
    ,round(cast((max(case when dv.sid=\'M_5_9_3MS\' then bgd.value Else 0 END)+max(case when dv.sid=\'F_5_9_3MS\' then bgd.value Else 0 END)) /
         max(case when dv.sid=\'pop_MF_3MS\' then bgd.value Else null END) as numeric),3)  "Percent Kids 5 to 9: 3 Mile"
    ,max(case when dv.sid=\'avg_age\' then bgd.value Else 0 END) "Average Age"
    ,round(cast(sum(case when dv.sid in(\'hi_0_10_3MS\',\'hi_10_15_3MS\',\'hi_15_20_3MS\',\'hi_20_25_3MS\',\'hi_25_30_3MS\',\'hi_30_35_3MS\',\'hi_35_40_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income under 40K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_40_45_3MS\',\'hi_45_50_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 40K to 50K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_50_60_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 50K to 60K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_60_75_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 60K to 75K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_75_100_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 75K to 100K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_100_125_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 100K to 125K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_125_150_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 125K to 150K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_150_200_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 150K to 200K: 3 Mile"
    ,round(cast(sum(case when dv.sid in(\'hi_200_999_3MS\') then bgd.value  else 0 END) /
      max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END) as numeric),3) "Household income 200K+: 3 Mile"
from "Block_Group" as bg
left join "BG_Data" as bgd on bg.bg_geo_id = bgd.bg_geo_id
inner join "Demo_Var" as dv on dv.full_variable_id=bgd.variable_id
group by bg.bg_geo_id
having
max(case when dv.sid=\'pop\' then bgd.value Else 0 END) > 0
and max(case when dv.sid=\'hi_tot_3MS\' then bgd.value Else 0 END)>0')
import.census
```
```{r}
#We need to query the variables used for the score as well as the score from the DB

census.scores <- sqldf('SELECT "Block Group ID" AS "Block.Group.ID", "Average Block Group Score" AS "Average.Block.Group.Score", Population, "Population: 3 Mile" AS "Population.3.Mile", "Households: 3 Mile" AS "Households.3.Mile", "Percent Kids under 5" AS "Percent.Kids.under.5", "Percent Kids under 5: 3 Mile" AS "Percent.Kids.under.5.3.Mile", "Percent Kids 5 to 9" AS "Percent.Kids.5.to.9", "Percent Kids 5 to 9: 3 Mile" AS "Percent.Kids.5.to.9.3 Mile", "Average Age" AS "Average.Age", "Household income under 40K: 3 Mile" AS "Household.income.under.40K.3Mile", "Household income 40K to 50K: 3 Mile" AS "Household.income.40K.to.50K.3.Mile", "Household income 50K to 60K: 3 Mile" AS "Household.income.50K.to.60K.3.Mile", "Household income 60K to 75K: 3 Mile" AS "Household.income.60K.to.75K.3.Mile", "Household income 75K to 100K: 3 Mile" AS "Household.income.75K.to.100K.3.Mile", "Household income 100K to 125K: 3 Mile" AS "Household.income.100:.125K.3.Mile", "Household income 125K to 150K: 3 Mile" AS "Household.income.125K.to.150K.3.Mile", "Household income 150K to 200K: 3 Mile" AS "Household.income.150K.to.200K.3.Mile", "Household income 200K+: 3 Mile"  AS "Household.income.200K+.3Mile"FROM "import.census" WHERE "Average Block Group Score" IS NOT NULL')

census.scores
#this is to make an export of the final model output
census.export <- census.scores
```

```{r}
#Change all variables into a single scale
#This is because percentages cannot compare to whole numbers and vice versa, its best to just normalize everything
min.max.norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#census.scores <- as.data.frame(lapply(census.scores[2:19], min.max.norm))
#census.scores

census.scores$Block.Group.ID <- as.numeric(census.scores$Block.Group.ID)

census.scores <- for (col in census.scores) {
                  if (col %in% c(census.scores$Block.Group.ID)) next
                    min.max.norm(census.scores)
}

help(col)
census.scores <- for (col in census.scores) {
  if (col != census.scores$Block.Group.ID)
    as.data.frame(min.max.norm(census.scores))
}

census.scores

census.scores$Population <- (census.scores$Population - min(census.scores$Population))/(max(census.scores$Population) - min(census.scores$Population))

census.scores$Population.3.Mile <- (census.scores$Population.3.Mile - min(census.scores$Population.3.Mile))/(max(census.scores$Population.3.Mile) - min(census.scores$Population.3.Mile))

census.scores$Households.3.Mile <- (census.scores$Households.3.Mile - min(census.scores$Households.3.Mile))/(max(census.scores$Households.3.Mile) - min(census.scores$Households.3.Mile))

census.scores$Percent.Kids.under.5 <- (census.scores$Percent.Kids.under.5 - min(census.scores$Percent.Kids.under.5))/(max(census.scores$Percent.Kids.under.5) - min(census.scores$Percent.Kids.under.5))

census.scores$Percent.Kids.under.5.3.Mile <- (census.scores$Percent.Kids.under.5.3.Mile - min(census.scores$Percent.Kids.under.5.3.Mile))/(max(census.scores$Percent.Kids.under.5.3.Mile) - min(census.scores$Percent.Kids.under.5.3.Mile))

census.scores$Percent.Kids.5.to.9 <- (census.scores$Percent.Kids.5.to.9 - min(census.scores$Percent.Kids.5.to.9))/(max(census.scores$Percent.Kids.5.to.9) - min(census.scores$Percent.Kids.5.to.9))

census.scores$`Percent Kids 5 to 9: 3 Mile` <- (census.scores$`Percent Kids 5 to 9: 3 Mile` - min(census.scores$`Percent Kids 5 to 9: 3 Mile`))/(max(census.scores$`Percent Kids 5 to 9: 3 Mile`) - min(census.scores$`Percent Kids 5 to 9: 3 Mile`))

census.scores$`Average Age` <- (census.scores$`Average Age` - min(census.scores$`Average Age`))/(max(census.scores$`Average Age`) - min(census.scores$`Average Age`))

census.scores$`Household income under 40K: 3 Mile` <- (census.scores$`Household income under 40K: 3 Mile` - min(census.scores$`Household income under 40K: 3 Mile`))/(max(census.scores$`Household income under 40K: 3 Mile`) - min(census.scores$`Household income under 40K: 3 Mile`))

census.scores$`Household income 40K to 50K: 3 Mile` <- (census.scores$`Household income 40K to 50K: 3 Mile` - min(census.scores$`Household income 40K to 50K: 3 Mile`))/(max(census.scores$`Household income 40K to 50K: 3 Mile`) - min(census.scores$`Household income 40K to 50K: 3 Mile`))

census.scores$`Household income 50K to 60K: 3 Mile` <- (census.scores$`Household income 50K to 60K: 3 Mile` - min(census.scores$`Household income 50K to 60K: 3 Mile`))/(max(census.scores$`Household income 50K to 60K: 3 Mile`) - min(census.scores$`Household income 50K to 60K: 3 Mile`))

census.scores$`Household income 60K to 75K: 3 Mile` <- (census.scores$`Household income 60K to 75K: 3 Mile` - min(census.scores$`Household income 60K to 75K: 3 Mile`))/(max(census.scores$`Household income 60K to 75K: 3 Mile`) - min(census.scores$`Household income 60K to 75K: 3 Mile`))

census.scores$`Household income 75K to 100K: 3 Mile` <- (census.scores$`Household income 75K to 100K: 3 Mile` - min(census.scores$`Household income 75K to 100K: 3 Mile`))/(max(census.scores$`Household income 75K to 100K: 3 Mile`) - min(census.scores$`Household income 75K to 100K: 3 Mile`))

census.scores$`Household income 100K to 125K: 3 Mile` <- (census.scores$`Household income 100K to 125K: 3 Mile` - min(census.scores$`Household income 100K to 125K: 3 Mile`))/(max(census.scores$`Household income 100K to 125K: 3 Mile`) - min(census.scores$`Household income 100K to 125K: 3 Mile`))

census.scores$`Household income 125K to 150K: 3 Mile` <- (census.scores$`Household income 125K to 150K: 3 Mile` - min(census.scores$`Household income 125K to 150K: 3 Mile`))/(max(census.scores$`Household income 125K to 150K: 3 Mile`) - min(census.scores$`Household income 125K to 150K: 3 Mile`))

census.scores$`Household income 200K+: 3 Mile` <- (census.scores$`Household income 200K+: 3 Mile` - min(census.scores$`Household income 200K+: 3 Mile`))/(max(census.scores$`Household income 200K+: 3 Mile`) - min(census.scores$`Household income 200K+: 3 Mile`))

census.scores$`Average Block Group Score` <- (census.scores$`Average Block Group Score` - min(census.scores$`Average Block Group Score`))/(max(census.scores$`Average Block Group Score`) - min(census.scores$`Average Block Group Score`))

census.scores
```


```{r}
#Now we make training and validation sets for census

#DISCLAIMER. Like the previous ones, I found ways to calculate the error because once you add more and more values, the plot sooner or later excludes it from the visualization. So I am limiting my inputs to show me the error for now. 

cens.training = sort(sample(nrow(census.scores), nrow(census.scores)*0.6))
cens.train <- census.scores[cens.training, ]
cens.valid <- census.scores[-cens.training, ]
cens.train
cens.valid
```


```{r}
#applying the model to the training set
set.seed(5)
neun.cens.train1 <- neuralnet(`Average Block Group Score` ~ Population + `Population: 3 Mile` + `Households: 3 Mile` + `Percent Kids under 5` + `Percent Kids under 5: 3 Mile` + `Percent Kids 5 to 9` + `Average Age` + `Household income 40K to 50K: 3 Mile` + `Household income 50K to 60K: 3 Mile` + `Household income 60K to 75K: 3 Mile`, data = cens.train, linear.output = T, hidden = c(4,1), rep = 10)


# display train weights
neun.cens.train1$weights

# display train predictions
predict.cens.train1 <- prediction(neun.cens.train1)
predict.cens.train1

#extracting data from array
predict.score.cens.train1 <- predict.cens.train1$rep1[,11]
predict.score.cens.train1

#to add it to the dataset
cens_train$Score.Predict <- predict.score.cens.train1

cens.train

# plot network
plot(neun.cens.train1, rep="best")
```


```{r}
#denormalize data 



library(Metrics)
neunet_cens_train1.RMSE <- rmse(cens_train$Area.Score, predict_score_cens_train1)
neunet_cens_train1.RMSE
```


```{r}
#lets apply this to a validation set
set.seed(5)
neun_cens_valid1 <- neuralnet(Area.Score ~ Population + Population..3.Mile + Households..3.Mile + Kids.under.5 + Kids.under.5..3.Mile + Kids.5.to.9 + Average.Age + Household.income.40K.to.50K..3.Mile + Household.income.50K.to.60K..3.Mile + Household.income.60K.to.75K..3.Mile, data = cens_valid, linear.output = T, hidden = c(4,1), rep = 10)


# display train weights
neun_cens_valid1$weights

# display train predictions
predict_cens_valid1 <- prediction(neun_cens_valid1)
predict_cens_valid1

#extracting data from array
predict_score_cens_valid1 <- predict_cens_valid1$rep1[,11]
predict_score_cens_valid1

#to add it to the dataset
cens_valid$Score.Predict <- predict_score_cens_valid1

cens_valid

# plot network
plot(neun_cens_valid1, rep="best")
```


```{r}
library(Metrics)
neunet_cens_valid1.RMSE <- rmse(cens_valid$Area.Score, predict_score_cens_valid1)
neunet_cens_valid1.RMSE

#so as we can see, the SSE error is much better, almost cut down by half, most likely due to the smaller amount of variables. Maybe because of this again, like I argued the previous model for building, this may be the reason why the RMSE is bigger. This time however, it is not as big of a difference as the building model.
```


```{r}
#lets try and work the census model with the tanh function.
cens_tanh <- census
cens_tanh

#set scale from 0 to 1 to -1 to 1
cens_tanh$Population <- (2 * ((cens_tanh$Population - min(cens_tanh$Population))/max(cens_tanh$Population) - min(cens_tanh$Population))) - 1
cens_tanh$Population..3.Mile <- (2 * ((cens_tanh$Population..3.Mile - min(cens_tanh$Population..3.Mile))/max(cens_tanh$Population..3.Mile) - min(cens_tanh$Population..3.Mile))) - 1
cens_tanh$Households..3.Mile <- (2 * ((cens_tanh$Households..3.Mile - min(cens_tanh$Households..3.Mile))/max(cens_tanh$Households..3.Mile) - min(cens_tanh$Households..3.Mile))) - 1
cens_tanh$Kids.under.5 <- (2 * ((cens_tanh$Kids.under.5 - min(cens_tanh$Kids.under.5))/max(cens_tanh$Kids.under.5) - min(cens_tanh$Kids.under.5))) - 1
cens_tanh$Kids.under.5..3.Mile <- (2 * ((cens_tanh$Kids.under.5..3.Mile - min(cens_tanh$Kids.under.5..3.Mile))/max(cens_tanh$Kids.under.5..3.Mile) - min(cens_tanh$Kids.under.5..3.Mile))) - 1
cens_tanh$Kids.5.to.9 <- (2 * ((cens_tanh$Kids.5.to.9 - min(cens_tanh$Kids.5.to.9))/max(cens_tanh$Kids.5.to.9) - min(cens_tanh$Kids.5.to.9))) - 1
cens_tanh$Kids.5.to.9..3.Mile <- (2 * ((cens_tanh$Kids.5.to.9..3.Mile - min(cens_tanh$Kids.5.to.9..3.Mile))/max(cens_tanh$Kids.5.to.9..3.Mile) - min(cens_tanh$Kids.5.to.9..3.Mile))) - 1
cens_tanh$Average.Age <- (2 * ((cens_tanh$Average.Age - min(cens_tanh$Average.Age))/max(cens_tanh$Average.Age) - min(cens_tanh$Average.Age))) - 1
cens_tanh$Household.income.under.40K..3.Mile <- (2 * ((cens_tanh$Household.income.under.40K..3.Mile - min(cens_tanh$Household.income.under.40K..3.Mile))/max(cens_tanh$Household.income.under.40K..3.Mile) - min(cens_tanh$Household.income.under.40K..3.Mile))) - 1
cens_tanh$Household.income.40K.to.50K..3.Mile <- (2 * ((cens_tanh$Household.income.40K.to.50K..3.Mile - min(cens_tanh$Household.income.40K.to.50K..3.Mile))/max(cens_tanh$Household.income.40K.to.50K..3.Mile) - min(cens_tanh$Household.income.40K.to.50K..3.Mile))) - 1
cens_tanh$Household.income.50K.to.60K..3.Mile <- (2 * ((cens_tanh$Household.income.50K.to.60K..3.Mile - min(cens_tanh$Household.income.50K.to.60K..3.Mile))/max(cens_tanh$Household.income.50K.to.60K..3.Mile) - min(cens_tanh$Household.income.50K.to.60K..3.Mile))) - 1
cens_tanh$Household.income.60K.to.75K..3.Mile <- (2 * ((cens_tanh$Household.income.60K.to.75K..3.Mile - min(cens_tanh$Household.income.60K.to.75K..3.Mile))/max(cens_tanh$Household.income.60K.to.75K..3.Mile) - min(cens_tanh$Household.income.60K.to.75K..3.Mile))) - 1
cens_tanh$Household.income.75K.to.100K..3.Mile <- (2 * ((cens_tanh$Household.income.75K.to.100K..3.Mile - min(cens_tanh$Household.income.75K.to.100K..3.Mile))/max(cens_tanh$Household.income.75K.to.100K..3.Mile) - min(cens_tanh$Household.income.75K.to.100K..3.Mile))) - 1
cens_tanh$Household.income.100K.to.125K..3.Mile <- (2 * ((cens_tanh$Household.income.100K.to.125K..3.Mile - min(cens_tanh$Household.income.100K.to.125K..3.Mile))/max(cens_tanh$Household.income.100K.to.125K..3.Mile) - min(cens_tanh$Household.income.100K.to.125K..3.Mile))) - 1
cens_tanh$Household.income.125K.to.150K..3.Mile <- (2 * ((cens_tanh$Household.income.125K.to.150K..3.Mile - min(cens_tanh$Household.income.125K.to.150K..3.Mile))/max(cens_tanh$Household.income.125K.to.150K..3.Mile) - min(cens_tanh$Household.income.125K.to.150K..3.Mile))) - 1
cens_tanh$Household.income.150K.to.200K..3.Mile <- (2 * ((cens_tanh$Household.income.150K.to.200K..3.Mile - min(cens_tanh$Household.income.150K.to.200K..3.Mile))/max(cens_tanh$Household.income.150K.to.200K..3.Mile) - min(cens_tanh$Household.income.125K.to.150K..3.Mile))) - 1
cens_tanh$Household.income.200K...3.Mile <- (2 * ((cens_tanh$Household.income.200K...3.Mile - min(cens_tanh$Household.income.200K...3.Mile))/max(cens_tanh$Household.income.200K...3.Mile) - min(cens_tanh$Household.income.200K...3.Mile))) - 1

cens_tanh
```


```{r}
#make training and validation sets
cens_training_tanh = sort(sample(nrow(cens_tanh), nrow(cens_tanh)*0.6))
cens_train_tanh <- cens_tanh[cens_training_tanh, ]
cens_valid_tanh <- cens_tanh[-cens_training_tanh, ]
cens_train_tanh
```


```{r}
#lets make a model with tanh!
set.seed(6)
neun_cens_train2 <- neuralnet(Area.Score ~ Population + Population..3.Mile + Households..3.Mile + Kids.under.5 + Kids.under.5..3.Mile + Kids.5.to.9 + Average.Age + Household.income.40K.to.50K..3.Mile + Household.income.50K.to.60K..3.Mile + Household.income.60K.to.75K..3.Mile, data = cens_train_tanh, linear.output = T, hidden = c(4,1), act.fct = "tanh", rep = 10)


#display weights
neun_cens_train2$weights

# display train predictions
predict_cens_train2 <- prediction(neun_cens_train2)
predict_cens_train2

#extracting data from array
predict_score_cens_train2 <- predict_cens_train2$rep1[,11]
predict_score_cens_train2

#to add it to the dataset
cens_train_tanh$Score.Predict <- predict_score_cens_train2

cens_train_tanh

# plot network
plot(neun_cens_train2, rep = "best")
```


```{r}
#denormalize data
predict_score_cens_train2 <- (predict_score_cens_train2 * 2) + 3
predict_score_cens_train2

cens_train_tanh$Area.Score <- (cens_train_tanh$Area.Score * 2) + 3
cens_train_tanh$Area.Score

library(Metrics)
neunet_cens_train2.RMSE <- rmse(cens_train_tanh$Area.Score, predict_score_cens_train2)
neunet_cens_train2.RMSE


    
confusionMatrix(as.factor(round(predict_score_cens_train2,0)),as.factor(cens_train_tanh$Area.Score))




class(cens_train_tanh$Area.Score)
class(predict_score_cens_train2)
```


```{r}
#To be honest, its 3AM and i just copied the model from the earlier one, and this things is pretty good. SSE of .08 i think and RMSE of .2 something. Im moving on to the validation set. 

set.seed(6)
neun_cens_valid2 <- neuralnet(Area.Score ~ Population + Population..3.Mile + Households..3.Mile + Kids.under.5 + Kids.under.5..3.Mile + Kids.5.to.9 + Average.Age + Household.income.40K.to.50K..3.Mile + Household.income.50K.to.60K..3.Mile + Household.income.60K.to.75K..3.Mile, data = cens_valid_tanh, linear.output = T, hidden = c(4,1), act.fct = "tanh", rep = 10)


#display weights
neun_cens_valid2$weights

# display train predictions
predict_cens_valid2 <- prediction(neun_cens_valid2)
predict_cens_valid2

#extracting data from array
predict_score_cens_valid2 <- predict_cens_valid2$rep1[,11]
predict_score_cens_valid2

#to add it to the dataset
cens_valid_tanh$Score.Predict <- predict_score_cens_valid2

cens_valid_tanh

# plot network
plot(neun_cens_valid2, rep = "best")

#denormalized data
predict_score_cens_valid2 <- (predict_score_cens_valid2 * 2) + 3
predict_score_cens_valid2

cens_valid_tanh$Area.Score <- (cens_valid_tanh$Area.Score * 2) + 3
cens_valid_tanh$Area.Score


library(Metrics)
neunet_cens_valid2.RMSE <- rmse(cens_valid_tanh$Area.Score, predict_score_cens_valid2)
neunet_cens_valid2.RMSE



confusionMatrix(as.factor(round(predict_score_cens_valid2,0)),as.factor(cens_valid_tanh$Area.Score))
```

```{r}
#make neuralnet

cens_tanh
set.seed(6)
census_neun <- neuralnet(Area.Score ~ Population + Population..3.Mile + Households..3.Mile + Kids.under.5 + Kids.under.5..3.Mile + Kids.5.to.9 + Average.Age + Household.income.40K.to.50K..3.Mile + Household.income.50K.to.60K..3.Mile + Household.income.60K.to.75K..3.Mile, data = cens_tanh, linear.output = T, hidden = c(4,1), act.fct = "tanh", rep = 10)


save(file="censusneunetmodel",census_neun)

# display weights
census_neun$weights

# display predictions
predict_census <- prediction(census_neun)
predict_census

#extracting data from array
predict_score_census <- predict_census$rep1[,11]
predict_score_census

#to add it to the dataset
cens_tanh$Score.Predict <- predict_score_census

cens_tanh

# plot network
plot(census_neun, rep="best")

actual = c(census$Area.Score)

library(Metrics)
census_neun.RMSE <- rmse(census$Area.Score, predict_score_census)
census_neun.RMSE
```


```{r}
#now make the CSV
census_for_csv <- read.csv("NeunetCensusSet.csv")

#check for duplicates
duplicated(census_for_csv$Population)

#remove rows with duplicates
census_for_csv <- census_for_csv[!duplicated(census_for_csv[ , "Population"]),]

census_for_csv

#denormalize predicted score
unnorm_predict_census_score <- predict_score_census * (max(census_for_csv$Area.Score)-min(census_for_csv$Area.Score)) + min(census_for_csv$Area.Score)
unnorm_predict_census_score

census_for_csv

census_for_csv$Score.Predict <- unnorm_predict_census_score
setwd("D:/Templates/UW Stuff/Classes/MSBA/Classes/PM Stuff for me/NeuralNet")
write.csv(census_for_csv, "D:/Templates/UW Stuff/Classes/MSBA/Classes/PM Stuff for me/NeuralNet\\Enrique_Census_Score.csv", row.names = FALSE)
```