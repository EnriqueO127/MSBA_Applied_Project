library(DBI, quietly = TRUE, warn.conflicts = FALSE)
library(sqldf, quietly = TRUE, warn.conflicts = FALSE)
library(dplyr, quietly = TRUE, warn.conflicts = FALSE)
library(neuralnet, quietly = TRUE, warn.conflicts = FALSE)
library(Metrics, quietly = TRUE, warn.conflicts = FALSE)
library(caret, quietly = TRUE, warn.conflicts = FALSE)
library(config, quietly = TRUE, warn.conflicts = FALSE)
library(datapackage.r, quietly = TRUE, warn.conflicts = FALSE)
library(jsonlite, quietly = TRUE, warn.conflicts = FALSE)
library(fastDummies, quietly = TRUE, warn.conflicts = FALSE)
library(datarium, quietly = TRUE, warn.conflicts = FALSE)


get_data <- function(scores_or_new, server, user, password, database, port){
  library(odbc)
  con <- DBI::dbConnect(odbc::odbc(),
                        driver = "PostgreSQL Unicode(x64)",
                        database = as.character(database),
                        UID      = as.character(user),
                        PWD      = as.character(password),
                        server = as.character(server),
                        port = as.character(port))
  sql.lease <- 'SELECT
    bld."CS_ID"
    ,avg(bs."Score") "Average Building Score"
    ,bld."Address_Line"
    ,bld."City"
    ,bld."Postal_Code"
    ,bld."Property_Type"
    ,bld."Year_Built"
    ,bld."Price"
    ,bld."SquareFeet"
    ,round(cast(coalesce(bld."Price" / bld."SquareFeet",NULL) as numeric),0) "$ per sq ft"
     ,bld."Sale_Type"
     ,bld."Currently_listed"
     ,bld."Price_monthly"
     ,bld."Price_yearly"
     ,bld."Space"
     ,bld."Condition"
     ,bld."Available"
     ,bld."Term"
    from "Building" as bld
    left join "Building_Score" as bs on bld."CS_ID" = bs.cs_id
    where bld."Sale_Lease"=\'Lease\'
    group by bld."CS_ID",bld."Address_Line",bld."City",bld."Postal_Code",bld."Property_Type",bld."Price",bld."Year_Built",bld."SquareFeet",bld."Sale_Type"'
  
  lease.import <- dbGetQuery(con,sql.lease)
  if(scores_or_new == 'scores'){
    lease.scores <- sqldf('SELECT "CS_ID", "Average Building Score", "Sale_Lease", "City", "Property_Type", "SquareFeet", "Currently_listed", 
                             "Price_monthly", "Price_yearly", "Space", "Condition", "Available", "Term" FROM "lease.import"
                             WHERE "Average Building Score" IS NOT NULL')
    return(lease.scores)
  }
  if(scores_or_new == 'new'){
    lease.new <- sqldf('SELECT "CS_ID", "Average Building Score", "Sale_Lease", "City", "Property_Type", "SquareFeet", "Currently_listed", 
                          "Price_monthly", "Price_yearly", "Space", "Condition", "Available", "Term" FROM "lease.import" 
                          WHERE "Average Building Score" IS NULL') 
    return(lease.new)
  }
  if(scores_or_new == 'all'){
    lease.all <- sqldf('SELECT "CS_ID", "Average Building Score", "Sale_Lease", "City", "Property_Type", "SquareFeet", "Currently_listed", 
                          "Price_monthly", "Price_yearly", "Space", "Condition", "Available", "Term" FROM "lease.import"') 
    return(lease.all)
  }
  return("")
}

convert_to_01<-function(array,min.amt=NULL,max.amt=NULL){
  if (is.null(min.amt)){
    min.amt <- min(array)
  }
  if (is.null(max.amt)){
    max.amt<-max(array)
  }
  new.range <- (array-min.amt)/(max.amt-min.amt)
  return(new.range)
}

#Convert array of numbers from 0-1 range to full range
convert_from_01<-function(array,min.amt,max.amt){
  full.range=(array*(max.amt-min.amt))+min.amt
  return(full.range)
}

clean_leasing_data <- function(lease.scores){
  #build.scores$Score <- as.factor(build.scores$Score)
  #dummy code catigorical variables
  lease.scores <- fastDummies::dummy_cols(lease.scores, select_columns = "City")
  lease.scores <- fastDummies::dummy_cols(lease.scores, select_columns = "Property_Type")
  lease.scores <- fastDummies::dummy_cols(lease.scores, select_columns = "Condition")
  lease.scores <- fastDummies::dummy_cols(lease.scores, select_columns = "Available")
  
  #rename select columns for modeling purposes
  names(lease.scores)[names(lease.scores) == "Condition_Full Build-Out"] <- "Condition_Full_Build_Out"
  names(lease.scores)[names(lease.scores) == "Condition_Not Listed"] <- "Condition_Not_Listed"
  names(lease.scores)[names(lease.scores) == "Condition_Partial Build-Out"] <- "Condition_Partial_Build_Out"
  names(lease.scores)[names(lease.scores) == "Available_30 Days"] <- "Available_30_Days"
  names(lease.scores)[names(lease.scores) == "Average Building Score"] <- "Average_Building_Score"
  
  #subset the data so it does not contain any useless variables
  leasing.scores <- subset(lease.scores, select = c(CS_ID, Average_Building_Score, Currently_listed, Price_monthly, Price_yearly, SquareFeet, City_Puyallup,                                                           City_Tacoma, Property_Type_Industrial, Property_Type_Office, Condition_Full_Build_Out, Condition_Not_Listed,                                                      Condition_Partial_Build_Out, Available_30_Days, Available_Now))
  
  #get rid of the NA's
  non_na <- complete.cases(leasing.scores[, c("CS_ID", "Currently_listed", "Price_monthly", "Price_yearly", "SquareFeet", "City_Puyallup",                                                                      "City_Tacoma", "Property_Type_Industrial", "Property_Type_Office", "Condition_Full_Build_Out",                                                                    "Condition_Not_Listed", "Condition_Partial_Build_Out", "Available_30_Days", "Available_Now")])
  out <- leasing.scores[non_na, ]
  
  #normalize numeric variables
  out$Average_Building_Score <- convert_to_01(out$Average_Building_Score)
  out$Price_monthly <- convert_to_01(out$Price_monthly)
  out$Price_yearly <- convert_to_01(out$Price_yearly)
  out$SquareFeet <- convert_to_01(out$SquareFeet)
  
  out$Currently_listed <- as.numeric(out$Currently_listed)
  
  #remove duplicates
  # out <- out[!duplicated(out[ , "Price"]),]
  
  return(out)
}

nn.lease.model.score <- function(data){
  #insert model
  set.seed(1)
  train.model <- neuralnet(Average_Building_Score ~ Currently_listed + Price_monthly + Price_yearly + SquareFeet + City_Puyallup + City_Tacoma +                    Property_Type_Industrial + Property_Type_Office + Condition_Full_Build_Out + Condition_Not_Listed + Condition_Partial_Build_Out + Available_30_Days +             Available_Now, data = data, linear.output = F, hidden = c(5,1))
  #extract scores and set scale 1-5
  predicted.scores <- prediction(train.model)
  predict.scores <- predicted.scores$rep1[,14]
  norm.predict.scores <- predict.scores * 5
  Scores.predict <- data.frame(data$CS_ID, norm.predict.scores)
  return(Scores.predict)
}

nn.lease.model.new <- function(server, user, password, database, port, data){
  scores_data <- get_data("scores", server, user, password, database, port)
  cleaned.scores.data <- clean_leasing_data(scores_data)
  #insert model
  set.seed(1)
  train.model.new <- neuralnet(Average_Building_Score ~ Currently_listed + Price_monthly + Price_yearly + SquareFeet + City_Puyallup + City_Tacoma +                    Property_Type_Industrial + Property_Type_Office + Condition_Full_Build_Out + Condition_Not_Listed + Condition_Partial_Build_Out + Available_30_Days +             Available_Now, data = cleaned.scores.data, linear.output = F, hidden = c(5,1))
  return(train.model)
}

new.lease.predictions <- function(model, data2){
  pred.scores <- convert_from_01(predict(model, newdata=data2, all.units = FALSE),1,5)
  return(pred.scores)
}

mainfunction.scores <- function(server, user, password, database, port){
  
  scores_data <- get_data("scores", as.character(server), as.character(user), as.character(password), as.character(database), port)
  
  clean.scores.data <- clean_leasing_data(scores_data)
  
  score.model <- nn.lease.model.score(clean.scores.data)
  
  return(score.model)
}

mainfunction.scores('greentrike.cfvgdrxonjze.us-west-2.rds.amazonaws.com', 'xxx', 'xxx', 'TEST', 5432)

mainfunction.new <- function(server, user, password, database, port){
  #get the data
  
  new_data <- get_data("new", as.character(server), as.character(user), as.character(password), as.character(database), port)
  #score_data <- get_data("scores", as.character(server), as.character(user), as.character(password), as.character(database), port)
  #clean the data
  
  clean.new.data <- clean_leasing_data(new_data)
  #clean.scores.data <- clean_leasing_data(score_data)
  
  #load in the subfunction
  
  model.new <- nn.lease.model.new('greentrike.cfvgdrxonjze.us-west-2.rds.amazonaws.com', 'xxx', 'xxx', 'TEST', 5432)
  
  #then predict with new
  
  Scores.predict.new <- new.lease.predictions(model.new,clean.new.data)
  return(Scores.predict.new)
}

mainfunction.new('greentrike.cfvgdrxonjze.us-west-2.rds.amazonaws.com', 'xxx', 'xxx', 'TEST', 5432)

mainfunction.all <- function(server, user, password, database, port){
  #get the data
  print("Getting Data")
  leasing.data <- get_data("all", server, user, password, database, port)
  #clean the data
  print("cleaning Data")
  clean.leasing.data <- clean_leasing_data(leasing.data)
  #load in the subfunction
  print("Modeling Data")
  model.all <- nn.lease.model.new(server, user, password, database, port)
  #predict with scores first
  print("Predicting data")
  Scores.predict.raw <- new.predictions(model.all, clean.leasing.data)
  Scores.rounded<-round(Scores.predict.raw)
  Scores.rounded[Scores.rounded<1] <- 1
  Scores.rounded[Scores.rounded>5] <- 5
  df<-as.data.frame(clean.leasing.data$CS_ID)
  df$raw_score<-as.numeric(Scores.predict.raw)
  df$score<-as.integer(Scores.rounded)
  colnames(df)<-c("CS_ID","raw_score","score")
  return(df)
}

df<-mainfunction.all('greentrike.cfvgdrxonjze.us-west-2.rds.amazonaws.com', 'xxx', 'xxx', 'TEST', 5432)
print(df)