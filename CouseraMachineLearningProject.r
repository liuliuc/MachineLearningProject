---
title: "MachineLearningProject"
author: "Li"
date: "3/27/2020"
output: html_document
---
## Get data

url_train="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
wd=getwd()
download.file(url=fileurl,destifile="wd/train.csv", method="curl", mode="wb")

```{r cars}
summary(cars)
```

## Including Plots

You can also embed plots, for example:

```{r pressure, echo=FALSE}
plot(pressure)
```

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.