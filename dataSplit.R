library(readr)
library(dplyr)

# Read in the data
data <- read_csv("df_weather_scaled_encoded.csv")

# Define city of interest
current_city <- "Charlotte"

# Filter data by city of interest
d <- data %>% 
   filter(city == current_city)

# Extract out the years
dates <- format(as.Date(d$datetime, format="%Y-%m-%d"), "%Y")

# Partition the files based on year
valid <- d[dates == "2016",]
test <- d[dates == "2017",]
train <- d[dates != "2016" & dates != "2017", ]

# Check to make sure partitioning was successful
if(nrow(train) + nrow(test) + nrow(valid) != nrow(d)){
  stop("Partitioning failed")
}

# If successful, write the file
write_csv(valid, path = paste("valid", current_city, ".csv", sep = ""))
write_csv(train, path = paste("train", current_city, ".csv", sep = ""))
write_csv(test, path = paste("test", current_city, ".csv", sep = ""))
