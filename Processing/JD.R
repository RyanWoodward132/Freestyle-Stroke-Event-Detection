# JD.R
# This file contains the R code used to process the raw accelerometer and event data of JD
 
#+ Load Libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(patchwork)
library(RSEIS)

#+ Import Data
acc = read.csv("Dat/Raw/Accel/JD.csv")
ev = read.csv("Dat/Raw/Events/JD.csv")

#+ Convert acc$Time to elapsed time in seconds from first timestamp
acc$Time = dmy_hms(acc$Time)
acc$Time = round(as.numeric(difftime(acc$Time, acc$Time[1], units = "secs")), 2)

#+ Finding the starting point
## Find approximate area
vert = ggplot(acc, aes(x = Time, y = Vertical)) +
  geom_line(colour = "#2196F3") + 
  labs(x = "Time", y = "Vertical") +
  theme_minimal()
vert

start = acc[acc$Time > 61.1, ]
start = start[start$Time < 61.5,]

# Check area
startg = ggplot(start, aes(x = Time, y = Vertical)) +
  geom_line(colour = "#2196F3") + 
  labs(x = "Time", y = "Vertical") +
  theme_minimal()
startg

# Calculate starting point
smin = start[start$Vertical == min(start$Vertical), 1]
smax = start[start$Vertical == max(start$Vertical), 1]

sp = round((smin + smax)/2, digits = 2)

## Verify Visually
start_sp = ggplot(start, aes(x = Time, y = Vertical)) +
  geom_line(colour = "#2196F3") + 
  labs(x = "Time", y = "Vertical") +
  theme_minimal() + 
  geom_vline(xintercept = sp)
start_sp

#+ Process ev
## Adjust the event timestamps
ev$Synced = ev$Synced + (sp - ev$Synced[ev$Event == "Start"])

## Drop Original Column
ev = select(ev, -Original)

## Export as .csv
write.csv(ev, "Dat/Processed/Events/JD.csv")

#+ Process acc
## Apply 2nd-Order Butterworth low-pass Filter to each accel axis
freq = 100
cutoff = 20

acc$Vertical = butfilt(acc$Vertical, fh = cutoff, deltat = 1/freq, type = "LP", proto = "BU", npoles = 2, zp = TRUE, RM = TRUE)
acc$Lateral = butfilt(acc$Lateral, fh = cutoff, deltat = 1/freq, type = "LP", proto = "BU", npoles = 2, zp = TRUE, RM = TRUE)
acc$Sagittal = butfilt(acc$Sagittal, fh = cutoff, deltat = 1/freq, type = "LP", proto = "BU", npoles = 2, zp = TRUE, RM = TRUE)

## Calculate Jerk and Jerk magnitude
acc$JerkV = c(NA, diff(acc$Vertical) / diff(acc$Time))
acc$JerkL = c(NA, diff(acc$Lateral) / diff(acc$Time))
acc$JerkS = c(NA, diff(acc$Sagittal) / diff(acc$Time))

acc$JerkMag = sqrt(acc$JerkV^2 + acc$JerkL^2 + acc$JerkS^2)

## Remove NA's
acc = na.omit(acc)

## Export as .csv
write.csv(acc, "Dat/Processed/Accel/JD.csv")