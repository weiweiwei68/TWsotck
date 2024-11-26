import csv 
import twstock
import os
import time

target = input("Please enter the sotck code : ")
stock = twstock.Stock(target)
file_path = f"stock_data_{target}.csv"
if not os.path.isfile(file_path):
    title = ["DATE", "CAPACITY", "TURNOVER", "OPEN", "HIGH", "LOW", "CLOSE", "CHANGE", "TRANSACTION"]
    data = []

    for i in range(2021, 2099):
        for j in range(1, 13):
            stocklist = stock.fetch(i, j)
            time.sleep(2)
            if not stocklist:
                break

            for info in stocklist:
                strdate = info.date.strftime("%Y-%m-%d")
                li = [strdate, info.capacity, info.turnover, info.open, info.high, info.low, info.close, info.change, info.transaction]
                data.append(li)


    outputfile = open(file_path, "w", newline="", encoding="big5")
    outputwriter = csv.writer(outputfile)

    outputwriter.writerow(title)
    for dataline in (data):
        outputwriter.writerow(dataline)
    outputfile.close()