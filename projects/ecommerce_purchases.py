#case


import pandas as pd
data=pd.read_csv("D:\Ecommerce Purchases.txt")
print(data)

print(data.head(10))  #prints first 10 rows
print(data.tail(5))  #prints last 5 rows
print(data.iloc[5:11])    #prints from 5th to 10th rows
print(data.loc[5:10])       #inclusive of both ends

#no of ppl with french as their language
print(len(data[data['Language']=='fr']))

#job title containing engineering
print(len(data[data['Job'].str.contains('engineer',case=False)]))  #add case=false to make cas insensitive

#find email of the person with ip address 132.207.160.22
print(data[data['IP Address']=='132.207.160.22']['Email'])

#how many ppl have a MasterCard as their credit card provider and have purchase above 50
print(data[(data['CC Provider']=='Mastercard') & (data['Purchase Price']>50)])

#find email of person with following creditcard number
print(data[data['Credit Card']=='4664825258997302']['Email'])

#how many ppl purchase during AM and how many during PM
print("Morning purchases: ",len(data[data["AM or PM"]=="AM"]))
print("Afternoon purchases: ",len(data[data["AM or PM"]=="PM"]))

#how many ppl have a credit card that expires in 2020
def expiry():
    count=0
    for date in data['CC Exp Date']:
        if date.split('/')[1]=='20':
            count+=1
    return(count)
print(expiry())

#top 5 most popular email providers
list1=[]
for email in data['Email']:
    list1.append(email.split('@')[1])    #appending each email provider to the list

data['temp_email']=list1    #adding a temporary column to data which we can remove later on
print(data['temp_email'].value_counts().head())    #prints top 5

data.drop('temp_email',axis=1,inplace=True)
print(data.columns)

    