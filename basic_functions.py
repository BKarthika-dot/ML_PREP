import pandas as pd
#An empty DataFrame in pandas is a table with no data but can have defined column names and indexes.
#It is useful for setting up a structure before adding data dynamically


n=int(input("enter for top n rows"))
m=int(input("enter for bottom m rows"))

dict1={'Name':['Akash','Ramya','Sumita','Rohit','Kalyani','Kishore','Sandhya','Siva'],
'Marks': [98,32,67,89,70,12,34,67],'Age':[14,14,15,16,13,14,15,15],'Gender':['M','F','F','M','F','M','F','F']}

df=pd.DataFrame(dict1)  #creating a dataframe using dictionary

if n>len(dict1):
    print("invalid")
else:
    print(df.head(n))

if m>len(dict1):
    print("invalid")
else:
    print(df.tail(m))

print(df.shape)   #no of rows and columns

print("no of rows:",df.shape[0])
print("no of columns:",df.shape[1])

print(df.info())   #get information about dataframe no of rows, columns, datatypes,memory

print(df.isnull())  #generates boolean values for missing values
print(df.isnull().sum())        #null values count columnwise
print(df.isnull().sum(axis=1))      #null values count rowwise


print(df.describe())    #overall statistics of dataframe----only works for numerical columns
print(df.describe(include='all'))  #overall statistics both for numerical and categorical data


print(df['Gender'].unique())        #unique values in given column
print(df['Gender'].nunique())           #no of unique values in given column
print(df['Gender'].value_counts())  #count for each unique value