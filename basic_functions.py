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


print(df['Marks'])
print(df['Marks']>=90)    #prints boolean values for whether condition is satisfied
print(df[df['Marks']>=90])    #prints exact case where the condititon is satisfied
print(len(df[df['Marks']>=90]))    #prints number of cases where the condition is satisfied

print(df['Marks'].between(30,90))   #prints boolean values for whether condition is satisfied
print(sum(df['Marks'].between(30,90)))  #prints number of cases where the condition is satisfied

print(df['Age'].mean())
print(df['Marks'].min())
print(df['Marks'].max())

#declaring some method
def marks(x):
    return x/2

print(df['Marks'].apply(marks))     #applying a user defined method to the column

df['Half Marks']=df['Marks'].apply(marks)   #adding a new column to the data frame 
print(df)


df['mark weightage out of 15']=df['Marks'].apply(lambda x: x*15/100)   #adding new column by usinglambda function in apply() method
print(df)

print(df['Name'].apply(len))            #finds length of each name

df["Gender value"]=(df['Gender'].map({'F':1,'M':0}))     #maps all F to 1 and all M to 0
print(df)

df.drop('Gender value',axis=1,inplace=True)   #drops a column from the same data frame
print(df)

df.drop(['Half Marks','mark weightage out of 15'],axis=1,inplace=True)  #to drop multiple columns at the same time dclare a list
print(df)

print(df.columns)  #columns is not a function its an inbuilt pandas attribute
print(df.index)     #shows row indices

df.sort_values(by='Marks',inplace=True)   #sorts based on marks in ascending order
print(df)
df.sort_values(by='Marks',inplace=True,ascending=False)   #descending order
print(df)

df.sort_values(by=['Marks','Age'],inplace=True)   #sorts based on 2 values
print(df)


#display name and marks of female students only
print(df[df['Gender']=='F'][['Name','Marks']])

#display name of male students using isin method
print(df[df['Gender'].isin(['M'])][['Name']])
