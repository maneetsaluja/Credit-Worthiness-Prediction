#!/usr/bin/env python
# coding: utf-8

# # BYOP - CAPSTONE PROJECT

# ### (Group - E)

# ## CREDIT CARD ELIGIBILITY PREDICTION

# In[1]:


# Import the Libraries


# In[2]:


import pandas as pd
pd.options.display.float_format = '{:.2f}'.format

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})

import seaborn as sns

import os

import warnings
warnings.filterwarnings('ignore')


# ### 1. Data combination and de-duplication - Data Pre-processing

# This dataset's objective is predicting whether an applicant's credit card approval will be approved or not. 
# 
# The dataset contains applicants' basic information and applicant's credit history. 
# 
# There are 438557 rows in application.csv. 
# ID is from 5008804 to 6842885. 
# 
# In credit_record.csv, there are 1048575 rows of 45985 ID's credit record. 
# ID is from 5001711 to 5150487.

# #### 1.1 Application record

# Data Dictionary:-

# * ID: Unique Id of the row
# * CODE_GENDER: Gender of the applicant. M is male and F is female.
# * FLAG_OWN_CAR: Is an applicant with a car. Y is Yes and N is NO.
# * FLAG_OWN_REALTY: Is an applicant with realty. Y is Yes and N is No.
# * CNT_CHILDREN: Count of children.
# * AMT_INCOME_TOTAL: the amount of the income.
# * NAME_INCOME_TYPE: The type of income (5 types in total).
# * NAME_EDUCATION_TYPE: The type of education (5 types in total).
# * NAME_FAMILY_STATUS: The type of family status (6 types in total).
# * DAYS_BIRTH: The number of the days from birth (Negative values).
# * DAYS_EMPLOYED: The number of the days from employed (Negative values). This column has error values.
# * FLAG_MOBIL: Is an applicant with a mobile. 1 is True and 0 is False.
# * FLAG_WORK_PHONE: Is an applicant with a work phone. 1 is True and 0 is False.
# * FLAG_PHONE: Is an applicant with a phone. 1 is True and 0 is False.
# * FLAG_EMAIL: Is an applicant with a email. 1 is True and 0 is False.
# * OCCUPATION_TYPE: The type of occupation (19 types in total). This column has missing values.
# * CNT_FAM_MEMBERS: The count of family members.

# In[3]:


# Check the directory

os.getcwd()


# In[4]:


# Change the directory of the project

os.chdir('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model')


# In[5]:


# Re-check the directory and confirm the changes

os.getcwd()


# In[6]:


# Read the dataset application_record

app = pd.read_csv('application_record.csv')
app.head()


# In[7]:


# Check the last few records of the dataset

app.tail()


# In[8]:


# Column names: convert to lower case

app = app.rename(columns = str.lower)
app.head()


# In[9]:


# Check the dimensions of the dataset

app.shape


# In[10]:


# Check the datatypes of the dataset

app.dtypes


# In[11]:


# Print information about the dataset

app.info()


# In[12]:


# Generate statistical summary of the continuous variables of the dataset and transpose it

app.describe().T


# In[13]:


# Generate statistical summary of the continuous and discrete variables of the dataset and transpose it

app.describe(include = 'all').T


# In[14]:


# Check the total count of unique values of all the variables in the dataset

app.nunique()


# In[15]:


# Find the missing values of all the variables in the dataset

app.isnull().sum().sort_values(ascending = False)


# In[16]:


(app.isnull().sum() / len(app) * 100).sort_values(ascending = False)


# Insights:-
# 
# * There are 30.60% of missing values in the occcupation_type variable of the application_record dataset (app).
# * Since this is an important variable, therefore we will keep it and impute it later.

# NOTE:-
# 
# * Any variable which has null values greater than 35% will be dropped (after considering other factors)

# In[17]:


# Analysing variables containing null values
# Threshold: 30%

null_var = app.isnull().sum().sort_values(ascending = False)
null_var


# In[18]:


null_var30 = null_var[null_var.values > (0.30 * len(app))]
null_var30


# Insights:-
# 
# * There is only one variable which has null values more than 30%.

# In[19]:


# Plotting Bar Graph for null values greater than 30%

plt.figure(figsize = (5, 5))
null_var30.plot(kind = 'bar', color = "#4CB391")                           
plt.title('List of Columns & null counts where null values are more than 30%') 

plt.xlabel("Null Columns", fontdict = {"fontsize": 12, "fontweight": 5}) #Setting X-label and Y-label
plt.ylabel("Count of null values", fontdict = {"fontsize": 12, "fontweight": 5})
plt.show()


# In[20]:


# Analysing variables containing null values
# Threshold: 35%

null_var35 = null_var[null_var.values > (0.35 * len(app))]
null_var35


# Insights:-
# 
# * There are no variable which has null values more than 35%.

# In[21]:


# Check the total number of columns having null values greater than 30%


# In[22]:


len(null_var30)


# In[23]:


# List the column name having null values greater than 30%

col_names = list(null_var30.index.values)
col_names

# app.drop(labels = col_names, axis = 1, inplace = True) #Droping those columns


# Insights:-
# 
# * We have decided to not remove this column as of now as it is below our dropping threshold of 35%

# In[24]:


# Check the total number of columns having null values greater than 30%


# In[25]:


len(null_var35)


# Insights:-
# 
# * There are no columns in the dataset that has null values greater than equal to 35%.
# * Had there been any such column/s then we may have dropped them after further analysis.

# In[26]:


app.shape


# In[27]:


# After removing null values, check the percentage of null values for each column again.


# In[28]:


null = (app.isnull().sum() / len(app) * 100).sort_values(ascending = False)
null


# Insights:-
#     
# * Since there were no columns to remove therefor our dataframe structure and dimensiond remains the same as before.

# In[29]:


# Check sample of duplicate records by combining DAYS_EMPLOYED and DAYS_BIRTH

app.loc[app.days_employed == -1194].loc[app.days_birth == -17778]


# ##### Observations

# * There are many duplicate rows in application_record.csv. 
# * They have the same values in rows except ID.  
# * In this approach we will keep these duplicate records.
# * In OCCUPATION_TYPE there is 134203 missing values, which is 30.60%. We will treat it later.

# In[30]:


# dropping duplicate values

# app = app.drop_duplicates(subset = app.columns[1:], keep = 'first', inplace = False)
# app.head()


# In[31]:


# app.shape


# #### 1.2 Credit record

# This is a csv file with credit record for a part of ID in application record. We can treat it a file to generate labels for modeling. For the applicants who have a record more than 59 past due, they should be rejected.

# Data Dictionary:-

# * ID: Unique Id of the row in application record.
# * MONTHS_BALANCE: The number of months from record time.
# * STATUS: Credit status for this month.
#       X: No loan for the month
#       C: paid off that month 
#       0: 1-29 days past due 
#       1: 30-59 days past due 
#       2: 60-89 days overdue
#       3: 90-119 days overdue 
#       4: 120-149 days overdue 
#       5: Overdue or bad debts, write-offs for more than 150 days 

# In[32]:


# Read the dataset credit_record

cred = pd.read_csv('credit_record.csv')
cred.head()


# In[33]:


# View the last few records of the dataset

cred.tail()


# In[34]:


# Column names: convert to lower case

cred = cred.rename(columns = str.lower)
cred.head()


# In[35]:


# Check the dimensions of the dataset

cred.shape


# In[36]:


# Check the datatypes of the dataset

cred.dtypes


# In[37]:


# Print information about the dataset

cred.info()


# In[38]:


# Generate statistical summary of the continuous variables of the dataset and transpose it

cred.describe().T


# In[39]:


# Generate statistical summary of the continuous and discrete variables of the dataset and transpose it

cred.describe(include = 'all').T


# In[40]:


# Find the total count of unique records

cred.nunique()


# In[41]:


# Find the total number of missing values in every variable

cred.isnull().sum()


# ##### Observations

# * The applicant's credit records are from current month to the past 60 months.
# * There are no missing values in the credit_record.

# In[42]:


# Replace X, and C by 0 of the status variable
# We will be considering '0' as Good Customer for our analysis and '1' as Bad Customer (in the status variable)

cred.status.replace('X', 0, inplace = True)
cred.status.replace('C', 0, inplace = True)

cred.head()


# In[43]:


# Change the datatype of status variable to 'int'

cred.status = cred.status.astype('int')


# In[44]:


# Re-check the datatypes of cred dataset

cred.dtypes


# In[45]:


# Verify the records on the basis of the status value

cred.loc[cred.status == 3]


# In[46]:


# Retrieve the frequency of the status variable

cred.status.value_counts()


# In[47]:


# Group the credit_recond dataset according to the 'id' variable to remove the similar 'id' records
# Also we will take the worst credit record of an applicant (i.e) the maximum value of the status variable against the
# applicant's id.
# In this filteration the other values of the staus will be discarded and only the highest values of the status will be
# considered.

cred = cred.groupby('id').status.max()


# In[48]:


# View the first few records

cred.head(10)


# In[49]:


# View the last few records

cred.tail()


# In[50]:


# View the entire records

cred


# In[51]:


# Retrieve the frequency of the status variable after removing the duplicate ids.

cred.value_counts()


# ##### Merge the Datasets

# In[52]:


# Merge the two datasets : app and cred

df = pd.merge(app, cred, how = 'inner', on = ['id'])


# In[53]:


# View the first few records

df.head(10)


# In[54]:


# Retrieve the frequency of the status variable after merging the datasets.

df.status.value_counts()


# In[55]:


# Verify any random id of an applicant to confirm if the highest/worst values of his status is chosen and other duplicate ids
# are removed.

df.loc[df.id == 5137203]


# In[56]:


# Check the dimension of the merged dataset

df.shape


# In[57]:


# Need to remove other values of status variable and required to keep on '0's and '1's.
# Any value in the status variable that is equal to 2 and above will be converted to '1's and below 2 will be converted to '0's.
# '0's means - Good Customers (including customers that are 0-29 days past due date)
# '1's means - Bad Customers

df.status = df.status.apply(lambda x: 1 if x >= 1 else 0)

df.head(10)


# In[58]:


# View last few records

df.tail(10)


# In[59]:


# Check the dimensions

df.shape


# In[60]:


# Verify the same id of an applicant (the one we checked earlier above) to confirm if the status values of 2 and above are
# converted to '1's or not.
# In this example id = 5137203 earlier had the status value of 4.
# But after applying the '1's and '0's functions, we see that it has converted to '1'.

df.loc[df.id == 5137203]


# ##### Status Variable

# In[61]:


# Retrieve the frequency of the status variable after converting to '1's and '0's.
# 1 means Rejected applicants
# 0 means Accepted applicants

df.status.value_counts()


# 4291 applicants are rejected.

# In[62]:


# Sort the dataset according to the 'amt_income_total' variable.

df = df.sort_values('amt_income_total')
df.head()


# In[63]:


# Reset the index

df = df.reset_index(drop = True)
df.head()


# In[64]:


# View the last few records

df.tail()


# In[65]:


# Replace the 'id' variable with the in-built numeric index values.

df.id = df.index
df.head()


# In[66]:


# View the dimensions of the dataset

df.shape


# In[67]:


# Save the current dataset as csv

df.to_csv('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\merged_data.csv', index = False)


# ##### Observations

# In[68]:


# Print the rejection rate

print('There are ' + str(df.status.sum()) + ' rejected applicants.\n', 
      str(round(df.status.sum() / len(df) * 100, 2)) + '% in 36457 applicants.')


# In[69]:


# Separate the Good applicants and the Bad applicants

# Good applicants

status0 = df.loc[df["status"] == 0] 
status0.shape[0]


# In[70]:


# Bad applicants

status1 = df.loc[df["status"] == 1] 
status1.shape[0]


# In[71]:


# Calculate the imbalance ratio

round(len(status0)/len(status1), 2)


# The Imbalance ratio we got  is "7.5"

# In[72]:


# Let’s check the distribution of the target variable (status) visually using a pie chart.

count1 = 0 
count0 = 0

for i in df['status'].values:
    if i == 1:
        count1 += 1
    else:
        count0 += 1


# In[73]:


count1


# In[74]:


count0


# In[75]:


count1_perc = (count1 / len(df['status'])) * 100
count0_perc = (count0 / len(df['status'])) * 100


# In[76]:


count1_perc


# In[77]:


count0_perc


# In[78]:


# Imbalance Ratio

imbalance_ratio = print(str(round(count0_perc / count1_perc, 2)))


# In[79]:


x = ['Bad Applicants (status = 1)', 'Good Applicants (status = 0)']
y = [count1_perc, count0_perc]
explode = (0.15, 0)  # only "explode" the 1st slice

colors = ['#ff9999','#99ff99']

fig1, ax1 = plt.subplots(figsize = (8,8))
ax1.pie(y, explode = explode, labels = x, colors=colors, autopct = '%1.2f%%', 
        shadow = True, startangle = 110, textprops = {'fontsize': 15})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.suptitle('Data Imbalance', fontsize = 28)
#plt.title('(Status variable)')

plt.show()


# Insights:-
# 
# * df dataframe that is application record is quite imbalanced. 
# * Rejected applicants is 11.77% and Accepted applicants is 88.23%.
# * Ratio is 7.50

# #### Correlation

# In[80]:


# Correlation of merged dataset df

plt.figure(figsize = (25, 22), dpi = 80, facecolor = 'white', edgecolor = 'k')

sns.set(font_scale = 2)

hm = sns.heatmap(df.corr(), annot = True, vmin = -1, vmax = 1, cmap = 'coolwarm', fmt = '.2f', 
                 cbar_kws = {"shrink": .82, 'label': 'Correlation %'},
                 annot_kws = {"size": 18}, linewidths = 0.1, linecolor = 'white', square = True)

plt.title('Correlation matrix of Merged Data (df)\n')

hm.set(xlabel = '\nApplicants Details', ylabel = 'Applicants Details\n')

hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 18, rotation = 45)

hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize = 18)

plt.savefig('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Credit Card Approval\\Temp\\ver 5\\Plotting_Correlation_HeatMap1.jpg')

plt.show()


# Insights:-
# 
# * As per the merged datasets of Application records and Credit records, we find:-
#     * Children count and Family count variables are highly correlated at 89%.
#     * Days_birth and Days_employed variabels are moderately inversely correlated at -62%. 
#     * Id variable will be dropped as it has no significance.

# ### 2. Exploratory data analysis - EDA

# #### 2.1 Binary Features

# There are 7 binary features in a dataset 'df':-
#    * code_gender
#    * flag_own_car
#    * flag_own_realty
#    * flag_mobil
#    * flag_work_phone Phone
#    * flag_phone
#    * flag_email

# Note:-
# * Since every applicant has a mobile phone, therefore, we will drop the entire column of 'flag_mobil'.

# In[81]:


binary_df = df.copy()
binary_df.head()


# In[82]:


binary_df.shape


# #### Convert the data types of Binary features to categorical datatypes

# In[83]:


# Convert the datatypes to category data-type

binary_df['code_gender'] = binary_df['code_gender'].astype('category')
binary_df['flag_own_car'] = binary_df['flag_own_car'].astype('category')
binary_df['flag_own_realty'] = binary_df['flag_own_realty'].astype('category')
binary_df['flag_mobil'] = binary_df['flag_mobil'].astype('category')
binary_df['flag_work_phone'] = binary_df['flag_work_phone'].astype('category')
binary_df['flag_phone'] = binary_df['flag_phone'].astype('category')
binary_df['flag_email'] = binary_df['flag_email'].astype('category')


# In[84]:


binary_df.info()


# #### Mobile Analysis

# In[85]:


# Reason for dropping 'flag_mobil' column

pd.crosstab(df['flag_mobil'], df['status'], margins = True)


# Insights:-
#     
#     * There is only one category of '1's, which means that every applicant has a mobile phone.
#     * Therefore, flag_mobil variable will be dropped as it is not significant for the model building.

# In[86]:


# Drop the 'flag_mobil' variable

binary_df = binary_df.drop(['flag_mobil'], axis = 1)
binary_df.head()


# In[87]:


# Drop the Id variable as it is not significant

binary_df = binary_df.drop(['id'], axis = 1)
binary_df.head()


# In[88]:


binary_df.shape


# #### Gender Analysis

# In[89]:


binary_df.code_gender.value_counts()


# In[90]:


# Use crosstabs

pd.crosstab(binary_df['code_gender'], binary_df['status'], margins = True)


# In[91]:


# Genderwise rejection break-up

gender_rej_perc = (binary_df.groupby('code_gender')['status']
           .value_counts(normalize = True)
           .reset_index(name = 'perc'))
gender_rej_perc


# In[92]:


# Gender distribution on the basis of Good applicants only

# Count

#status0_gen_c = status0.code_gender.value_counts()
#status0_gen_c

status0_gen_c = binary_df.loc[binary_df["status"] == 0] #.code_gender.value_counts() 
status0_gen_c.code_gender.value_counts()


# In[93]:


# Total Male and Female gender distribution

plt.subplots(figsize = (8, 8))
sns.countplot(binary_df['code_gender'])


# In[94]:


# Percentage

status0_gen_p = binary_df.loc[binary_df["status"] == 0]
status0_gen_p.code_gender.value_counts(normalize = True)


# In[95]:


# Gender distribution on the basis of Bad applicants only

# Count

status1_gen_c = binary_df.loc[binary_df["status"] == 1] #.code_gender.value_counts() 
status1_gen_c.code_gender.value_counts()


# In[96]:


# Percentage

status1_gen_p = binary_df.loc[binary_df["status"] == 1]
status1_gen_p.code_gender.value_counts(normalize = True)


# In[97]:


plt.figure(figsize = (18, 20))

plt.subplot(221)
sns.countplot(x = 'status', hue = 'code_gender', data = status0, palette = 'Set2')
plt.title("Gender Distribution in Good Applicants\n")

plt.subplot(222)
sns.countplot(x = 'status', hue = 'code_gender', data = status1, palette = 'Set2')
plt.title("Gender Distribution in Bad Applicants\n")

plt.show()


# Insights:-
# 
# * 67.37% Females are Good applicants and 32.63% Males are Good applicants.
# * 64.27% Female are Bad applicants and 35.73% Males are Bad applicants.

# In[98]:


# Check the status count of rejection and acceptance on the basis of gender

gender_df = binary_df.groupby(["code_gender", 'status'])["status"].count()
gender_df


# In[99]:


# Total rejection count of Males

gender_df_m = binary_df.loc[binary_df.status == 1].loc[binary_df.code_gender == 'M']
gender_df_m.shape[0]


# In[100]:


# Total rejection count of Females

gender_df_f = binary_df.loc[binary_df.status == 1].loc[binary_df.code_gender == 'F']
gender_df_f.shape[0]


# In[101]:


# Total rejections

gender_tot = gender_df_f.shape[0] + gender_df_m.shape[0]
gender_tot


# In[102]:


# Total eligibles

gender_df_m_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.code_gender == 'M']
print("Total Eligible Males: " + str(gender_df_m_eleg.shape[0]))

gender_df_f_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.code_gender == 'F']
print("Total Eligible Females: " + str(gender_df_f_eleg.shape[0]))

gender_eleg = gender_df_f_eleg.shape[0] + gender_df_m_eleg.shape[0]
print("Total Eligible applicants : " + str(gender_eleg))


# In[103]:


# Percencatage of rejection of Males out of total rejections

print('There are ' + str(gender_tot) + ' rejected applicants.')
print('Out of this:-')
print('Males are', gender_df_m.shape[0])
print('Females are', gender_df_f.shape[0], '\n')

print('Percentage of rejection of Males out of total rejections is', str(round(gender_df_m.shape[0] / gender_tot * 100, 2)) + '%.')

print('Percentage of rejection of Females out of total rejections is', str(round(gender_df_f.shape[0] / gender_tot * 100, 2)) + '%.', '\n', '\n')


print('There are ' + str(gender_eleg) + ' eligible applicants.')
print('Out of this:-')
print('Males are', gender_df_m_eleg.shape[0])
print('Females are', gender_df_f_eleg.shape[0], '\n')

print('Percentage of eligible of Males out of total eligible applicants is', str(round(gender_df_m_eleg.shape[0] / gender_eleg * 100, 2)) + '%.')

print('Percentage of eligible of Females out of total eligible applicants is', str(round(gender_df_f_eleg.shape[0] / gender_eleg * 100, 2)) + '%.')


# In[104]:


# Total reject percentage out of 36457 records

tot_gen_rejects_perc = binary_df["status"].sum() / round(len(binary_df["status"])) * 100
print(str(round(tot_gen_rejects_perc, 2)) + '%')


# In[105]:


# Total reject percentage of Males out of 36457 records

tot_gen_rej_counts_m = round((gender_df_m.shape[0] / (len(binary_df))) * 100, 2)
print(str(tot_gen_rej_counts_m) + '%')


# In[106]:


# Total reject percentage of Females out of 36457 records

tot_gen_rej_counts_f = round((gender_df_f.shape[0] / (len(binary_df))) * 100, 2)
print(str(tot_gen_rej_counts_f) + '%')


# In[107]:


# Create a new dataframe of just gender and then add status to it
# Also replace 'M's and 'F's in gender with '1's and '0's

gender_tot_df = ['code_gender']
gender_perc = binary_df[gender_tot_df + ['status']] .replace('M', 1).replace('F', 0)


# In[108]:


gender_perc.head()


# In[109]:


gender_perc.value_counts()


# In[110]:


dict_list = []
for code_gender   in gender_tot_df:
    for one_type in [0, 1]:
        dict_list.append({'feature': code_gender,
                          'type': one_type,
                          'reject_rate_percentage': round(len(gender_perc[gender_perc[code_gender] == one_type][gender_perc.status == 1])
                                                       / len(gender_perc[gender_perc[code_gender] == one_type]) * 100, 2),
                          'count': len(gender_perc[gender_perc[code_gender] == one_type]),
                          'reject_count': len(gender_perc[gender_perc[code_gender] == one_type][gender_perc.status == 1])
                         })


# In[111]:


gender_binary = pd.DataFrame.from_dict(dict_list)
gender_binary


# In[112]:


plt.subplots(figsize = (12, 12))
sns.barplot(x = "feature", y = "reject_rate_percentage", hue = "type", data = gender_binary)
plt.show()


# #### Observations

# * Reject rate as per same gender:-
#     * Total Male applicants are 12027 and out if them 1533 applicants are rejected.
#     * It means that Males rejection percentage out of the total Male applicants is 12.75%.
# 
#     * Total Female applicants are 24430 and out if them 2758 applicants are rejected.
#     * It means that Females rejection percentage out of the total Female applicants is 11.29%.
# 
#     * Therefore, Males are more vulnerable than Feales w.r.t rejection.
# 
# * Rejection rate as per rejected applicants:-
#     * Total rejection of 'gender' is of 4291 applicants out of the total records of 36457 applicants.
#     * And percentage-wise it is 11.77%.
#     
#     * Females rejection percentage out of the total rejected applicants is 64.27%.
#     * Whereas Males rejection percentage out of the total rejected applicants is 35.73%.
#     
#     * Here, the Females rejection rate is HIGHER than that of the Males.
#     
# * According to total records of 36457 applicants:-
#     * Females rejection percentage is 7.57%.
#     * Males rejection percentage is 4.2%
#     
#     * Again, we can see that Females rejection rate is higher than that of the Males.
#     
# * We can clearly see that the REJECTION RATE OF FEMALES is HIGHER than the MALES on 2 counts out of the 3.

# In[113]:


binary_df.head()


# In[114]:


# Convert the categories of 'code_gender' variable back from 'M's and 'F's to '1's and '0's
# Where Male = M = 1 and
# Female = F = 0

binary_df['code_gender'] = binary_df['code_gender'].replace('M', 1).replace('F', 0)
binary_df.head()


# #### flag_own_car Analysis

# In[115]:


binary_df.head()


# In[116]:


binary_df.shape


# In[117]:


binary_df.flag_own_car.value_counts()


# In[118]:


# Total Yes and No own_car distribution

plt.subplots(figsize = (8, 8))
sns.countplot(binary_df['flag_own_car'])


# In[119]:


# Use crosstabs

pd.crosstab(binary_df['flag_own_car'], binary_df['status'], margins = True)


# In[120]:


# Car ownership rejection break-up

car_rej_perc = (binary_df.groupby('flag_own_car')['status'].value_counts(normalize = True).reset_index(name = 'perc'))
car_rej_perc


# In[121]:


# Car ownership break-up of total applicants

car_rej_tot_perc = binary_df.flag_own_car.value_counts(normalize = True).reset_index(name = 'perc')
car_rej_tot_perc


# Insights:-
# 
# * 62.03% of the applicants do not have a car
# * 37.97% of the applicants have a car

# In[122]:


# Car ownership distribution on the basis of Good applicants only

# Count

#status0_car_c = status0.code_gender.value_counts()
#status0_car_c

status0_car_c = binary_df.loc[binary_df["status"] == 0] #.code_gender.value_counts() 
status0_car_c.flag_own_car.value_counts()


# In[123]:


# Percentage

status0_car_p = binary_df.loc[binary_df["status"] == 0]
status0_car_p.flag_own_car.value_counts(normalize = True)


# In[124]:


# Car ownership distribution on the basis of Bad applicants only

# Count

status1_car_c = binary_df.loc[binary_df["status"] == 1] #.code_gender.value_counts() 
status1_car_c.flag_own_car.value_counts()


# In[125]:


# Percentage

status1_car_p = binary_df.loc[binary_df["status"] == 1]
status1_car_p.flag_own_car.value_counts(normalize = True)


# In[126]:


plt.figure(figsize = (18, 20))

plt.subplot(221)
sns.countplot(x = 'status', hue = 'flag_own_car', data = status0, palette = 'Set2')
plt.title("Car Ownership in Good Applicants\n")

plt.subplot(222)
sns.countplot(x = 'status', hue = 'flag_own_car', data = status1, palette = 'Set2')
plt.title("Car Ownership in Bad Applicants\n")

plt.show()


# Insights:-
# 
# * 61.84% Without Car are Good applicants and 38.16% With Car are Good applicants.
# * 63.44% Without Car are Bad applicants and 36.56% With Car are Bad applicants.

# In[127]:


# Find the applicants count who don't own a car w.r.t. status

own_car_st_count = binary_df.groupby(["flag_own_car"])["status"].value_counts(normalize = False).reset_index(name = 'count')
own_car_st_count


# In[128]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_own_car", y = "count", hue = "status", data = own_car_st_count)
plt.show()


# In[129]:


# Find the applicants count who don't own a car w.r.t. status

own_car_st_perc = binary_df.groupby(["flag_own_car"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
own_car_st_perc


# In[130]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_own_car", y = "perc", hue = "status", data = own_car_st_perc)
plt.show()


# Analysis:-

#     * 12.03% of the applicants who don't own the car are rejected.
#     * 11.33% of the applicants who own the car are rejected.

# In[131]:


# Find the applicants count who don't own a car w.r.t. gender

own_car_count = binary_df.groupby(["flag_own_car"])["code_gender"].value_counts(normalize = False).reset_index(name = 'count')
own_car_count


# In[132]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_own_car", y = "count", hue = "level_1", data = own_car_count)
plt.show()


# In[133]:


# Find the applicants percentage who don't own a car w.r.t. gender

own_car_perc = binary_df.groupby(["flag_own_car"])["code_gender"].value_counts(normalize = True).reset_index(name = 'perc')
own_car_perc


# In[134]:


plt.subplots(figsize = (8,8))
sns.barplot(x = "flag_own_car", y = "perc", hue = "level_1", data = own_car_perc)
plt.show()


# Analysis:-
#     
#     * Out of 22614 applicants who don't own a car - 80.30% are Females and 19.70% are Males
#     * Similarly, out of 13843 applicants who own a car = 45.29% are Females and 54.70% are Males

# In[135]:


# Find the applicants count who don't own a car w.r.t. status

own_car_gen_count = binary_df.groupby(["flag_own_car", 'code_gender'])["status"].value_counts(normalize = False).reset_index(name = 'count')
own_car_gen_count


# In[136]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_own_car", y = "count", hue = "code_gender", data = own_car_gen_count)
plt.show()


# In[137]:


# Find the applicants percentage who don't own a car w.r.t. gender

own_car_gen_perc = binary_df.groupby(["flag_own_car", 'code_gender'])["status"].value_counts(normalize = True).reset_index(name = 'perc')
own_car_gen_perc


# Analysis:-

#     Applicants who don't own a car:
#     * 11.56% of Females who don't own a car are rejected.
#     * 13.94% of Males who don't own a car are rejected.
#     
#     Applicants who own a car:
#     * 10.47% of Females who own a car are rejected.
#     * 12.04% of applicants who own a car are rejected.

# In[138]:


# Check the status count of rejection and acceptance on the basis of own_car

own_car_df = binary_df.groupby(["flag_own_car", 'status'])["status"].count()
own_car_df


# In[139]:


# Total rejection count of applicants who don't own a car (N)

own_car_df_n = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_own_car == 'N']
own_car_df_n.shape[0]


# In[140]:


# Total rejection count of applicants who own a car (Y)

own_car_df_y = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_own_car == 'Y']
own_car_df_y.shape[0]


# In[141]:


# Total rejections

own_car_tot = own_car_df_n.shape[0] + own_car_df_y.shape[0]
own_car_tot


# In[142]:


# Total eligibles

own_car_df_n_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_own_car == 'N']
print("Total Eligible with No Car: " + str(own_car_df_n_eleg.shape[0]))

own_car_df_y_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_own_car == 'Y']
print("Total Eligible with a Car: " + str(own_car_df_y_eleg.shape[0]))

own_car_eleg = own_car_df_n_eleg.shape[0] + own_car_df_y_eleg.shape[0]
print("Total Eligible applicants : " + str(own_car_eleg))


# In[143]:


# Percencatage of rejection of applicants with or without a car out of total rejections

print('There are ' + str(own_car_tot) + ' rejected applicants.')
print('Out of this:-')
print('Applicants without a car are', own_car_df_n.shape[0])
print('Applicants with a car are', own_car_df_y.shape[0], '\n')

print('Percentage of rejection of applicants without a car out of total rejections is', 
      str(round(own_car_df_n.shape[0] / own_car_tot * 100, 2)) + '%.')

print('Percentage of rejection of applicants with a car out of total rejections is', 
      str(round(own_car_df_y.shape[0] / own_car_tot * 100, 2)) + '%.', '\n', '\n')


print('There are ' + str(own_car_eleg) + ' eligible applicants.')
print('Out of this:-')
print('Applicants without a car are', own_car_df_n_eleg.shape[0])
print('Applicants with a car are', own_car_df_y_eleg.shape[0], '\n')

print('Percentage of applicants without a car out of total eligible applicants is', str(round(own_car_df_n_eleg.shape[0] / own_car_eleg * 100, 2)) + '%.')

print('Percentage of applicants with a car out of total eligible applicants is', str(round(own_car_df_y_eleg.shape[0] / own_car_eleg * 100, 2)) + '%.')


# In[144]:


pd.crosstab(binary_df['flag_own_car'], binary_df['status'], margins = True)


# In[145]:


# Create a new dataframe of just own_car and then add status to it

own_car_tot_df = ['flag_own_car']
own_car_perc = binary_df[own_car_tot_df + ['status']].replace('Y', 1).replace('N', 0)


# In[146]:


own_car_perc.head()


# In[147]:


own_car_perc.value_counts()


# In[148]:


dict_list = []
for flag_own_car in own_car_tot_df:
    for one_type in [0, 1]:
        dict_list.append({'feature': flag_own_car,
                          'type': one_type,
                          'reject_rate_percentage': round(len(own_car_perc[own_car_perc[flag_own_car] == one_type]
                                                        [own_car_perc.status == 1])
                                                       / len(own_car_perc[own_car_perc[flag_own_car] == one_type]) * 100, 2),
                          'count': len(own_car_perc[own_car_perc[flag_own_car] == one_type]),
                          'reject_count': len(own_car_perc[own_car_perc[flag_own_car] == one_type][own_car_perc.status == 1])
                         })


# In[149]:


own_car_binary = pd.DataFrame.from_dict(dict_list)
own_car_binary


# In[150]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "feature", y = "reject_rate_percentage", hue = "type", data = own_car_binary)
plt.show()


# #### Observations:-

# * Percentage as per not owning a car:-
#     * Total applicants are 36457 and out of them 22614 applicants don't own a car.
#     * It means that 62.03% of the applicants don't own a car.
# 
#     * Total applicants are 36457 and out of them 13843 applicants own a car.
#     * It means that 37.97% of the applicants own a car.
#     
#     * Percentage of applicants without a car is HIGHER than those who own a car.
#     
# * Rejection rate as per car status:-   
#     * 22614 applicants who don't own the car - 12.03% of the applicants are rejected.
#     * 13843 applicants who own the car - 11.33% of the applicants are rejected.
#     
#     * Rejection rate of of applicants without a car is slightly HIGHER than those with a car.
#     
# * Percentage of car ownership status as per gender:-
#     * Total applicants who don't own the car, out of it 80.30% are Females.
#     * Total applicants who don't own the car, out of it 19.70% are Males.
#     * Total applicants who own the car, out of it 45.29% are Females.
#     * Total applicants who own the car, out of it 54.71% are the Males.
#     
#     * Males have the highest ownership of cars in comparison to Females.
#     * But with regard to not owning a car there is a huge gap between the Males and Females with females at 80.30%.
#     
# * Rejection rate as per the car status on gender basis:-
#     * Don't own the car:
#         * 11.56% of the Females are rejected who don't own the car.
#         * 13.94% of the Males are rejected who don't own the car.
#     * Own the car:-
#         * 10.47% of the Females are rejected who own the car.
#         * 12.04% of the Males are rejected who own the car.
#         
#     * Here Males have HIGHER rejection rate as compared to Females who don't own the car.
#     * And on owning a car, again Males have the HIGER rejection rate.
#     
# * Rejection rate as per rejected applicants:-
#     * Total rejection of 'own_car' is of 4291 applicants out of the total records of 36457 applicants.
#     * And percentage-wise it is 11.77%.
#     
#     * Applicants rejection percentage out of the total rejected applicants owning no car is 63.44%.
#     * Whereas applicants rejection percentage out of the total rejected applicants owning a car is 36.56%.
#     
#     * Here, the rejection rate of applicants without a car is HIGHER than that of the applicants owning a car.
# 
# * We can clearly see that the REJECTION RATE OF APPLICANTS is HIGHER if they don't own a car and this impact the Males more than   the Females.

# In[151]:


binary_df.head()


# In[152]:


# Convert the categories of 'flag_own_car' variable back from 'Y's and 'N's to '1's and '0's
# Where Y = 1 and
# N = 0

binary_df['flag_own_car'] = binary_df['flag_own_car'].replace('Y', 1).replace('N', 0)
binary_df.head()


# #### flag_own_realty Analysis

# In[153]:


binary_df.head()


# In[154]:


binary_df.flag_own_realty.value_counts()


# Analysis:-
#     
# * Out of the 36457 applicants:-
#     * 24506 applicants own a property
#     * 11951 applicants don't own a property

# In[155]:


# Total Yes and No own_realty distribution

plt.subplots(figsize = (8, 8))
sns.countplot(binary_df['flag_own_realty'])


# In[156]:


# Use crosstabs

pd.crosstab(binary_df['flag_own_realty'], binary_df['status'], margins = True)


# In[157]:


# Find the applicants count who don't own a property w.r.t. status

own_prop_st_count = binary_df.groupby(["flag_own_realty"])["status"].value_counts(normalize = False).reset_index(name = 'count')
own_prop_st_count


# Analysis:-
# 
# * 1561 pplicants without property are rejected.
# * 2730 applicants with property are rejected.

# In[158]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_own_realty", y = "count", hue = "status", data = own_prop_st_count)
plt.show()


# In[159]:


# Find the applicants percentage who don't own a car w.r.t. status

own_prop_st_perc = binary_df.groupby(["flag_own_realty"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
own_prop_st_perc


# Analysis:-
# 
# 
# * 13.06% of the applicants without property are rejected.
# * 11.14% of the applicants with property are rejected.
# 
# * The difference between the 2 is merely 1.92%.

# In[160]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_own_realty", y = "perc", hue = "status", data = own_prop_st_perc)
plt.show()


# In[161]:


# Find the applicants count who don't own a car w.r.t. gender

own_prop_gen_count = binary_df.groupby(["flag_own_realty"])["code_gender"].value_counts(normalize = False).reset_index(name = 'count')
own_prop_gen_count


# In[162]:


# Find the applicants percentage who don't own a car w.r.t. gender

own_prop_gen_perc = binary_df.groupby(["flag_own_realty"])["code_gender"].value_counts(normalize = True).reset_index(name = 'perc')
own_prop_gen_perc


# In[163]:


# Find the applicants count who don't own a property w.r.t. gender and rejected as per the status

own_prop_count = binary_df.groupby(["flag_own_realty", 'code_gender'])["status"].value_counts(normalize = False).reset_index(name = 'count')
own_prop_count


# Analysis:-
# 
# * 950 Female applicants without property are rejected.
# * 611 Male applicants without property are rejected.
# 
# * 1808 Female applicants with property are rejected.
# * 922 Male applicants with property are rejected.

# In[164]:


# Find the applicants percentage who don't own a property w.r.t. gender and rejected as per the status

own_prop_perc = binary_df.groupby(["flag_own_realty", 'code_gender'])["status"].value_counts(normalize = True).reset_index(name = 'perc')
own_prop_perc


# Analysis:-
# 
# * 12.50% of Female applicants without property are rejected.
# * 14.04% of Male applicants without property are rejected.
# 
# * 10.74% of Female applicants with property are rejected.
# * 12.01% of Male applicants with property are rejected.
# 
# * There is a HIGHER rejection rate in case of male applicants as compared to female applicants. But having a property reduces the rejection rate by 2% approx in both the male and female applicants.

# In[165]:


# Check the status count of rejection and acceptance on the basis of own_property

own_prop_df = binary_df.groupby(["flag_own_realty", 'status'])["status"].count()
own_prop_df


# In[166]:


# Total rejection count of applicants who don't own a property (N)

own_prop_df_n = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_own_realty == 'N']
own_prop_df_n.shape[0]


# In[167]:


# Total rejection count of applicants who own a property (Y)

own_prop_df_y = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_own_realty == 'Y']
own_prop_df_y.shape[0]


# In[168]:


# Total rejections

own_prop_tot = own_prop_df_n.shape[0] + own_prop_df_y.shape[0]
own_prop_tot


# In[169]:


# Total eligibles

own_prop_df_n_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_own_realty == 'N']
print("Total Eligible without a property: " + str(own_prop_df_n_eleg.shape[0]))

own_prop_df_y_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_own_realty == 'Y']
print("Total Eligible with a property: " + str(own_prop_df_y_eleg.shape[0]))

own_prop_eleg = own_prop_df_n_eleg.shape[0] + own_prop_df_y_eleg.shape[0]
print("Total Eligible applicants : " + str(own_prop_eleg))


# In[170]:


# Percencatage of rejection of applicants with or without a property out of total rejections

print('There are ' + str(own_prop_tot) + ' rejected applicants.')
print('Out of this:-')
print('Applicants without a property are', own_prop_df_n.shape[0])
print('Applicants with a property are', own_prop_df_y.shape[0], '\n')

print('Percentage of rejection of applicants without a property out of total rejections is', 
      str(round(own_prop_df_n.shape[0]/own_prop_tot * 100, 2)) + '%.')

print('Percentage of rejection of applicants with a property out of total rejections is', 
      str(round(own_prop_df_y.shape[0]/own_prop_tot * 100, 2)) + '%.', '\n', '\n')


print('There are ' + str(own_prop_eleg) + ' eligible applicants.')
print('Out of this:-')
print('Applicants without a property are', own_prop_df_n_eleg.shape[0])
print('Applicants with a property are', own_prop_df_y_eleg.shape[0], '\n')

print('Percentage of eligibility of applicants without a property out of total eligibles is', str(round(own_prop_df_n_eleg.shape[0] / own_prop_eleg * 100, 2)) + '%.')

print('Percentage of eligibility of applicants with a property out of total eligibles is', str(round(own_prop_df_y_eleg.shape[0] / own_prop_eleg * 100, 2)) + '%.')


# Analysis:-
#     
#     * A strange thing to note from the above observation is that applicants who own a property consists of 63.62% of the rejections out of the total rejections count.

# In[171]:


pd.crosstab(binary_df['flag_own_realty'], binary_df['status'], margins = True)


# In[172]:


# Create a new dataframe of just own_property and then add status to it
# Also replace 'Y's and 'N's with '1's and '0's in the own_property column

own_prop_tot_df = ['flag_own_realty']
own_prop_perc = binary_df[own_prop_tot_df + ['status']].replace('Y', 1).replace('N', 0)


# In[173]:


own_prop_perc.head()


# In[174]:


own_prop_perc.value_counts()


# In[175]:


dict_list = []
for flag_own_realty in own_prop_tot_df:
    for one_type in [0, 1]:
        dict_list.append({'feature': flag_own_realty,
                          'type': one_type,
                          'reject_rate_percentage': round(len(own_prop_perc[own_prop_perc[flag_own_realty] == one_type]
                                                        [own_prop_perc.status == 1])
                                                       / len(own_prop_perc[own_prop_perc[flag_own_realty] == one_type]) * 100, 2),
                          'count': len(own_prop_perc[own_prop_perc[flag_own_realty] == one_type]),
                          'reject_count': len(own_prop_perc[own_prop_perc[flag_own_realty] == one_type][own_prop_perc.status == 1])
                         })


# In[176]:


own_prop_binary = pd.DataFrame.from_dict(dict_list)
own_prop_binary


# In[177]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "feature", y = "reject_rate_percentage", hue = "type", data = own_prop_binary)
plt.show()


# #### Observations:-

# * Percentage as per not owning a property:-
#     * Total applicants are 36457 and out of them 11951 applicants don't own a property.
#     * It means that 32.78% of the applicants don't own a property.
# 
#     * Total applicants are 36457 and out of them 24506 applicants own a property.
#     * It means that 67.22% of the applicants own a property.
#     
#     * Percentage of applicants with a property is HIGHER than those who don't own a property.
#     
# * Rejection rate as per property status:-   
#     * 11951 applicants who don't own the property - 13.06% of the applicants are rejected.
#     * 24506 applicants who own the property - 11.14% of the applicants are rejected.
#     
#     * Rejection rate of the applicants without a property is HIGHER than those with a property.
#     
# * Percentage of property ownership status as per gender:-
#     * Total applicants who don't own the property, out of it 63.59% are Females.
#     * Total applicants who don't own the property, out of it 36.41% are Males.
#     * Total applicants who own the property, out of it 68.68% are Females.
#     * Total applicants who own the property, out of it 31.32% are the Males.
#     
#     * Females have the highest ownership of property in comparison to Males.
#     * But with regard to not owning a property Females again has the HIGHEST percentage.
#     
# * Rejection rate as per the property status on gender basis:-
#     * Don't own the property:
#         * 12.50% of the Females are rejected who don't own the property.
#         * 14.04% of the Males are rejected who don't own the property.
#     * Own the property:-
#         * 10.74% of the Females are rejected who own the property.
#         * 12.01% of the Males are rejected who own the property.
#         
#     * Here Males have HIGHER rejection rate as compared to Females who don't own the property.
#     * And on owning a property, again Males have the HIGER rejection rate.
#     
# * Rejection rate as per rejected applicants:-
#     * Total rejection of 'own_property' is of 4291 applicants out of the total records of 36457 applicants.
#     * And percentage-wise it is 11.77%.
#     
#     * Applicants rejection percentage out of the total rejected applicants owning no property is 36.38%.
#     * Whereas applicants rejection percentage out of the total rejected applicants owning a property is 63.62%.
#     
#     * Here, the rejection rate of applicants with a property is HIGHER than that of the applicants not owning a property.
# 
# * We can clearly see that the REJECTION RATE OF APPLICANTS is HIGHER if they own a property and this impact the Males more than the Females.

# In[178]:


binary_df.head()


# In[179]:


# Convert the categories of 'flag_own_realty' variable back from 'Y's and 'N's to '1's and '0's
# Where Y = 1 and
# N = 0

binary_df['flag_own_realty'] = binary_df['flag_own_realty'].replace('Y', 1).replace('N', 0)
binary_df.head()


# #### Work Phone Analysis

# In[180]:


binary_df.head()


# In[181]:


binary_df.flag_work_phone.value_counts()


# In[182]:


binary_df.flag_work_phone.value_counts(normalize = True)


# Analysis:-
#     
# * Out of the 36457 applicants:-
#     * 28235 applicants don't own a work phone which consists of 77.44%
#     * 8222 applicants own a work phone which consists of 22.56%

# In[183]:


# Total Yes and No own_car distribution

plt.subplots(figsize = (8, 8))
sns.countplot(binary_df['flag_work_phone'])


# In[184]:


# Use crosstabs

pd.crosstab(binary_df['flag_work_phone'], binary_df['status'], margins = True)


# In[185]:


# Find the applicants count who don't own a work phone w.r.t. status

wp_st_count = binary_df.groupby(["flag_work_phone"])["status"].value_counts(normalize = False).reset_index(name = 'count')
wp_st_count


# In[186]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_work_phone", y = "count", hue = "status", data = wp_st_count)
plt.show()


# In[187]:


# Find the applicants percentage who don't own a work phone w.r.t. status

wp_st_perc = binary_df.groupby(["flag_work_phone"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
wp_st_perc


# Analysis:-
#     
#     * 11.77% of applicants are rejected for not having a work phone.
#     * 11.76% of applicants are rejected for having a work phone.
#     
# * As per the above observation work phone doesn't seem to have any impact on the rejection of the applicants.

# In[188]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_work_phone", y = "perc", hue = "status", data = wp_st_perc)
plt.show()


# In[189]:


# Find the applicants count who don't own a work phone w.r.t. gender

wp_count = binary_df.groupby(["flag_work_phone"])["code_gender"].value_counts(normalize = False).reset_index(name = 'count')
wp_count


# In[190]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_work_phone", y = "count", hue = "level_1", data = wp_count)
plt.show()


# In[191]:


# Find the applicants percentage who don't own a work phone w.r.t. gender

wp_perc = binary_df.groupby(["flag_work_phone"])["code_gender"].value_counts(normalize = True).reset_index(name = 'perc')
wp_perc


# In[192]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_work_phone", y = "perc", hue = "level_1", data = wp_perc)
plt.show()


# Analysis:-
#     
#     * Out of 28235 applicants who don't own a work phone - 68.66% are Females and 31.34% are Males
#     * Similarly, out of 8222 applicants who own a work phone = 61.35% are Females and 38.65% are Males

# In[193]:


# Find the applicants count who don't own a work phone w.r.t. status and gender

wp_gen_count = binary_df.groupby(["flag_work_phone", 'code_gender'])["status"].value_counts(normalize = False).reset_index(name = 'count')
wp_gen_count


# In[194]:


# Find the applicants percentage who don't own a work phone w.r.t. status and gender

wp_gen_perc = binary_df.groupby(["flag_work_phone", 'code_gender'])["status"].value_counts(normalize = True).reset_index(name = 'perc')
wp_gen_perc


# Analysis:-
# 
#     Applicants who don't own a work phone:
#         * 11.27% of Females who don't own a work phone are rejected.
#         * 12.87% of Males who don't own a work phone are rejected.
# 
#     Applicants who own a work phone:
#         * 11.36% of Females who own a work phone are rejected.
#         * 12.40% of applicants who own a work phone are rejected.

# In[195]:


# Check the status count of rejection and acceptance on the basis of work_phone

wp_df = binary_df.groupby(["flag_work_phone", 'status'])["status"].count()
wp_df


# In[196]:


# Total rejection count of applicants who don't own a work_phone (N)

wp_df_n = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_work_phone == 0]
wp_df_n.shape[0]


# In[197]:


# Total rejection count of applicants who own a work_phone (Y)

wp_df_y = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_work_phone == 1]
wp_df_y.shape[0]


# In[198]:


# Total rejections

wp_tot = wp_df_n.shape[0] + wp_df_y.shape[0]
wp_tot


# In[199]:


# Total eligibles

wp_df_n_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_work_phone == 0]
print("Total Eligible applicants without a work phone: " + str(wp_df_n_eleg.shape[0]))

wp_df_y_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_work_phone == 1]
print("Total Eligible applicants with a work phone: " + str(wp_df_y_eleg.shape[0]))

wp_eleg = wp_df_n_eleg.shape[0] + wp_df_y_eleg.shape[0]
print("Total Eligible applicants : " + str(wp_eleg))


# In[200]:


# Percencatage of rejection of applicants with or without a work phone out of total rejections

print('There are ' + str(wp_tot) + ' rejected applicants.')
print('Out of this:-')
print('Applicants without a work phone are', wp_df_n.shape[0])
print('Applicants with a work phone are', wp_df_y.shape[0], '\n')

print('Percentage of rejection of applicants without a work phone out of total rejections is', 
      str(round(wp_df_n.shape[0]/wp_tot * 100, 2)) + '%.')

print('Percentage of rejection of applicants with a work phone out of total rejections is', 
      str(round(wp_df_y.shape[0]/wp_tot * 100, 2)) + '%.', '\n', '\n')


print('There are ' + str(wp_eleg) + ' eligible applicants.')
print('Out of this:-')
print('Applicants without a work phone are', wp_df_n_eleg.shape[0])
print('Applicants with a work phone are', wp_df_y_eleg.shape[0], '\n')

print('Percentage of eligibility of applicants without a work phone out of total eligibles is', str(round(wp_df_n_eleg.shape[0] / wp_eleg * 100, 2)) + '%.')

print('Percentage of eligibility of applicants with a work phone out of total eligibles is', str(round(wp_df_y_eleg.shape[0] / wp_eleg * 100, 2)) + '%.')


# In[201]:


pd.crosstab(binary_df['flag_work_phone'], binary_df['status'], margins = True)


# In[202]:


# Create a new dataframe of just work_phone and then add status to it

wp_tot_df = ['flag_work_phone']
wp_df_perc = binary_df[wp_tot_df + ['status']]


# In[203]:


wp_df_perc.head()


# In[204]:


wp_df_perc.value_counts()


# In[205]:


dict_list = []
for flag_work_phone in wp_tot_df:
    for one_type in [0, 1]:
        dict_list.append({'feature': flag_work_phone,
                          'type': one_type,
                          'reject_rate_percentage': round(len(wp_df_perc[wp_df_perc[flag_work_phone] == one_type]
                                                        [wp_df_perc.status == 1])
                                                       / len(wp_df_perc[wp_df_perc[flag_work_phone] == one_type]) * 100, 2),
                          'count': len(wp_df_perc[wp_df_perc[flag_work_phone] == one_type]),
                          'reject_count': len(wp_df_perc[wp_df_perc[flag_work_phone] == one_type][wp_df_perc.status == 1])
                         })


# In[206]:


wp_binary = pd.DataFrame.from_dict(dict_list)
wp_binary


# In[207]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "feature", y = "reject_rate_percentage", hue = "type", data = wp_binary)
plt.show()


# #### Observations:-

# * Percentage as per not owning a work phone:-
#     * Total applicants are 36457 and out of them 28235 applicants don't own a work phone.
#     * It means that 77.44% of the applicants don't own a work phone.
# 
#     * Total applicants are 36457 and out of them 8222 applicants own a work phone.
#     * It means that 22.56% of the applicants own a work phone.
#     
#     * Percentage of applicants without a work phone is HIGHER than those who own a work phone.
#     
# * Rejection rate as per work phone status:-   
#     * 28235 applicants who don't own the work phone - 11.77% of the applicants are rejected.
#     * 8222 applicants who own the work phone - 11.76% of the applicants are rejected.
#     
#     * Rejection rate of the applicants with or without a work phone is exactly the same.
#     
# * Percentage of work phone ownership status as per gender:-
#     * Total applicants who don't own the work phone, out of it 68.66% are Females.
#     * Total applicants who don't own the work phone, out of it 31.34% are Males.
#     * Total applicants who own the work phone, out of it 61.35% are Females.
#     * Total applicants who own the work phone, out of it 38.65% are the Males.
#     
#     * Females have the highest ownership of work phone in comparison to Males.
#     * But with regard to not owning a work phone Females again has the HIGHEST percentage.
#     
# * Rejection rate as per the work phone status on gender basis:-
#     * Don't own the work phone:
#         * 11.27% of the Females are rejected who don't own the work phone.
#         * 12.87% of the Males are rejected who don't own the work phone.
#     * Own the work phone:-
#         * 11.36% of the Females are rejected who own the work phone.
#         * 12.40% of the Males are rejected who own the work phone.
#         
#     * Here Males have HIGHER rejection rate as compared to Females who don't own the work phone.
#     * And on owning a work phone, again Males have the HIGER rejection rate.
#     * But there is hardly any variation in the rejection rates of both males and females on whether they own the work phone or don't own the work phone.
#     
# * Rejection rate as per rejected applicants:-
#     * Total rejection of 'work_phone' is of 4291 applicants out of the total records of 36457 applicants.
#     * And percentage-wise it is 11.77%.
#     
#     * Applicants rejection percentage out of the total rejected applicants owning no work phone is 77.46%.
#     * Whereas applicants rejection percentage out of the total rejected applicants owning a work phone is 22.54%.
#     
#     * Here, the rejection rate of applicants without a work phone is much HIGHER than that of the applicants owning a work phone.
# 
# * Overall, we can clearly see that the REJECTION RATE OF APPLICANTS is not impacted with or without the work phone.

# In[208]:


binary_df.head()


# #### Phone Analysis

# In[209]:


binary_df.head()


# In[210]:


binary_df.flag_phone.value_counts()


# In[211]:


binary_df.flag_phone.value_counts(normalize = True)


# Analysis:-
# 
# * Out of the 36457 applicants:-
# * 25709 applicants don't own a phone which consists of 70.52%
# * 10748 applicants own a phone which consists of 29.48%

# In[212]:


# Use crosstabs

pd.crosstab(binary_df['flag_phone'], binary_df['status'], margins = True)


# In[213]:


# Total Male and Female gender distribution

plt.subplots(figsize = (8, 8))
sns.countplot(binary_df['flag_phone'])


# In[214]:


# Find the applicants count who don't own a phone w.r.t. status

ph_st_count = binary_df.groupby(["flag_phone"])["status"].value_counts(normalize = False).reset_index(name = 'count')
ph_st_count


# In[215]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_phone", y = "count", hue = "status", data = ph_st_count)
plt.show()


# In[216]:


# Find the applicants percentage who don't own a phone w.r.t. status

ph_st_perc = binary_df.groupby(["flag_phone"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
ph_st_perc


# Analysis:-
# 
# * 11.90% of the applicants who don't own the phone are rejected.
# * 11.45% of the applicants who own the phone are rejected.

# In[217]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_phone", y = "perc", hue = "status", data = ph_st_perc)
plt.show()


# In[218]:


# Find the applicants count who don't own a phone w.r.t. gender

ph_count = binary_df.groupby(["flag_phone"])["code_gender"].value_counts(normalize = False).reset_index(name = 'count')
ph_count


# In[219]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_phone", y = "count", hue = "level_1", data = ph_count)
plt.show()


# In[220]:


# Find the applicants percentage who don't own a phone w.r.t. gender

ph_perc = binary_df.groupby(["flag_phone"])["code_gender"].value_counts(normalize = True).reset_index(name = 'perc')
ph_perc


# Analysis:-
# 
# * Out of 22709 applicants who don't own a phone - 66.19% are Females and 33.80% are Males
# * Similarly, out of 10748 applicants who own a phone - 68.96% are Females and 31.03% are Males

# In[221]:


# Find the applicants count who don't own a phone w.r.t. status and gender

ph_gen_count = binary_df.groupby(["flag_phone", 'code_gender'])["status"].value_counts(normalize = False).reset_index(name = 'count')
ph_gen_count


# In[222]:


# Find the applicants percentage who don't own a phone w.r.t. status and gender

ph_gen_perc = binary_df.groupby(["flag_phone", 'code_gender'])["status"].value_counts(normalize = True).reset_index(name = 'perc')
ph_gen_perc


# Analysis:-
# 
#     Applicants who don't own a phone:
#         * 11.57% of Females who don't own a phone are rejected.
#         * 12.54% of Males who don't own a phone are rejected.
# 
#     Applicants who own a phone:
#         * 10.63% of Females who own a phone are rejected.
#         * 13.27% of applicants who own a phone are rejected.

# In[223]:


# Check the phone count of rejection and acceptance on the basis of status

ph_df = binary_df.groupby(["flag_phone", 'status'])["status"].count()
ph_df


# In[224]:


# Check the phone percentage of rejection and acceptance on the basis of status

ph_df = binary_df.groupby(["flag_phone"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
ph_df


# In[225]:


# Total rejection count of applicants who don't own a phone (N)

ph_df_n = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_phone == 0]
ph_df_n.shape[0]


# In[226]:


# Total rejection count of applicants who own a phone (Y)

ph_df_y = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_phone == 1]
ph_df_y.shape[0]


# In[227]:


# Total rejections

ph_tot = ph_df_n.shape[0] + ph_df_y.shape[0]
ph_tot


# In[228]:


# Total eligibles

ph_df_n_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_phone == 0]
print("Total Eligible without a phone: " + str(ph_df_n_eleg.shape[0]))

ph_df_y_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_phone == 1]
print("Total Eligible with a phone: " + str(ph_df_y_eleg.shape[0]))

ph_eleg = ph_df_n_eleg.shape[0] + ph_df_y_eleg.shape[0]
print("Total Eligible applicants : " + str(ph_eleg))


# In[229]:


# Percencatage of rejection of applicants with or without a phone out of total rejections

print('There are ' + str(ph_tot) + ' rejected applicants.')
print('Out of this:-')
print('Applicants without a phone are', ph_df_n.shape[0])
print('Applicants with a phone are', ph_df_y.shape[0], '\n')

print('Percentage of rejection of applicants without a phone out of total rejections is', 
      str(round(ph_df_n.shape[0]/ph_tot * 100, 2)) + '%.')

print('Percentage of rejection of applicants with a phone out of total rejections is', 
      str(round(ph_df_y.shape[0]/ph_tot * 100, 2)) + '%.' '\n', '\n')



print('There are ' + str(ph_eleg) + ' eligible applicants.')
print('Out of this:-')
print('Applicants without a phone are', ph_df_n_eleg.shape[0])
print('Applicants with a phone are', ph_df_y_eleg.shape[0], '\n')

print('Percentage of eligibility of applicants without a phone out of total eligibles is', str(round(ph_df_n_eleg.shape[0] / ph_eleg * 100, 2)) + '%.')

print('Percentage of eligibility of applicants with a phone out of total eligibles is', str(round(ph_df_y_eleg.shape[0] / ph_eleg * 100, 2)) + '%.')


# In[230]:


pd.crosstab(binary_df['flag_phone'], binary_df['status'], margins = True)


# In[231]:


# Create a new dataframe of just phone and then add status to it

ph_tot_df = ['flag_phone']
ph_df_perc = binary_df[ph_tot_df + ['status']]


# In[232]:


ph_df_perc.head()


# In[233]:


ph_df_perc.value_counts()


# In[234]:


dict_list = []
for flag_phone in ph_tot_df:
    for one_type in [0, 1]:
        dict_list.append({'feature': flag_phone,
                          'type': one_type,
                          'reject_rate_percentage': round(len(ph_df_perc[ph_df_perc[flag_phone] == one_type]
                                                        [ph_df_perc.status == 1])
                                                       / len(ph_df_perc[ph_df_perc[flag_phone] == one_type]) * 100, 2),
                          'count': len(ph_df_perc[ph_df_perc[flag_phone] == one_type]),
                          'reject_count': len(ph_df_perc[ph_df_perc[flag_phone] == one_type][ph_df_perc.status == 1])
                         })


# In[235]:


ph_binary = pd.DataFrame.from_dict(dict_list)
ph_binary


# In[236]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "feature", y = "reject_rate_percentage", hue = "type", data = ph_binary)
plt.show()


# #### Observations:-

# * Percentage as per not owning a phone:-
#     * Total applicants are 36457 and out of them 25709 applicants don't own a phone.
#     * It means that 70.52% of the applicants don't own a phone.
# 
#     * Total applicants are 36457 and out of them 10748 applicants own a phone.
#     * It means that 29.48% of the applicants own a phone.
#     
#     * Percentage of applicants without a phone is HIGHER than those who own a phone.
#     
# * Rejection rate as per phone status:-   
#     * 25709 applicants who don't own the phone - 11.90% of the applicants are rejected.
#     * 10748 applicants who own the phone - 11.45% of the applicants are rejected.
#     
#     * Rejection rate of the applicants with or without a phone is quite close.
#     
# * Percentage of phone ownership status as per gender:-
#     * Total applicants who don't own the phone, out of it 66.19% are Females.
#     * Total applicants who don't own the phone, out of it 33.81% are Males.
#     * Total applicants who own the phone, out of it 68.96% are Females.
#     * Total applicants who own the phone, out of it 31.04% are the Males.
#     
#     * Females have the highest ownership of phone in comparison to Males.
#     * But with regard to not owning a phone Females again has the HIGHEST percentage.
#     
# * Rejection rate as per the phone status on gender basis:-
#     * Don't own the phone:
#         * 11.57% of the Females are rejected who don't own the phone.
#         * 12.54% of the Males are rejected who don't own the phone.
#     * Own the work phone:-
#         * 10.63% of the Females are rejected who own the phone.
#         * 13.27% of the Males are rejected who own the phone.
#         
#     * Here Males have HIGHER rejection rate as compared to Females who don't own the phone.
#     * And on owning a phone, again Males have the HIGER rejection rate.
#     * But there is hardly any variation in the rejection rates of both males and females on whether they own the phone or don't own the phone.
#     
# * Rejection rate as per rejected applicants:-
#     * Total rejection of 'phone' is of 4291 applicants out of the total records of 36457 applicants.
#     * And percentage-wise it is 11.77%.
#     
#     * Applicants rejection percentage out of the total rejected applicants owning no phone is 71.31%.
#     * Whereas applicants rejection percentage out of the total rejected applicants owning a phone is 28.69%.
#     
#     * Here, the rejection rate of applicants without a phone is much HIGHER than that of the applicants owning a phone.
# 
# * Overall, we can clearly see that the REJECTION RATE OF APPLICANTS is not much impacted with or without the phone.

# In[237]:


binary_df.head()


# #### Email Analysis

# In[238]:


binary_df.head()


# In[239]:


binary_df.flag_email.value_counts()


# In[240]:


binary_df.flag_email.value_counts(normalize = True)


# Analysis:-
# 
# * Out of the 36457 applicants:-
# * 33186 applicants don't have an email which consists of 91.03%
# * 3271 applicants have an email which consists of 8.97%

# In[241]:


# Use crosstabs

pd.crosstab(binary_df['flag_email'], binary_df['status'], margins = True)


# In[242]:


# Total Male and Female gender distribution

plt.subplots(figsize = (8, 8))
sns.countplot(binary_df['flag_email'])


# In[243]:


# Find the applicants count who don't own an email w.r.t. status

e_st_count = binary_df.groupby(["flag_email"])["status"].value_counts(normalize = False).reset_index(name = 'count')
e_st_count


# In[244]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_email", y = "count", hue = "status", data = e_st_count)
plt.show()


# In[245]:


# Find the applicants percentage who don't own an email w.r.t. status

e_st_perc = binary_df.groupby(["flag_email"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
e_st_perc


# In[246]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_email", y = "perc", hue = "status", data = e_st_perc)
plt.show()


# Analysis:-
# 
#     * 11.57% of the applicants who don't own an email are rejected.
#     * 13.78% of the applicants who own an email are rejected.

# In[247]:


# Find the applicants count who don't own an email w.r.t. gender

e_gen_count = binary_df.groupby(["flag_email"])["code_gender"].value_counts(normalize = False).reset_index(name = 'count')
e_gen_count


# In[248]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "flag_email", y = "count", hue = "level_1", data = e_gen_count)
plt.show()


# In[249]:


# Find the applicants percentage who don't own an email w.r.t. gender

e_gen_perc = binary_df.groupby(["flag_email"])["code_gender"].value_counts(normalize = True).reset_index(name = 'perc')
e_gen_perc


# In[250]:


plt.subplots(figsize = (8, 8))


sns.barplot(x = "flag_email", y = "perc", hue = "level_1", data = e_gen_perc)
plt.xlabel("Gender Distribution of Email")
plt.ylabel("Percentage")
plt.grid(False)
plt.legend(title = "Gender", loc = 1)
plt.show()


# Analysis:-
# 
#     * Out of 33186 applicants who don't own an email - 66.96% are Females and 33.04% are Males
#     * Similarly, out of 3271 applicants who own a phone - 67.50% are Females and 32.50% are Males

# In[251]:


# Find the applicants count who don't own an email w.r.t. status and gender

e_gen_st_count = binary_df.groupby(["flag_email", 'code_gender'])["status"].value_counts(normalize = False).reset_index(name = 'count')
e_gen_st_count


# In[252]:


# Find the applicants count who don't own an email w.r.t. status and gender

e_gen_st_perc = binary_df.groupby(["flag_email", 'code_gender'])["status"].value_counts(normalize = True).reset_index(name = 'perc')
e_gen_st_perc


# Analysis:-
# 
#     Applicants who don't own an email:
#          * 11.11% of Females who don't own an email are rejected.
#          * 12.48% of Males who don't own an email are rejected.
# 
#     Applicants who own an email:
#         * 12.99% of Females who own an email are rejected.
#         * 15.42% of applicants who own an email are rejected.

# In[253]:


# Check an email count of rejection and acceptance on the basis of status

e_df = binary_df.groupby(["flag_email", 'status'])["status"].count()
e_df


# In[254]:


# Check an email percentage of rejection and acceptance on the basis of status

e_df = binary_df.groupby(["flag_email"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
e_df


# In[255]:


# Total rejection count of applicants who don't own an email (N)

e_df_n = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_email == 0]
e_df_n.shape[0]


# In[256]:


# Total rejection count of applicants who own an email (Y)

e_df_y = binary_df.loc[binary_df.status == 1].loc[binary_df.flag_email == 1]
e_df_y.shape[0]


# In[257]:


# Total rejections

e_tot = e_df_n.shape[0] + e_df_y.shape[0]
e_tot


# In[258]:


# Total eligibles

e_df_n_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_email == 0]
print("Total Eligible without an email: " + str(e_df_n_eleg.shape[0]))

e_df_y_eleg = binary_df.loc[binary_df.status == 0].loc[binary_df.flag_email == 1]
print("Total Eligible with an email: " + str(e_df_y_eleg.shape[0]))

e_eleg = e_df_n_eleg.shape[0] + e_df_y_eleg.shape[0]
print("Total Eligible applicants : " + str(e_eleg))


# In[259]:


# Percencatage of rejection of applicants with or without an email out of total rejections

print('There are ' + str(e_tot) + ' rejected applicants.')
print('Out of this:-')
print('Applicants without an email are', e_df_n.shape[0])
print('Applicants with an email are', e_df_y.shape[0], '\n')

print('Percentage of rejection of applicants without an email out of total rejections is', 
      str(round(e_df_n.shape[0]/e_tot * 100, 2)) + '%.')

print('Percentage of rejection of applicants with an email out of total rejections is', 
      str(round(e_df_y.shape[0]/e_tot * 100, 2)) + '%.', '\n', '\n')



print('There are ' + str(e_eleg) + ' eligible applicants.')
print('Out of this:-')
print('Applicants without an email are', e_df_n_eleg.shape[0])
print('Applicants with an email are', e_df_y_eleg.shape[0], '\n')

print('Percentage of eligibility of applicants without an email out of total eligible is', str(round(e_df_n_eleg.shape[0] / e_eleg * 100, 2)) + '%.')

print('Percentage of eligibility of applicants with an email out of total eligible is', str(round(e_df_y_eleg.shape[0] / e_eleg * 100, 2)) + '%.')


# In[260]:


pd.crosstab(binary_df['flag_email'], binary_df['status'], margins = True)


# In[261]:


# Create a new dataframe of just an email and then add status to it

e_tot_df = ['flag_email']
e_df_perc = binary_df[e_tot_df + ['status']]


# In[262]:


e_df_perc.head()


# In[263]:


e_df_perc.value_counts()


# In[264]:


dict_list = []
for flag_email in e_tot_df:
    for one_type in [0, 1]:
        dict_list.append({'feature': flag_email,
                          'type': one_type,
                          'reject_rate_percentage': round(len(e_df_perc[e_df_perc[flag_email] == one_type]
                                                        [e_df_perc.status == 1])
                                                       / len(e_df_perc[e_df_perc[flag_email] == one_type]) * 100, 2),
                          'count': len(e_df_perc[e_df_perc[flag_email] == one_type]),
                          'reject_count': len(e_df_perc[e_df_perc[flag_email] == one_type][e_df_perc.status == 1])
                         })


# In[265]:


e_binary = pd.DataFrame.from_dict(dict_list)
e_binary


# In[266]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "feature", y = "reject_rate_percentage", hue = "type", data = e_binary)
plt.grid(False)
plt.legend(title = "Email", loc = 1)
plt.show()


# #### Observations:-

# * Percentage as per not owning an email:-
#     * Total applicants are 36457 and out of them 33186 applicants don't own an email.
#     * It means that 91.03% of the applicants don't own an email.
# 
#     * Total applicants are 36457 and out of them 3271 applicants own an email.
#     * It means that 8.97% of the applicants own an email.
#     
#     * Percentage of applicants without an email is HIGHER than those who own an email.
#     
# * Rejection rate as per email status:-   
#     * 33186 applicants who don't own an email - 11.57% of the applicants are rejected.
#     * 3271 applicants who own an email - 13.78% of the applicants are rejected.
#     
#     * Rejection rate of the applicants with an email is bit HIGHER than the ones without an email.
#     
# * Percentage of email ownership status as per gender:-
#     * Total applicants who don't own an email, out of it 66.96% are Females.
#     * Total applicants who don't own an email, out of it 33.04% are Males.
#     * Total applicants who own an email, out of it 67.50% are Females.
#     * Total applicants who own an email, out of it 32.50% are the Males.
#     
#     * Females have the highest ownership of emails in comparison to Males.
#     * But with regard to not owning an email Females again has the HIGHEST percentage.
#     
# * Rejection rate as per an email status on gender basis:-
#     * Don't own an email:
#         * 11.11% of the Females are rejected who don't own an email.
#         * 12.48% of the Males are rejected who don't own an email.
#     * Own an email:-
#         * 12.99% of the Females are rejected who own an email.
#         * 15.42% of the Males are rejected who own an email.
#         
#     * Here Males have HIGHER rejection rate as compared to Females who don't own an email.
#     * And on owning an email, again Males have the HIGER rejection rate.
#     * But there is very little variation in the rejection rates of both males and females on whether they own an email or don't own an email.
#     
# * Rejection rate as per rejected applicants:-
#     * Total rejection of 'email' is of 4291 applicants out of the total records of 36457 applicants.
#     * And percentage-wise it is 11.77%.
#     
#     * Applicants rejection percentage out of the total rejected applicants owning no email is 89.49%.
#     * Whereas applicants rejection percentage out of the total rejected applicants owning an email is 10.51%.
#     
#     * Here, the rejection rate of applicants without an email is much HIGHER than that of the applicants owning an email.
# 
# * Overall, we can clearly see that the REJECTION RATE OF APPLICANTS is not much impacted with or without an email.

# In[267]:


binary_df.head()


# #### Combined plot and summary of Binary Features

# In[268]:


binary_df.head()


# In[269]:


binary_features = ['code_gender', 'flag_own_car', 'flag_own_realty', 'flag_work_phone', 'flag_phone', 'flag_email']
binary_df_plot = binary_df[binary_features + ['status']]
dict_list = []
for feature in binary_features:
    for one_type in [0, 1]:
        dict_list.append({'feature': feature,
                          'type': one_type,
                          'reject_rate_percentage': round(len(binary_df_plot[binary_df_plot[feature] == one_type][binary_df_plot.status == 1]) / 
                          len(binary_df_plot[binary_df_plot[feature] == one_type]) * 100, 2),
                          'count': len(binary_df_plot[binary_df_plot[feature] == one_type]),
                          'reject_count': len(binary_df_plot[binary_df_plot[feature] == one_type][binary_df_plot.status == 1])
                         })


# In[270]:


binary_df_plot.head()


# In[271]:


group_binary = pd.DataFrame.from_dict(dict_list)
group_binary


# In[272]:


plt.subplots(figsize = (20, 12))
sns.barplot(y = "feature", x = "reject_rate_percentage", hue = "type", data = group_binary, orient = 'h')
plt.grid(False)
plt.legend(title = "Type", loc = 1)
plt.show()


# #### CONVERT THE BINARIES TO Ys AND Ns OR Ms AND Fs

# In[273]:


binary_df.head()


# In[274]:


# Convert the binaries of variables back from '1's and '0's to 'Y's and 'N's or to 'M's and 'F's
# Where Y = 1 or M = 1 and
# N = 0 or F = 1

binary_df['code_gender'] = binary_df['code_gender'].replace(1, 'M').replace(0, 'F')
binary_df['flag_own_car'] = binary_df['flag_own_car'].replace(1, 'Y').replace(0, 'N')
binary_df['flag_own_realty'] = binary_df['flag_own_realty'].replace(1, 'Y').replace(0, 'N')
binary_df['flag_work_phone'] = binary_df['flag_work_phone'].replace(1, 'Y').replace(0, 'N')
binary_df['flag_phone'] = binary_df['flag_phone'].replace(1, 'Y').replace(0, 'N')
binary_df['flag_email'] = binary_df['flag_email'].replace(1, 'Y').replace(0, 'N')
binary_df.head()


# #### 2.2 Continuous Features
# There are 5 binary features in a dataset 'new_df':-

# * cnt_children
# * amt_income_total
# * days_birth
# * days_employed
# * cnt_fam_members

# In[275]:


binary_df.head()


# In[276]:


continuous_df = binary_df.copy()
continuous_df.head()


# In[277]:


continuous_df.shape


# In[278]:


numerical_col = continuous_df.select_dtypes(include='number').columns
len(numerical_col)


# In[279]:


continuous_df.info()


# In[280]:


fig , axes = plt.subplots(nrows = 3, ncols = 2, constrained_layout = True)

fig.subplots_adjust(left = 0, bottom = 0, right = 3, top = 5, wspace = 0.09, hspace = 0.3)


for ax, column in zip(axes.flatten(), numerical_col):
    sns.boxplot(continuous_df[column], ax = ax)
plt.grid(False)
plt.show()


# Insights:-
#     
# * There are outliers to be treated.

# #### cnt_children Analysis

# In[281]:


plt.figure(figsize=(10, 8)) 
sns.countplot(x = "cnt_children", data = continuous_df, palette = "viridis_r")
children_count = continuous_df.cnt_children.value_counts()
children_count
for a, b in zip(range(len(children_count)), children_count):
    plt.text(a, b, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 14)
plt.grid(False)
plt.show()


# Palette color codes:-
# 
# ‘Accent’, ‘Accent_r’, ‘Blues’, ‘Blues_r’, ‘BrBG’, ‘BrBG_r’, ‘BuGn’, ‘BuGn_r’, ‘BuPu’, ‘BuPu_r’, ‘CMRmap’, ‘CMRmap_r’, ‘Dark2’, ‘Dark2_r’, ‘GnBu’, ‘GnBu_r’,  ‘Greens’, ‘Greens_r’, ‘Greys’, ‘Greys_r’, ‘OrRd’, ‘OrRd_r’, ‘Oranges’, ‘Oranges_r’,  ‘PRGn’, ‘PRGn_r’, ‘Paired’, ‘Paired_r’, ‘Pastel1’, ‘Pastel1_r’, ‘Pastel2’,  ‘Pastel2_r’, ‘PiYG’, ‘PiYG_r’, ‘PuBu’, ‘PuBuGn’, ‘PuBuGn_r’, ‘PuBu_r’, ‘PuOr’, ‘PuOr_r’, ‘PuRd’, ‘PuRd_r’, ‘Purples’, ‘Purples_r’, ‘RdBu’, ‘RdBu_r’, ‘RdGy’, ‘RdGy_r’, ‘RdPu’, ‘RdPu_r’, ‘RdYlBu’, ‘RdYlBu_r’, ‘RdYlGn’, ‘RdYlGn_r’, ‘Reds’, ‘Reds_r’, ‘Set1’, ‘Set1_r’, ‘Set2’, ‘Set2_r’, ‘Set3’, ‘Set3_r’, ‘Spectral’, ‘Spectral_r’, ‘Wistia’, ‘Wistia_r’, ‘YlGn’, ‘YlGnBu’, ‘YlGnBu_r’, ‘YlGn_r’, ‘YlOrBr’, ‘YlOrBr_r’, ‘YlOrRd’, ‘YlOrRd_r’, ‘afmhot’, ‘afmhot_r’, ‘autumn’, ‘autumn_r’, ‘binary’, ‘binary_r’,  ‘bone’, ‘bone_r’, ‘brg’, ‘brg_r’, ‘bwr’, ‘bwr_r’, ‘cividis’, ‘cividis_r’, ‘cool’, ‘cool_r’,  ‘coolwarm’, ‘coolwarm_r’, ‘copper’, ‘copper_r’, ‘cubehelix’, ‘cubehelix_r’, ‘flag’, ‘flag_r’,  ‘gist_earth’, ‘gist_earth_r’, ‘gist_gray’, ‘gist_gray_r’, ‘gist_heat’, ‘gist_heat_r’, ‘gist_ncar’,  ‘gist_ncar_r’, ‘gist_rainbow’, ‘gist_rainbow_r’, ‘gist_stern’, ‘gist_stern_r’, ‘gist_yarg’,  ‘gist_yarg_r’, ‘gnuplot’, ‘gnuplot2’, ‘gnuplot2_r’, ‘gnuplot_r’, ‘gray’, ‘gray_r’, ‘hot’, ‘hot_r’,  ‘hsv’, ‘hsv_r’, ‘icefire’, ‘icefire_r’, ‘inferno’, ‘inferno_r’, ‘jet’, ‘jet_r’, ‘magma’, ‘magma_r’,  ‘mako’, ‘mako_r’, ‘nipy_spectral’, ‘nipy_spectral_r’, ‘ocean’, ‘ocean_r’, ‘pink’, ‘pink_r’,  ‘plasma’, ‘plasma_r’, ‘prism’, ‘prism_r’, ‘rainbow’, ‘rainbow_r’, ‘rocket’, ‘rocket_r’,  ‘seismic’, ‘seismic_r’, ‘spring’, ‘spring_r’, ‘summer’, ‘summer_r’, ‘tab10’, ‘tab10_r’,’tab20′, ‘tab20_r’, ‘tab20b’, ‘tab20b_r’, ‘tab20c’, ‘tab20c_r’, ‘terrain’, ‘terrain_r’, ‘turbo’,  ‘turbo_r’, ‘twilight’, ‘twilight_r’, ‘twilight_shifted’, ‘twilight_shifted_r’, ‘viridis’,  ‘viridis_r’, ‘vlag’, ‘vlag_r’, ‘winter’, ‘winter_r’

# In[282]:


continuous_df.cnt_children.value_counts()


# In[283]:


continuous_df.cnt_children.value_counts(normalize = True)


# In[284]:


# Find the applicants children count w.r.t. status

child_st_count = continuous_df.groupby(["cnt_children"])["status"].value_counts(normalize = False).reset_index(name = 'count')
child_st_count


# In[285]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = "cnt_children", y = "count", hue = "status", data = child_st_count)
plt.grid(False)
plt.legend(loc = 1, title = 'Status')
plt.show()


# In[286]:


# Find the applicants children percentage w.r.t. status

child_st_perc = continuous_df.groupby(["cnt_children"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
child_st_perc


# In[287]:


plt.subplots(figsize = (16, 8))
sns.barplot(x = "cnt_children", y = "perc", hue = "status", data = child_st_perc)
plt.grid(False)
plt.legend(loc = 1, title = 'Status')
plt.show()


# In[288]:


# Check the children count

child_df = continuous_df["cnt_children"].value_counts()
child_df


# In[289]:


# Check the children percentage

child_df_perc = continuous_df["cnt_children"].value_counts(normalize = True).reset_index(name = 'perc')
child_df_perc


# Analysis:-
#     
#     * Very few applicants have more than 2 child.

# In[290]:


# Check the children count of rejection and acceptance on the basis of status

child__st_df_count = continuous_df.groupby(["cnt_children"])["status"].value_counts()
child__st_df_count


# In[291]:


# Check the children count of rejection on the basis of status

#child__st_df_count_r = new_df.groupby(["total_children"])["status"]
child__st_df_count_r = continuous_df.loc[continuous_df.status == 1]
child__st_df_count_r = child__st_df_count_r.cnt_children.value_counts(normalize = False).reset_index(name = 'count')
child__st_df_count_r


# In[292]:


# Rename the index to total_children

child__st_df_count_r = child__st_df_count_r.rename(columns = {'index' : 'cnt_children'})
child__st_df_count_r


# In[293]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = 'cnt_children', y = 'count', data = child__st_df_count_r)
plt.grid(False)

plt.show()


# In[294]:


# Check the children percentage of rejection and acceptance on the basis of status

child__st_df_perc = continuous_df.groupby(["cnt_children"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
child__st_df_perc


# In[295]:


# Fetch only the rejected records of children percentage

child__st_df_perc_r = child__st_df_perc.loc[child__st_df_perc.status == 1]
child__st_df_perc_r


# In[296]:


plt.subplots(figsize = (8, 8))
sns.barplot(x = 'cnt_children', y = 'perc', data = child__st_df_perc_r)
plt.grid(False)

plt.show()


# Analysis:-
#     
#     * The rejection rate of applicants with 0, 1, 2 or 3 children are not quite different.

# In[297]:


# Dividing applicants into 5 parts on the basis of rejection count

child_count_5 = [children_count[0], children_count[1], children_count[2], children_count[3], children_count[4:].sum()]


# In[298]:


child_count_5


# In[299]:


child_count_5_r = [len(continuous_df[continuous_df.cnt_children == 0][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_children == 1][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_children == 2][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_children == 3][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_children >= 4][continuous_df.status == 1])]


# In[300]:


child_count_5_r


# In[301]:


child_count_df_5 = pd.DataFrame.from_dict({
    'children_count' : child_count_5,
    'reject_count' : child_count_5_r
})


# In[302]:


child_count_df_5['approved_count'] = child_count_df_5.children_count - child_count_df_5.reject_count


# In[303]:


child_count_df_5['reject_rate'] = child_count_df_5.reject_count / child_count_df_5.children_count


# In[304]:


child_count_df_5


# In[305]:


plt.subplots(figsize = (12, 8))
sns.barplot(x = 'children_count', y = 'reject_rate', data = child_count_df_5)
plt.grid(False)
plt.show()


# In[306]:


# Create new columns in new_df for children count and copy its contents

continuous_df['children_cnt_bucket'] = continuous_df['cnt_children']
continuous_df.head()


# In[307]:


continuous_df.shape


# In[308]:


continuous_df.children_cnt_bucket.value_counts()


# In[309]:


# Create buckets


# In[310]:


continuous_df['children_cnt_bucket'] = continuous_df['cnt_children']
continuous_df['children_cnt_bucket'].value_counts()


# In[311]:


continuous_df.loc[(continuous_df.children_cnt_bucket > 5),  'children_cnt_bucket'] = 'More than Five'
continuous_df['children_cnt_bucket'].value_counts()


# In[312]:


continuous_df['children_cnt_bucket'] = continuous_df['children_cnt_bucket'].replace(0, 'None').replace(
    1, 'One').replace(2, 'Two').replace(3, 'Three').replace(4, 'Four').replace(5, 'Five')

continuous_df['children_cnt_bucket'].value_counts()


# In[313]:


continuous_df.head()


# #### cnt_fam_members Analysis

# In[314]:


continuous_df.head()


# In[315]:


continuous_df.shape


# In[316]:


plt.figure(figsize=(16, 8)) 
sns.countplot(x = "cnt_fam_members", data = continuous_df, palette = "viridis_r")
family_count = continuous_df.cnt_fam_members.value_counts()
family_count
for a, b in zip(range(len(family_count)), family_count):
    plt.text(a, b, '%.0f' % b, ha = 'center', va = 'bottom', fontsize = 14)
plt.grid(False)
plt.show()


# In[317]:


continuous_df.cnt_fam_members.value_counts()


# In[318]:


continuous_df.cnt_fam_members.value_counts(normalize = True)


# In[319]:


# Find the applicants family count w.r.t. status

fam_st_count = continuous_df.groupby(["cnt_fam_members"])["status"].value_counts(normalize = False).reset_index(name = 'count')
fam_st_count


# In[320]:


plt.subplots(figsize = (12, 8))
sns.barplot(x = "cnt_fam_members", y = "count", hue = "status", data = fam_st_count)
plt.grid(False)
plt.legend(loc = 1, title = "Status")
plt.show()


# In[321]:


# Find the applicants family percentage w.r.t. status

fam_st_perc = continuous_df.groupby(["cnt_fam_members"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
fam_st_perc


# In[322]:


plt.subplots(figsize = (16, 8))
sns.barplot(x = "cnt_fam_members", y = "perc", hue = "status", data = fam_st_perc)
plt.grid(False)
plt.show()


# In[323]:


# Check the family count

fam_df = continuous_df["cnt_fam_members"].value_counts()
fam_df


# In[324]:


# Check the family percentage

fam_df_perc = continuous_df["cnt_fam_members"].value_counts(normalize = True).reset_index(name = 'perc')
fam_df_perc


# Analysis:-
#     
#     * Very few applicants have more than 4 family members

# In[325]:


# Check the family count of rejection and acceptance on the basis of status

fam_st_df_count = continuous_df.groupby(["cnt_fam_members"])["status"].value_counts()
fam_st_df_count


# In[326]:


# Check the family count of rejection on the basis of status

#fam_st_df_count_r = new_df.groupby(["total_children"])["status"]
fam_st_df_count_r = continuous_df.loc[continuous_df.status == 1]
fam_st_df_count_r = fam_st_df_count_r.cnt_fam_members.value_counts(normalize = False).reset_index(name = 'count')
fam_st_df_count_r


# In[327]:


# Rename the index to total_family_members

fam_st_df_count_r = fam_st_df_count_r.rename(columns = {'index' : 'cnt_fam_members'})
fam_st_df_count_r


# In[328]:


plt.subplots(figsize = (12, 8))
sns.barplot(x = 'cnt_fam_members', y = 'count', data = fam_st_df_count_r)
plt.grid(False)
plt.show()


# In[329]:


# Check the family percentage of rejection and acceptance on the basis of status

fam_st_df_perc = continuous_df.groupby(["cnt_fam_members"])["status"].value_counts(normalize = True).reset_index(name = 'perc')
fam_st_df_perc


# In[330]:


# Fetch only the rejected records of children percentage

fam_st_df_perc_r = fam_st_df_perc.loc[fam_st_df_perc.status == 1]
fam_st_df_perc_r


# In[331]:


plt.subplots(figsize = (12, 8))
sns.barplot(x = 'cnt_fam_members', y = 'perc', data = fam_st_df_perc_r)
plt.grid(False)
plt.show()


# Analysis:-
# 
#     * The rejection rate of applicants with 1, 2 or 3 family members are not quite different.

# In[332]:


# Dividing applicants into 4 parts on the basis of rejection count

fam_count_5 = [family_count[1], family_count[2], family_count[3], family_count[4], family_count[5:].sum()]


# In[333]:


fam_count_5


# In[334]:


fam_count_5_r = [len(continuous_df[continuous_df.cnt_fam_members == 1.0][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_fam_members == 2.0][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_fam_members == 3.0][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_fam_members == 4.0][continuous_df.status == 1]),
                  len(continuous_df[continuous_df.cnt_fam_members >= 5.0][continuous_df.status == 1])]


# In[335]:


fam_count_5_r


# In[336]:


fam_count_df_5 = pd.DataFrame.from_dict({
    'family_mem_count' : fam_count_5,
    'reject_count' : fam_count_5_r
})


# In[337]:


fam_count_df_5['approved_count'] = fam_count_df_5.family_mem_count - fam_count_df_5.reject_count


# In[338]:


fam_count_df_5['reject_rate'] = fam_count_df_5.reject_count / fam_count_df_5.family_mem_count


# In[339]:


fam_count_df_5


# In[340]:


plt.subplots(figsize = (12, 8))
sns.barplot(x = 'family_mem_count', y = 'reject_rate', data = fam_count_df_5)
plt.grid(False)
plt.show()


# #### Verify the relationship between the Children count and the Family Members count

# In[341]:


fig = plt.figure()
ax = fig.add_subplot(111)
gp = continuous_df.groupby(by = ['cnt_children', 'cnt_fam_members'])
gp_df = gp.size().reset_index(name = 'times').sort_values('times', ascending = False)[:6]
gp_df['times_10'] = gp_df['times'].apply(lambda x: x/10)
ax.scatter(gp_df['cnt_children'], gp_df['cnt_fam_members'], s = gp_df['times_10'])
plt.xticks(range(4))
plt.yticks(range(6))
plt.xlabel('cnt_children')
plt.ylabel('cnt_fam_members')
plt.grid(False)
plt.show()


# In[342]:


gp = continuous_df.groupby(by = ['cnt_children', 'cnt_fam_members'])
gp_df = gp.size().reset_index(name='times').sort_values('times', ascending = False)[:6]

gp_df


# In[343]:


continuous_df.head()


# In[344]:


continuous_df.shape


# In[345]:


# Create buckets


# In[346]:


continuous_df.cnt_fam_members.value_counts()


# In[347]:


continuous_df['cnt_fam_members_bucket'] = continuous_df['cnt_fam_members']
continuous_df['cnt_fam_members_bucket'].value_counts()


# In[348]:


continuous_df.loc[(continuous_df.cnt_fam_members_bucket > 7),  'cnt_fam_members_bucket'] = 'More than Seven'
continuous_df['cnt_fam_members_bucket'].value_counts()


# In[349]:


continuous_df['cnt_fam_members_bucket'] = continuous_df['cnt_fam_members_bucket'].replace(1, 'One').replace(2, 'Two').replace(
    3, 'Three').replace(4, 'Four').replace(5, 'Five').replace(6, 'Six').replace(7, 'Seven')

continuous_df['cnt_fam_members_bucket'].value_counts()


# In[350]:


continuous_df.head()


# In[351]:


continuous_df.shape


# #### Income Amount Analysis

# In[352]:


continuous_df.head()


# In[353]:


# Distribution Plot

plt.subplots(figsize = (8, 8))
income_plot = pd.Series(continuous_df.amt_income_total, name = "income")
plt.ylabel('Reject_rate')
sns.distplot(income_plot)
plt.ticklabel_format(style='plain')
plt.grid(False)
plt.show()


# In[354]:


# Remove the scientific notations

# pd.options.display.float_format = '{:.1f}'.format

# Revert back to scientific notation

# pd.reset_option('display.float_format')


# In[355]:


continuous_df.amt_income_total.describe()


# In[356]:


# Check the quantiles

np.quantile(continuous_df.amt_income_total, 0.99)


# In[357]:


continuous_df.amt_income_total.quantile([.01, .25, .5, .75, 0.99])


# We can notice that most applicants' income is lower than 560250. So we select these applicants to get box plot.

# In[358]:


continuous_df.amt_income_total.value_counts()


# In[359]:


# Box Plot

sns.boxplot(x = "status", y = "amt_income_total", data = continuous_df)
plt.grid(False)
plt.show()


# In[360]:


plt.boxplot(continuous_df['amt_income_total'])
plt.grid(False)


# #### Outlier Detection of Income Column

# In[361]:


# IQR

Q1 = np.percentile(continuous_df.amt_income_total, 25)
Q1


# In[362]:


Q3 = np.percentile(continuous_df.amt_income_total, 75)
Q3


# In[363]:


Q1,Q3 = np.percentile(continuous_df.amt_income_total, [25,75])
Q1,Q3


# In[364]:


IQR = Q3 - Q1
ul = Q3 + 1.5 * IQR
ll = Q1 - 1.5 * IQR
IQR, ul, ll


# In[365]:


outliers = continuous_df.amt_income_total[(continuous_df.amt_income_total > ul) | (continuous_df.amt_income_total < ll)]
print(outliers.head())


# In[366]:


outliers = pd.DataFrame(outliers)


# In[367]:


outliers.head()


# In[368]:


outliers.columns = outliers.columns.str.replace('amt_income_total', 'income_outliers')


# In[369]:


outliers.head()


# In[370]:


outliers.income_outliers.value_counts()


# In[371]:


outliers.income_outliers.shape


# In[372]:


plt.subplots(figsize = (8, 8))
sns.distplot(outliers['income_outliers'])
plt.ticklabel_format(style = 'plain')
plt.grid(False)


# #### Compare the Income column with Outliers to Income column without Outliers

# In[373]:


non_outliers = continuous_df[continuous_df['amt_income_total'] < ul]
non_outliers.shape


# In[374]:


plt.figure(figsize=(20,12))

plt.subplot(2,2,1)
sns.distplot(continuous_df['amt_income_total'])
plt.grid(False)
plt.subplot(2,2,2)
sns.boxplot(continuous_df['amt_income_total'])
plt.grid(False)
plt.subplot(2,2,3)
sns.distplot(non_outliers['amt_income_total'])
plt.grid(False)
plt.subplot(2,2,4)
sns.boxplot(non_outliers['amt_income_total'])
plt.grid(False)
plt.show()


# #### Bucketing the Income Column

# In[375]:


continuous_df['income_bucket'] = pd.qcut(continuous_df.amt_income_total, 
                                         q = [0, 0.2, 0.5, 0.8, 0.95, 1], 
                                         labels = ['Very_low', 'Low', "Medium", 'High', 'Very_high'])


# In[376]:


continuous_df['income_bucket'].head()


# In[377]:


continuous_df['income_bucket'].value_counts()


# In[378]:


continuous_df.head()


# In[379]:


continuous_df.shape


# #### days_birth Analysis

# In[380]:


continuous_df.head()


# In[381]:


continuous_df.info()


# In[382]:


# We firstly transform the days from birth into years, and get the histogram and Box diagram.


# In[383]:


continuous_df['days_birth'] = abs(continuous_df['days_birth'])


# In[384]:


continuous_df['days_birth'].head()


# In[385]:


print(continuous_df['days_birth'].unique())


# In[386]:


print(continuous_df['days_birth'].nunique())


# In[387]:


continuous_df['age'] = (continuous_df['days_birth'] / 365.25).astype(int)


# In[388]:


continuous_df['age'].unique()


# In[389]:


continuous_df.head()


# In[390]:


plt.ylabel('Reject_rate')
age_plot = pd.Series(continuous_df.age, name = "age")
sns.distplot(age_plot)
plt.grid(False)
plt.show()


# In[391]:


sns.boxplot(x = "status", y = "age", data = continuous_df)
plt.grid(False)
plt.show()


# In[392]:


continuous_df.age.value_counts()


# In[393]:


continuous_df.age.describe()


# In[394]:


# Binning / Bucketing

continuous_df['age_bucket'] = pd.cut(continuous_df['age'], 
                                     bins = [18, 25, 35, 60, 100], labels=['Very_Young', 'Young', 'Middle_Age', 'Senior_Citizen'])


# In[395]:


continuous_df[['age','age_bucket']].head()


# In[396]:


continuous_df.head()


# In[397]:


continuous_df.shape


# In[398]:


continuous_df.age_bucket.value_counts()


# In[399]:


continuous_df['age'].plot(kind = 'hist', bins = 20, density = True)
plt.grid(False)


# In[400]:


# Separate the Good applicants and the Bad applicants

# Good applicants

new_status0 = continuous_df.loc[continuous_df["status"] == 0] 
new_status0.shape[0]


# In[401]:


# Bad applicants

new_status1 = continuous_df.loc[continuous_df["status"] == 1] 
new_status1.shape[0]


# In[402]:


plt.figure(figsize = (30, 10)) 

plt.subplot(121)
plt.title("For Eligible Applicants = 0")
sns.countplot(x = 'status', hue = 'age_bucket', data = new_status0, palette = 'Set2')
plt.grid(False)

plt.subplot(122)
plt.title("For Not-Eligible Applicants = 1")
sns.countplot(x = 'status', hue = 'age_bucket', data = new_status1 , palette = 'Set2')
plt.grid(False)
plt.show()


# Insights:-
# 
# * Middle Age(35-60) the group seems to applied higher than any other age group for loans in the case of Defaulters as well as Non-defaulters.
# * Also, Middle Age group facing paying difficulties the most.
# * While Senior Citizens(60-100) and Very young(19-25) age group facing paying difficulties less as compared to other age groups.

# In[403]:


continuous_df.info()


# #### days_employed Analysis

# There are error values in this column. We will drop them first and get the employed year of each applicants

# In[404]:


continuous_df.head()


# In[405]:


continuous_df.days_employed


# In[406]:


# We firstly transform the days from employed into years, and get the histogram and Box diagram.


# In[407]:


print(continuous_df['days_employed'].unique())


# In[408]:


print(continuous_df['days_employed'].nunique())


# In[409]:


continuous_df['employed_years'] = continuous_df[continuous_df.days_employed < 0].days_employed.apply(lambda x: int(-x / 365.25))


# In[410]:


print(continuous_df['employed_years'].unique())


# In[411]:


print(continuous_df['employed_years'].nunique())


# In[412]:


continuous_df['employed_years'].value_counts().head(10)


# In[413]:


continuous_df['employed_years'].isnull().sum()


# In[414]:


(continuous_df.isnull().sum() / len(continuous_df) * 100).sort_values(ascending = False)


# Note:-
#     
# * Since the null values in the employed_years variable is less than 35%, therefore, we will impute it.

# In[415]:


continuous_df.head()


# In[416]:


# Replacing NaN values with Zero (0), as pensioners are retired and they are not employed.

continuous_df['employed_years'] = continuous_df['employed_years'].replace(np.nan, 0)


# In[417]:


continuous_df.head()


# In[418]:


continuous_df.employed_years.value_counts()


# In[419]:


continuous_df.isnull().sum()


# Note:-
#     
# * The null values of employed_years is imputed with Zero (0)

# In[420]:


plt.subplots(figsize = (14, 8))
plt.ylabel('Reject_rate')
employed_year_plot = pd.Series(continuous_df.employed_years, name = "employed_years")
sns.distplot(employed_year_plot)
plt.grid(False)
plt.show()


# In[421]:


sns.boxplot(x = "status", y = "employed_years", data = continuous_df)
plt.grid(False)
plt.show()


# In[422]:


continuous_df.employed_years.describe()


# #### We are going to develop the relation between Status with age and income or employed years and income.

# In[423]:


continuous_df.head()


# In[424]:


continuous_df.shape


# In[425]:


continuous_df.info()


# In[426]:


comparison_df = continuous_df.copy()
comparison_df.head()


# In[427]:


comparison_df['age_5'] = comparison_df.age.apply(lambda x: int(x / 5) * 5)


# In[428]:


comparison_df['age_5'].head(10)


# In[429]:


# comparison_df['employed_year_5'] = comparison_df[comparison_df.work_experience < 0].work_experience.apply(lambda x: int(-x / 365.25 / 5) * 5)

comparison_df['employed_year_5'] = comparison_df.employed_years.apply(lambda x: int(x / 5) * 5)


# In[430]:


comparison_df['employed_year_5'].head(10)


# In[431]:


plot_fig = plt.figure()
plt.subplots(figsize = (16, 10))
aei_plot = sns.boxplot(x = "age_5", y = "amt_income_total", hue = 'status', data = comparison_df[comparison_df.amt_income_total <= 382500])
plt.grid(False)
plt.show()
plt.subplots(figsize = (16, 10))
aei_plot = sns.boxplot(x = "employed_year_5", y = "amt_income_total", hue = 'status', data = comparison_df[comparison_df.amt_income_total <= 382500])
plt.grid(False)
plt.show()


# Analysis:-
# 
#     * As figures above, we can know that in terms of age and income rejected applicants are not quite different from 
#       approved applicants through the combination of five-number summary in boxplot. 
# 
#     * However, in terms of age and employed year, applicants with more than 30 years of service are more likely not to be 
#       rejected.

# #### 2.3 Categorical Features

# There are 5 categorical features in a dataset 'continuous_df':-
# 
#     * name_income_type              
#     * name_education_type                
#     * name_family_status           
#     * name_housing_type               
#     * occupation_type               
# 

# In[432]:


continuous_df.head()


# In[433]:


categorical_df = continuous_df.copy()


# In[434]:


categorical_df.head()


# In[435]:


categorical_df.shape


# #### name_income_type Analysis

# In[436]:


categorical_df.isnull().sum()


# In[437]:


categorical_df.name_income_type.nunique()


# In[438]:


categorical_df.name_income_type.unique()


# In[439]:


categorical_df.name_income_type.value_counts()


# In[440]:


categorical_df.name_income_type.value_counts(normalize = True)


# In[441]:


pd.crosstab(categorical_df['name_income_type'], categorical_df['status'], margins = True)


# In[442]:


inctyp_total = categorical_df.groupby(by = ['name_income_type']).size().reset_index(name = 'times')
inctyp_total


# In[443]:


inctyp_reject = categorical_df[categorical_df.status == 1].groupby(by = ['name_income_type']).size().reset_index(name = 'reject_times')
inctyp_reject


# In[444]:


inctyp_reject_rate = pd.merge(inctyp_total, inctyp_reject, how = 'outer', on = ['name_income_type']).fillna(0)
inctyp_reject_rate


# In[445]:


inctyp_reject_rate['reject_rate'] = inctyp_reject_rate.reject_times / inctyp_reject_rate.times
inctyp_reject_rate


# In[446]:


plt.subplots(figsize = (12, 8))
sns.barplot(y = "name_income_type", x = "reject_rate", data = inctyp_reject_rate, orient = 'h')
plt.grid(False)
plt.show()


# #### Education Analysis

# In[447]:


categorical_df.head()


# In[448]:


categorical_df.info()


# In[449]:


categorical_df.isnull().sum()


# In[450]:


categorical_df.name_education_type.nunique()


# In[451]:


categorical_df.name_education_type.unique()


# In[452]:


categorical_df.name_education_type.value_counts()


# In[453]:


categorical_df.name_education_type.value_counts(normalize = True)


# In[454]:


pd.crosstab(categorical_df['name_education_type'], categorical_df['status'], margins = True)


# In[455]:


edu_total = categorical_df.groupby(by = ['name_education_type']).size().reset_index(name = 'times')
edu_total


# In[456]:


edu_reject = categorical_df[categorical_df.status == 1].groupby(by = ['name_education_type']).size().reset_index(name = 'reject_times')
edu_reject


# In[457]:


edu_reject_rate = pd.merge(edu_total, edu_reject, how = 'outer', on = ['name_education_type']).fillna(0)
edu_reject_rate


# In[458]:


edu_reject_rate['reject_rate'] = edu_reject_rate.reject_times / edu_reject_rate.times
edu_reject_rate


# In[459]:


plt.subplots(figsize = (15, 8))
sns.barplot(y = "name_education_type", x = "reject_rate", data = edu_reject_rate, orient = 'h')
plt.grid(False)
plt.show()


# #### Marital Status Analysis

# In[460]:


categorical_df.head()


# In[461]:


categorical_df.info()


# In[462]:


categorical_df.isnull().sum()


# In[463]:


categorical_df.name_family_status.nunique()


# In[464]:


categorical_df.name_family_status.unique()


# In[465]:


categorical_df.name_family_status.value_counts()


# In[466]:


categorical_df.name_family_status.value_counts(normalize = True)


# In[467]:


pd.crosstab(categorical_df['name_family_status'], categorical_df['status'], margins = True)


# In[468]:


ms_total = categorical_df.groupby(by = ['name_family_status']).size().reset_index(name = 'times')
ms_total


# In[469]:


ms_reject = categorical_df[categorical_df.status == 1].groupby(by = ['name_family_status']).size().reset_index(name = 'reject_times')
ms_reject


# In[470]:


ms_reject_rate = pd.merge(ms_total, ms_reject, how = 'outer', on = ['name_family_status']).fillna(0)
ms_reject_rate


# In[471]:


ms_reject_rate['reject_rate'] = ms_reject_rate.reject_times / ms_reject_rate.times
ms_reject_rate


# In[472]:


plt.subplots(figsize = (15, 8))
sns.barplot(x = "name_family_status", y = "reject_rate", data = ms_reject_rate)
plt.grid(False)
plt.show()


# #### House Type Analysis

# In[473]:


categorical_df.head()


# In[474]:


categorical_df.info()


# In[475]:


categorical_df.isnull().sum()


# In[476]:


categorical_df.name_housing_type.nunique()


# In[477]:


categorical_df.name_housing_type.unique()


# In[478]:


categorical_df.name_housing_type.value_counts()


# In[479]:


categorical_df.name_housing_type.value_counts(normalize = True)


# In[480]:


pd.crosstab(categorical_df['name_housing_type'], categorical_df['status'], margins = True)


# In[481]:


h_total = categorical_df.groupby(by = ['name_housing_type']).size().reset_index(name = 'times')
h_total


# In[482]:


h_reject = categorical_df[categorical_df.status == 1].groupby(by = ['name_housing_type']).size().reset_index(name = 'reject_times')
h_reject


# In[483]:


h_reject_rate = pd.merge(h_total, h_reject, how = 'outer', on = ['name_housing_type']).fillna(0)
h_reject_rate


# In[484]:


h_reject_rate['reject_rate'] = h_reject_rate.reject_times / h_reject_rate.times
h_reject_rate


# In[485]:


plt.subplots(figsize = (15, 10))
sns.barplot(y = "name_housing_type", x = "reject_rate", data = h_reject_rate, orient = 'h')
plt.grid(False)
plt.show()


# #### Occupation Analysis

# In[486]:


categorical_df.head()


# In[487]:


categorical_df.info()


# In[488]:


categorical_df.isnull().sum()


# In[489]:


categorical_df.occupation_type.nunique()


# In[490]:


categorical_df.occupation_type.unique()


# In[491]:


categorical_df.occupation_type.value_counts()


# In[492]:


categorical_df.occupation_type.value_counts().sum()


# In[493]:


# Verify the records to fill / replace

pensioner = categorical_df.loc[categorical_df.name_income_type == 'Pensioner'].loc[categorical_df.employed_years == 0] #.groupby(by = ['name_family_status']).size().reset_index(name = 'reject_times')
pensioner.count()


# In[494]:


pensioner.shape[0]


# In[495]:


pensioner.name_income_type.count()


# In[496]:


pensioner.employed_years.count()


# In[497]:


# By checking multiple conditions

categorical_df['occupation_type'] = np.where((categorical_df['name_income_type'] == 'Pensioner') & (categorical_df['employed_years'] == 0), 'Retired', categorical_df['occupation_type'])


# In[498]:


categorical_df['occupation_type'].value_counts()


# In[499]:


categorical_df.isnull().sum()


# * There are still pending null values in the occupations variable to be treated.

# In[500]:


# Impute pending missing values by creating a new category 'Others' in the Occupation column

categorical_df['occupation_type'] = categorical_df['occupation_type'].fillna("Others")

# applications['OCCUPATION_TYPE'].fillna(value='Other', inplace=True)


# In[501]:


categorical_df.occupation_type.value_counts()


# In[502]:


categorical_df.occupation_type.value_counts(normalize = True)


# In[503]:


categorical_df.isnull().sum()


# * There are no more null values in the datasetset now.

# In[504]:


pd.crosstab(categorical_df['occupation_type'], categorical_df['status'], margins = True)


# In[505]:


occ_total = categorical_df.groupby(by = ['occupation_type']).size().reset_index(name = 'times')
occ_total


# In[506]:


occ_reject = categorical_df[categorical_df.status == 1].groupby(by = ['occupation_type']).size().reset_index(name = 'reject_times')
occ_reject


# In[507]:


occ_reject_rate = pd.merge(occ_total, occ_reject, how = 'outer', on = ['occupation_type']).fillna(0)
occ_reject_rate


# In[508]:


occ_reject_rate['reject_rate'] = occ_reject_rate.reject_times / occ_reject_rate.times
occ_reject_rate


# In[509]:


plt.subplots(figsize = (30, 20))
sns.barplot(x = "reject_rate", y = "occupation_type", data = occ_reject_rate, orient = 'h')
plt.grid(False)
plt.show()


# #### Drop few insignificant columns

# In[510]:


categorical_df.info()


# In[511]:


categorical_df = categorical_df.drop(['days_birth', 'days_employed'], axis = 1)


# In[512]:


categorical_df.head()


# In[513]:


categorical_df.shape


# In[514]:


# Drop the more variables:-

categorical_df = categorical_df.drop(['cnt_children', 'cnt_fam_members', 'income_bucket', 'age_bucket'], axis = 1)


# In[515]:


categorical_df.head()


# In[516]:


categorical_df.shape


# In[517]:


cleaned_df = categorical_df.copy()


# In[518]:


cleaned_df.head()


# In[519]:


# Save the cleaned EDA dataset

cleaned_df.to_csv('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\cleaned_df_final.csv', index = False)


# #### EDA ENDS HERE !!!

# ## VISUALIZATIONS

# ### UNIVARIATE ANALYSIS

# ### Continuous Variables

# In[520]:


continuous_df.head()


# In[521]:


plt.rcParams.update({'figure.figsize': (12.0, 8.0)})


# #### cnt_children

# In[522]:


# Distribution Plot

plt.subplots(figsize = (12, 8))

sns.distplot(continuous_df['cnt_children'], hist = True, kde = True)

plt.title('Histogram-cum-Density Plot of Children Count\n')
plt.xlabel('\nTotal Children')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### amt_income_total

# In[523]:


# Distribution Plot

plt.subplots(figsize = (12, 8))

sns.distplot(continuous_df['amt_income_total'], hist = True, kde = True)

plt.title('Histogram-cum-Density Plot of Total Income\n')
plt.xlabel('\nTotal Income')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### cnt_fam_members

# In[524]:


# Distribution Plot

plt.subplots(figsize = (12, 8))

sns.distplot(continuous_df['cnt_fam_members'], hist = True, kde = True)

plt.title('Histogram-cum-Density Plot of Family Members Count\n')
plt.xlabel('\nTotal Family Members')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### age

# In[525]:


# Distribution Plot

plt.subplots(figsize = (12, 8))

sns.distplot(continuous_df['age'], hist = True, kde = True)

plt.title('Histogram-cum-Density Plot of Age\n')
plt.xlabel('\nAge')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### employed_years

# In[526]:


# Distribution Plot

plt.subplots(figsize = (12, 8))

sns.distplot(continuous_df['employed_years'], hist = True, kde = True)

plt.title('Histogram-cum-Density Plot of Years of Current Employment\n')
plt.xlabel('\nYears of Current Employment')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### status

# In[527]:


# Distribution Plot

plt.subplots(figsize = (12, 8))

sns.distplot(continuous_df['status'], hist = True, kde = True)

plt.title('Histogram-cum-Density Plot of Eligibility\n')
plt.xlabel('\nEligible Vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# ### Categorical Variables

# In[528]:


categorical_df.head()


# #### code_gender

# In[529]:


# Bar Plot

categorical_df['code_gender'].value_counts(normalize = True).sort_index().plot.bar()

plt.title('Bar Plot of Gender Distribution\n')
plt.xlabel('\nGender')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### flag_own_car

# In[530]:


# Bar Plot

categorical_df['flag_own_car'].value_counts(normalize = True).sort_index().plot.bar()

plt.title('Bar Plot of Cars Ownership\n')
plt.xlabel('\nCar')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### flag_own_realty

# In[531]:


# Bar Plot

categorical_df['flag_own_realty'].value_counts(normalize = True).sort_index().plot.bar()

plt.title('Bar Plot of Property Ownership\n')
plt.xlabel('\nProperty')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### name_income_type

# In[532]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['name_income_type'].value_counts(normalize = True).sort_values(ascending=False).plot.bar()

plt.title('Bar Plot of Income Type\n')
plt.xlabel('\nIncome Type')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### name_education_type

# In[533]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['name_education_type'].value_counts(normalize = True).sort_values(ascending=False).plot.bar()

plt.title('Bar Plot of Education Type\n')
plt.xlabel('\nEducation Type')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0, fontsize = 16)
plt.yticks(rotation = 0, fontsize = 16)

plt.grid(False)


# #### name_family_status

# In[534]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['name_family_status'].value_counts(normalize = True).sort_values(ascending=False).plot.bar()

plt.title('Bar Plot of Marital Status\n')
plt.xlabel('\nMarital Type')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### name_housing_type

# In[535]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['name_housing_type'].value_counts(normalize = True).sort_values(ascending=False).plot.bar()

plt.title('Bar Plot of Housing Type\n')
plt.xlabel('\nHousing Type')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)
plt.xticks(rotation = 0, fontsize = 16)
plt.yticks(rotation = 0, fontsize = 16)

plt.grid(False)


# #### occupation_type

# In[536]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['occupation_type'].value_counts(normalize = True).sort_values(ascending=True).plot.barh()

plt.title('Bar Plot of Occupation Type\n')
plt.xlabel('\nOccupation Type')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)
plt.xticks(rotation = 0, fontsize = 16)
plt.yticks(rotation = 0, fontsize = 16)

plt.grid(False)


# #### flag_work_phone

# In[537]:


# Bar Plot

#plt.subplots(figsize = (16, 8))

categorical_df['flag_work_phone'].value_counts(normalize = True).sort_index().plot.bar()

plt.title('Bar Plot of Work Phone\n')
plt.xlabel('\nWork Phone')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### flag_phone

# In[538]:


# Bar Plot

#plt.subplots(figsize = (16, 8))

categorical_df['flag_phone'].value_counts(normalize = True).sort_index().plot.bar()

plt.title('Bar Plot of Phone\n')
plt.xlabel('\nPhone')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### flag_email

# In[539]:


# Bar Plot

#plt.subplots(figsize = (16, 8))

categorical_df['flag_email'].value_counts(normalize = True).sort_index().plot.bar()

plt.title('Bar Plot of Email\n')
plt.xlabel('\nEmail')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### children_cnt_bucket

# In[540]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['children_cnt_bucket'].value_counts(normalize = True).sort_values(ascending=False).plot.bar()

plt.title('Bar Plot of Children Count Bucket\n')
plt.xlabel('\nChildren Count Bucket')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)

plt.grid(False)


# #### cnt_fam_members_bucket

# In[541]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['cnt_fam_members_bucket'].value_counts(normalize = True).sort_values(ascending=False).plot.bar()

plt.title('Bar Plot of Family Members Count Bucket\n')
plt.xlabel('\nFamily Members Count Bucket')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)
plt.xticks(rotation = 0, fontsize = 16)
plt.yticks(rotation = 0, fontsize = 16)

plt.grid(False)


# #### status

# In[542]:


# Bar Plot

plt.subplots(figsize = (16, 8))

categorical_df['status'].value_counts(normalize = True).sort_values(ascending=False).plot.bar()

plt.title('Bar Plot of Eligibility\n')
plt.xlabel('\nEligible vs Non-Eligible')
plt.ylabel('Percentage\n')
plt.xticks(rotation = 0)
plt.xticks(rotation = 0, fontsize = 16)
plt.yticks(rotation = 0, fontsize = 16)

plt.grid(False)


# ### BI-VARIATE ANALYSIS

# ### Continuous Variables Vs Categorical Variable

# In[543]:


continuous_df.head()


# In[544]:


# Correlation


# In[545]:


continuous_df[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years', 'status']].corr()


# * There  is a strong correlation between cnt_children and cnt_fam_members variables of 89%

# In[546]:


# Scatter plot to view the correlation pattern

sns.scatterplot(continuous_df.cnt_children, continuous_df.cnt_fam_members)
#plt.ylim(0,25)
plt.grid(False)
plt.show()


# #### cnt_children

# In[547]:


# KDE Plot

plt.subplots(figsize = (12, 8))

sns.kdeplot(data = continuous_df, x = 'cnt_children', hue = 'status', fill = True)

plt.title('KDE Plot of Children Count with Eligibility\n')
plt.xlabel('\nChildren Count')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### amt_income_total

# In[548]:


# KDE Plot

plt.subplots(figsize = (12, 8))

sns.kdeplot(data = continuous_df, x = 'amt_income_total', hue = 'status', fill = True)

plt.title('KDE Plot of Total Income with Eligibility\n')
plt.xlabel('\nTotal Income')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### cnt_fam_members

# In[549]:


# KDE Plot

plt.subplots(figsize = (12, 8))

sns.kdeplot(data = continuous_df, x = 'cnt_fam_members', hue = 'status', fill = True)

plt.title('KDE Plot of Family Members Count with Eligibility\n')
plt.xlabel('\nFamily Members Count')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### age

# In[550]:


# KDE Plot

plt.subplots(figsize = (12, 8))

sns.kdeplot(data = continuous_df, x = 'age', hue = 'status', fill = True)

plt.title('KDE Plot of Age with Eligibility\n')
plt.xlabel('\nAge')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# #### employed_years

# In[551]:


# KDE Plot

plt.subplots(figsize = (12, 8))

sns.kdeplot(data = continuous_df, x = 'employed_years', hue = 'status', fill = True)

plt.title('KDE Plot of Years of Current Employment with Eligibility\n')
plt.xlabel('\nFamily Members Count')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# ### Categorical Variables Vs Categorical Variable

# In[552]:


categorical_df.head()


# In[553]:


# Group-by

continuous_df.groupby(by = 'status').agg('mean')[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']]


# * In the above table, we can see that the average across all the variables of all the eligible and not-eligible applicants is almost similar. This shows that any applicant can be rejected or approved for the credit card.

# #### code_gender

# In[554]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'code_gender')

plt.title('Count Plot of Gender with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[555]:


pd.crosstab(categorical_df.code_gender, categorical_df.status, margins = True)


# In[556]:


all = pd.crosstab(categorical_df.code_gender, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.code_gender, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for both the genders is very close (87% is close to 89%). Hence, it seems that there wasn’t any discrimination against any gender.

# #### flag_own_car

# In[557]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'flag_own_car')

plt.title('Count Plot of Cars with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[558]:


pd.crosstab(categorical_df.flag_own_car, categorical_df.status, margins = True)


# In[559]:


all = pd.crosstab(categorical_df.flag_own_car, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.flag_own_car, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for both the car-ownership and non-car-ownership is very close (88% is close to 88%). Hence, it seems that there wasn’t any discrimination against any car owners or non-owners.

# #### flag_own_realty

# In[560]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'flag_own_realty')

plt.title('Count Plot of Property with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[561]:


pd.crosstab(categorical_df.flag_own_realty, categorical_df.status, margins = True)


# In[562]:


all = pd.crosstab(categorical_df.flag_own_realty, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.flag_own_realty, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for both the property-owners and non-property-owner is very close (87% is close to 89%). Hence, it seems that there wasn’t any discrimination against any property owners or non-owners.

# #### name_income_type

# In[563]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'name_income_type')

plt.title('Count Plot of Income Type with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[564]:


pd.crosstab(categorical_df.name_income_type, categorical_df.status, margins = True)


# In[565]:


all = pd.crosstab(categorical_df.name_income_type, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.name_income_type, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for all the income types is very close (87% is close to 91%). Hence, it seems that there wasn’t any discrimination against any income types.

# #### name_education_type

# In[566]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'name_education_type')

plt.title('Count Plot of Education Type with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[567]:


pd.crosstab(categorical_df.name_education_type, categorical_df.status, margins = True)


# In[568]:


all = pd.crosstab(categorical_df.name_education_type, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.name_education_type, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for the Academic degree type (78%) is lower than the others (85% to 90%). Hence, we need to check if there was any discrimination against the Academic degree type!

# In[569]:


# Group-by

continuous_df.groupby(by = 'status').agg('mean')[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']]


# In[570]:


# Now filter rows by Academic degree

continuous_df[continuous_df.name_education_type == 'Academic degree'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# In[571]:


# Now filter rows by Lower secondary

continuous_df[continuous_df.name_education_type == 'Lower secondary'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is not clear that why the applicant with an Academic degree education has a higher rejection rate than the applicant with a lower secondary education.
# 
# * Almost on all the counts the applicant with an Academic degree education served better than the applicant with a lower secondary education.

# #### name_family_status

# In[572]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'name_family_status')

plt.title('Count Plot of Marital Status with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[573]:


pd.crosstab(categorical_df.name_family_status, categorical_df.status, margins = True)


# In[574]:


all = pd.crosstab(categorical_df.name_family_status, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.name_family_status, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for all the marital status types is very close (87% is close to 89%). Hence, it seems that there wasn’t any discrimination against any marital status types.

# #### name_housing_type

# In[575]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'name_housing_type')

plt.title('Count Plot of Housing Type with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[576]:


pd.crosstab(categorical_df.name_housing_type, categorical_df.status, margins = True)


# In[577]:


all = pd.crosstab(categorical_df.name_housing_type, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.name_housing_type, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for all the housing types is very close (85% is close to 92%). Hence, it seems that there wasn’t any discrimination against any housing types.

# #### flag_work_phone

# In[578]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'flag_work_phone')

plt.title('Count Plot of Work Phone with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[579]:


pd.crosstab(categorical_df.flag_work_phone, categorical_df.status, margins = True)


# In[580]:


all = pd.crosstab(categorical_df.flag_work_phone, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.flag_work_phone, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for all the applicants with a work phone is similar (88%). Hence, it seems that there wasn’t any discrimination against the applicants with or without a work phone.

# #### flag_phone

# In[581]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'flag_phone')

plt.title('Count Plot of Phone with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[582]:


pd.crosstab(categorical_df.flag_phone, categorical_df.status, margins = True)


# In[583]:


all = pd.crosstab(categorical_df.flag_phone, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.flag_phone, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for all the applicants with a phone is very close (88% is close to 89%). Hence, it seems that there wasn’t any discrimination against the applicants with or without a phone.

# #### flag_email

# In[584]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'flag_email')

plt.title('Count Plot of Email with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.grid(False)
plt.show()


# In[585]:


pd.crosstab(categorical_df.flag_email, categorical_df.status, margins = True)


# In[586]:


all = pd.crosstab(categorical_df.flag_email, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.flag_email, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for all the applicants with a email is very close (86% is close to 88%). Hence, it seems that there wasn’t any discrimination against the applicants with or without a email.

# #### occupation_type

# In[587]:


# Count Plot

plt.subplots(figsize = (14, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'occupation_type')

plt.title('Count Plot of Occupation Type with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.legend(bbox_to_anchor = (1.01, 1), loc = 'upper left', borderaxespad = 0, fontsize = 12, 
           title = "Occupation Type", title_fontsize = 14)

plt.grid(False)
plt.show()


# In[588]:


pd.crosstab(categorical_df.occupation_type, categorical_df.status, margins = True)


# In[589]:


all = pd.crosstab(categorical_df.occupation_type, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.occupation_type, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for most of the occupation types is quite close (87% is close to 94%). But few of the occupation types like Low-skill Laborers, IT staff, HR staff, Security staff and Medicine staff have higher rejection rates of 19%, 18%, 16%, 15% and 14% respectively. Hence, we need to check if there was any discrimination against these occupation type!

# In[590]:


# Group-by

continuous_df.groupby(by = 'status').agg('mean')[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']]


# In[591]:


# Now filter rows by Low-skill Laborers

continuous_df[continuous_df.occupation_type == 'Low-skill Laborers'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is visible that the applicant with a Low-skill Laborers occupation type has a lower average income than the approved applicants average income. This could be the deciding factor in this case.

# In[592]:


# Now filter rows by IT staff

continuous_df[continuous_df.occupation_type == 'IT staff'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is visible that the applicant with an IT staff occupation type has a lower average age as well as the lower years of employment than the approved applicants average age and the years of employment. This could be the deciding factor in this case.

# In[593]:


# Now filter rows by HR staff

continuous_df[continuous_df.occupation_type == 'HR staff'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is visible that the applicant with an HR staff occupation type has a lower average age as well as the lower years of employment than the approved applicants average age and the years of employment. This could be the deciding factor in this case.

# In[594]:


# Now filter rows by Security staff

continuous_df[continuous_df.occupation_type == 'Security staff'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is visible that the applicant with an Security staff occupation type has a lower average income as well as the lower years of employment than the approved applicants average income and the years of employment. This could be the deciding factor in this case.

# In[595]:


# Now filter rows by Medicine staff

continuous_df[continuous_df.occupation_type == 'Medicine staff'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is visible that the applicant with a Medicine staff occupation type has a lower average income than the approved applicants average income. This could be the deciding factor in this case.

# #### children_cnt_bucket

# In[596]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'children_cnt_bucket')

plt.title('Count Plot of Children Count Bucket with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.legend(bbox_to_anchor = (1.01, 1), loc = 'upper left', borderaxespad = 0, fontsize = 12, 
           title = "Occupation Type", title_fontsize = 14)

plt.grid(False)
plt.show()


# In[597]:


pd.crosstab(categorical_df.children_cnt_bucket, categorical_df.status, margins = True)


# In[598]:


all = pd.crosstab(categorical_df.children_cnt_bucket, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.children_cnt_bucket, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for the applicants with children count is quite close (87% is close to 92%) except an applicant with 5 children who has an eligibility rate of 100%. But applicants with more than five children have higher rejection rate of 67%. Hence, we need to check if there was any discrimination against these applicants!

# In[599]:


# Group-by

continuous_df.groupby(by = 'status').agg('mean')[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']]


# In[600]:


# Now filter rows by More than Five

continuous_df[continuous_df.children_cnt_bucket == 'More than Five'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is visible that the applicants with more than five children has a lower average income as well as the lower years of employment than the approved applicants average income and the years of employment. Apart from this, their children and family count is also way beyond the applicants average of the same. Also their average age is also lower than the approved applicants average age. These could be the deciding factor in this case.

# #### cnt_fam_members_bucket

# In[601]:


# Count Plot

plt.subplots(figsize = (12, 8))

sns.countplot(data = categorical_df, x = 'status', hue = 'cnt_fam_members_bucket')

plt.title('Count Plot of Family Members Count Bucket with Eligibility\n')
plt.xlabel('\nEligibile vs Non-Eligible')
plt.ylabel('Percentage\n')

plt.legend(bbox_to_anchor = (1.01, 1), loc = 'upper left', borderaxespad = 0, fontsize = 12, 
           title = "Occupation Type", title_fontsize = 14)

plt.grid(False)
plt.show()


# In[602]:


pd.crosstab(categorical_df.cnt_fam_members_bucket, categorical_df.status, margins = True)


# In[603]:


all = pd.crosstab(categorical_df.cnt_fam_members_bucket, categorical_df.status, margins = True)['All']
pd.crosstab(categorical_df.cnt_fam_members_bucket, categorical_df.status, margins = True).divide(all, axis = 0).dropna()


# * In the above table, we can see that the eligibility percentage for the applicants with family count is quite close (86% is close to 91%) except an applicant with 7 family members who has an eligibility rate of 100%. But applicants with more than seven family members have higher rejection rate of 67%. Hence, we need to check if there was any discrimination against these applicants!

# In[604]:


# Group-by

continuous_df.groupby(by = 'status').agg('mean')[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']]


# In[605]:


# Now filter rows by More than Seven

continuous_df[continuous_df.cnt_fam_members_bucket == 'More than Seven'][['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']].agg('mean')


# * From the above observation it is visible that the applicants with more than seven family members has a lower average income as well as the lower years of employment than the approved applicants average income and the years of employment. Apart from this, their children and family count is also way beyond the applicants average of the same. Also their average age is also lower than the approved applicants average age. These could be the deciding factor in this case.

# ### Continuous Variables Vs Continuous Variable

# In[606]:


continuous_df.head()


# #### amt_income_total vs cnt_children

# In[607]:


# Scatter plot

plt.subplots(figsize = (12, 8))

sns.scatterplot(continuous_df.cnt_children, continuous_df.amt_income_total)
#plt.ylim(0,25)

plt.title('Scatter Plot of Children Count with Total Income\n')
plt.xlabel('\nTotal Children Count')
plt.ylabel('Total Income\n')

plt.grid(False)
plt.show()


# #### amt_income_total vs cnt_fam_members

# In[608]:


# Scatter plot

plt.subplots(figsize = (12, 8))

sns.scatterplot(continuous_df.cnt_fam_members, continuous_df.amt_income_total)
#plt.ylim(0,25)

plt.title('Scatter Plot of Family Members Count with Total Income\n')
plt.xlabel('\nTotal Family Members Count')
plt.ylabel('Total Income\n')

plt.grid(False)
plt.show()


# #### amt_income_total vs age

# In[609]:


# Scatter plot

plt.subplots(figsize = (12, 8))

sns.scatterplot(continuous_df.age, continuous_df.amt_income_total)
#plt.ylim(0,25)

plt.title('Scatter Plot of Age with Total Income\n')
plt.xlabel('\nAge')
plt.ylabel('Total Income\n')

plt.grid(False)
plt.show()


# #### amt_income_total vs employed_years

# In[610]:


# Scatter plot

plt.subplots(figsize = (12, 8))

sns.scatterplot(continuous_df.employed_years, continuous_df.amt_income_total)
#plt.ylim(0,25)

plt.title('Scatter Plot of Years of Employment with Total Income\n')
plt.xlabel('\nYears of Employment')
plt.ylabel('Total Income\n')

plt.grid(False)
plt.show()


# * The applicants who are getting income at zero years of employment are the pensioners.

# ### MULTI-VARIATE ANALYSIS

# ### Continuous Variables Vs Target Variable

# In[611]:


# PairGrid

g = sns.PairGrid(data = continuous_df[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years', 'status']], 
             hue = 'status', size = 2.5, palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)
g.add_legend()


# * For the above graph it is observed that only cnt_children and cnt_fam_members have a strong pattern.
# * This need to be treated by dropping one of them as they are strongly correlated with each other.

# #### Boxplot of Outliers Detection

# In[612]:


boxplot_df = continuous_df.copy()


# In[613]:


boxplot_df.head()


# In[614]:


boxplot_df.info()


# In[615]:


numerical_col2 = boxplot_df[['cnt_children', 'amt_income_total', 'cnt_fam_members', 'age', 'employed_years']]
numerical_col2.head()


# In[616]:


fig , axes = plt.subplots(nrows = 3, ncols = 2, constrained_layout = True)

fig.subplots_adjust(left = 0, bottom = 0, right = 3, top = 5, wspace = 0.09, hspace = 0.3)


for ax, column in zip(axes.flatten(), numerical_col2):
    sns.boxplot(numerical_col2[column], ax = ax)
plt.grid(False)

fig.delaxes(axes[2][1])
plt.show()


# #### VISUALIZATION ENDS HERE !!!

# ## NLP PART

# ### Word CLoud

# In[617]:


from wordcloud import WordCloud


# In[618]:


nlp_df = cleaned_df.copy()


# In[619]:


nlp_df.isna().sum()


# #### name_income_type

# In[620]:


# Create the an object fields and store the categorical columns in it

fields = ['name_income_type', 'name_education_type', 'name_family_status', 'name_housing_type', 'occupation_type']

# Read the dataset (use the cleaned EDA dataset)

text = pd.read_csv('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\cleaned_df_final.csv', usecols = fields)


# In[621]:


inc_text = ' '.join(text['name_income_type'])


# In[622]:


# Creating word_cloud with text as argument in .generate() method

word_cloud_inc = WordCloud(collocations = False, background_color = 'white').generate(inc_text)


# In[623]:


# Generate plot

plt.figure(figsize = (15, 8))
plt.imshow(word_cloud_inc)
plt.title('name_income_type', fontsize = 30)
plt.axis("off")
plt.show()


# #### name_education_type

# In[624]:


edu_text = ' '.join(text['name_education_type'])


# In[625]:


# Creating word_cloud with text as argument in .generate() method

word_cloud_edu = WordCloud(collocations = False, background_color = 'white').generate(edu_text)


# In[626]:


# Generate plot

plt.figure(figsize = (15, 8))
plt.imshow(word_cloud_edu)
plt.title('name_education_type', fontsize = 30)
plt.axis("off")
plt.show()


# #### name_family_status

# In[627]:


fam_text = ' '.join(text['name_family_status'])


# In[628]:


# Creating word_cloud with text as argument in .generate() method

word_cloud_fam = WordCloud(collocations = False, background_color = 'white').generate(fam_text)


# In[629]:


# Generate plot

plt.figure(figsize = (15, 8))
plt.imshow(word_cloud_fam)
plt.title('name_family_status', fontsize = 30)
plt.axis("off")
plt.show()


# #### name_housing_type

# In[630]:


house_text = ' '.join(text['name_housing_type'])


# In[631]:


# Creating word_cloud with text as argument in .generate() method

word_cloud_house = WordCloud(collocations = False, background_color = 'white').generate(house_text)


# In[632]:


# Generate plot

plt.figure(figsize = (15, 8))
plt.imshow(word_cloud_house)
plt.title('name_housing_type', fontsize = 30)
plt.axis("off")
plt.show()


# #### occupation_type

# In[633]:


occ_text = ' '.join(text['occupation_type'])


# In[634]:


# Creating word_cloud with text as argument in .generate() method

word_cloud_occ = WordCloud(collocations = False, background_color = 'white').generate(occ_text)


# In[635]:


# Generate plot

plt.figure(figsize = (15, 8))
plt.imshow(word_cloud_occ)
plt.title('occupation_type', fontsize = 30)
plt.axis("off")
plt.show()


# #### Correlation Matrix

# In[636]:


# Correlation of cleaned dataset categorical_df after EDA
 
plt.figure(figsize = (8, 8), dpi = 80, facecolor = 'white', edgecolor = 'k')

sns.set(font_scale = 2)

hm_corr = sns.heatmap(cleaned_df.corr(), annot = True, vmin = -1, vmax = 1, cmap = 'coolwarm', fmt = '.2f', 
                 cbar_kws = {"shrink": .82, 'label': 'Correlation %'},
                 annot_kws = {"size": 18}, linewidths = 0.1, linecolor = 'white', square = True)

plt.title('Correlation matrix of Cleaned Data (cleaned_df)\n')

hm_corr.set(xlabel = '\nApplicants Details', ylabel = 'Applicants Details\n')

hm_corr.set_xticklabels(hm_corr.get_xmajorticklabels(), fontsize = 12, rotation = 45)

hm_corr.set_yticklabels(hm_corr.get_ymajorticklabels(), fontsize = 12)

plt.savefig('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\corr_matrix_eda2.jpg')

plt.show()


# #### END OF NLP WORD CLOUD

# ### Label Encoding

# In[637]:


encoding_df = cleaned_df.copy()
encoding_df.head()


# In[638]:


from sklearn.preprocessing import LabelEncoder


# In[639]:


# code_gender

encoding_df.code_gender.unique()


# In[640]:


encoding_df.code_gender.value_counts()


# In[641]:


le_code_gender = LabelEncoder()

encoding_df['code_gender'] = le_code_gender.fit_transform(encoding_df['code_gender'])

encoding_df['code_gender'].unique()


# In[642]:


encoding_df.code_gender.value_counts()


# In[643]:


# flag_own_car

encoding_df.flag_own_car.unique()


# In[644]:


encoding_df.flag_own_car.value_counts()


# In[645]:


le_flag_own_car = LabelEncoder()

encoding_df['flag_own_car'] = le_flag_own_car.fit_transform(encoding_df['flag_own_car'])

encoding_df['flag_own_car'].unique()


# In[646]:


encoding_df.flag_own_car.value_counts()


# In[647]:


# flag_own_realty

encoding_df.flag_own_realty.unique()


# In[648]:


encoding_df.flag_own_realty.value_counts()


# In[649]:


le_flag_own_realty = LabelEncoder()

encoding_df['flag_own_realty'] = le_flag_own_realty.fit_transform(encoding_df['flag_own_realty'])

encoding_df['flag_own_realty'].unique()


# In[650]:


encoding_df.flag_own_realty.value_counts()


# In[651]:


# name_income_type

encoding_df.name_income_type.unique()


# In[652]:


encoding_df.name_income_type.value_counts()


# In[653]:


le_name_income_type = LabelEncoder()

encoding_df['name_income_type'] = le_name_income_type.fit_transform(encoding_df['name_income_type'])

encoding_df['name_income_type'].unique()


# In[654]:


encoding_df.name_income_type.value_counts()


# In[655]:


# name_education_type

encoding_df.name_education_type.unique()


# In[656]:


encoding_df.name_education_type.value_counts()


# In[657]:


le_name_education_type = LabelEncoder()

encoding_df['name_education_type'] = le_name_education_type.fit_transform(encoding_df['name_education_type'])

encoding_df['name_education_type'].unique()


# In[658]:


encoding_df.name_education_type.value_counts()


# In[659]:


# name_family_status

encoding_df.name_family_status.unique()


# In[660]:


encoding_df.name_family_status.value_counts()


# In[661]:


le_name_family_status = LabelEncoder()

encoding_df['name_family_status'] = le_name_family_status.fit_transform(encoding_df['name_family_status'])

encoding_df['name_family_status'].unique()


# In[662]:


encoding_df.name_family_status.value_counts()


# In[663]:


# name_housing_type

encoding_df.name_housing_type.unique()


# In[664]:


encoding_df.name_housing_type.value_counts()


# In[665]:


le_name_housing_type = LabelEncoder()

encoding_df['name_housing_type'] = le_name_housing_type.fit_transform(encoding_df['name_housing_type'])

encoding_df['name_housing_type'].unique()


# In[666]:


encoding_df.name_housing_type.value_counts()


# In[667]:


# occupation_type

encoding_df.occupation_type.unique()


# In[668]:


encoding_df.occupation_type.value_counts()


# In[669]:


le_occupation_type = LabelEncoder()

encoding_df['occupation_type'] = le_occupation_type.fit_transform(encoding_df['occupation_type'])

encoding_df['occupation_type'].unique()


# In[670]:


encoding_df.occupation_type.value_counts()


# In[671]:


# flag_work_phone

encoding_df.flag_work_phone.unique()


# In[672]:


encoding_df.flag_work_phone.value_counts()


# In[673]:


le_flag_work_phone = LabelEncoder()

encoding_df['flag_work_phone'] = le_flag_work_phone.fit_transform(encoding_df['flag_work_phone'])

encoding_df['flag_work_phone'].unique()


# In[674]:


encoding_df.flag_work_phone.value_counts()


# In[675]:


# flag_phone

encoding_df.flag_phone.unique()


# In[676]:


encoding_df.flag_phone.value_counts()


# In[677]:


le_flag_phone = LabelEncoder()

encoding_df['flag_phone'] = le_flag_phone.fit_transform(encoding_df['flag_phone'])

encoding_df['flag_phone'].unique()


# In[678]:


encoding_df.flag_phone.value_counts()


# In[679]:


# flag_email

encoding_df.flag_email.unique()


# In[680]:


encoding_df.flag_email.value_counts()


# In[681]:


le_flag_email = LabelEncoder()

encoding_df['flag_email'] = le_flag_email.fit_transform(encoding_df['flag_email'])

encoding_df['flag_email'].unique()


# In[682]:


encoding_df.flag_email.value_counts()


# In[683]:


# children_cnt_bucket

encoding_df.children_cnt_bucket.unique()


# In[684]:


encoding_df.children_cnt_bucket.value_counts()


# In[685]:


le_children_cnt_bucket = LabelEncoder()

encoding_df['children_cnt_bucket'] = le_children_cnt_bucket.fit_transform(encoding_df['children_cnt_bucket'])

encoding_df['children_cnt_bucket'].unique()


# In[686]:


encoding_df.children_cnt_bucket.value_counts()


# In[687]:


# cnt_fam_members_bucket

encoding_df.cnt_fam_members_bucket.unique()


# In[688]:


encoding_df.cnt_fam_members_bucket.value_counts()


# In[689]:


le_cnt_fam_members_bucket = LabelEncoder()

encoding_df['cnt_fam_members_bucket'] = le_cnt_fam_members_bucket.fit_transform(encoding_df['cnt_fam_members_bucket'])

encoding_df['cnt_fam_members_bucket'].unique()


# In[690]:


encoding_df.cnt_fam_members_bucket.value_counts()


# ### CORRELATION MATRIX

# In[691]:


encoding_df.info()


# In[692]:


# Correlation of cleaned dataset encoding_df after Label Encoder
 
plt.figure(figsize = (20, 20), dpi = 80, facecolor = 'white', edgecolor = 'k')

sns.set(font_scale = 2)

hm_corr2 = sns.heatmap(encoding_df.corr(), annot = True, vmin = -1, vmax = 1, cmap = 'coolwarm', fmt = '.2f', 
                 cbar_kws = {"shrink": .82, 'label': 'Correlation %'},
                 annot_kws = {"size": 18}, linewidths = 0.1, linecolor = 'white', square = True)

plt.title('Correlation matrix of Encoded Data (encoding_df)\n')

hm_corr2.set(xlabel = '\nApplicants Details', ylabel = 'Applicants Details\n')

hm_corr2.set_xticklabels(hm_corr2.get_xmajorticklabels(), fontsize = 12, rotation = 45)

hm_corr2.set_yticklabels(hm_corr2.get_ymajorticklabels(), fontsize = 12)

plt.savefig('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\corr_matrix_eda3.jpg')

plt.show()


# #### Check VIF to treat Multicollinearity

# In[693]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[694]:


encoding_df.head()


# In[695]:


vif_data = pd.DataFrame()


# In[696]:


vif_data["Columns"] = encoding_df.columns


# In[697]:


vif_data["VIF"] = [variance_inflation_factor(encoding_df.values, i)
                          for i in range(len(encoding_df.columns))]


# In[698]:


vif_data.sort_values('VIF', ascending = False)


# * Although the VIF of age is 17.48 but we will not drop it, as it's a very significant variable from the business point of view.
# * Instead, we will drop children_cnt_bucket as it is above 10 with a score of 11.47.

# In[699]:


encoding_df = encoding_df.drop(['children_cnt_bucket'], axis = 1)
encoding_df.head()


# In[700]:


encoding_df.shape


# In[701]:


encoding_df.info()


# In[702]:


# Recheck the VIF of the dataset


# In[703]:


vif_data2 = pd.DataFrame()


# In[704]:


vif_data2["Columns"] = encoding_df.columns


# In[705]:


vif_data2["VIF"] = [variance_inflation_factor(encoding_df.values, i)
                          for i in range(len(encoding_df.columns))]


# In[706]:


vif_data2.sort_values('VIF', ascending = False)


# In[707]:


encoding_df.head()


# In[708]:


# Correlation of cleaned dataset encoding_df after Label Encoder
 
plt.figure(figsize = (20, 20), dpi = 80, facecolor = 'white', edgecolor = 'k')

sns.set(font_scale = 2)

hm_corr3 = sns.heatmap(encoding_df.corr(), annot = True, vmin = -1, vmax = 1, cmap = 'coolwarm', fmt = '.2f', 
                 cbar_kws = {"shrink": .82, 'label': 'Correlation %'},
                 annot_kws = {"size": 18}, linewidths = 0.1, linecolor = 'white', square = True)

plt.title('Correlation matrix of Encoded Data (encoding_df)\n')

hm_corr3.set(xlabel = '\nApplicants Details', ylabel = 'Applicants Details\n')

hm_corr3.set_xticklabels(hm_corr3.get_xmajorticklabels(), fontsize = 12, rotation = 45)

hm_corr3.set_yticklabels(hm_corr3.get_ymajorticklabels(), fontsize = 12)

plt.savefig('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\corr_matrix_vif4.jpg')

plt.show()


# * Multicollinearity is taken care of.

# #### Checking p-values of the variables

# In[709]:


import statsmodels.api as sm


# In[710]:


p_value_df = encoding_df.copy()
p_value_df.head()


# In[711]:


X_p_value_df = p_value_df.drop(['status'], axis = 1)


# In[712]:


y_p_value_df = p_value_df['status']


# In[713]:


X_p_value_df = sm.add_constant(X_p_value_df)
model_demo = sm.OLS(y_p_value_df, X_p_value_df)
# model_demo = sm.Logit(y_demo, X_demo)
results = model_demo.fit()
print(results.summary())


# ### Model Building

# In[714]:


model_df = encoding_df.copy()


# In[715]:


model_df.head()


# In[716]:


model_df.shape


# In[717]:


# Save the Dataset for model building

model_df.to_csv('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\model_dataset.csv', index = False)


# In[718]:


# X value contains all the variables except status (target variable)

X = model_df.drop(['status'], axis = 1)


# In[719]:


X.head()


# In[720]:


X.shape


# In[721]:


# y contains only status (target variable)

y = model_df['status']


# In[722]:


y.head()


# In[723]:


y.shape


# #### Split the dataset

# In[724]:


# We create the test train split first

from sklearn.model_selection import train_test_split


# In[725]:


X_balanced, X_test_balanced, y_balanced, y_test_balanced = train_test_split(X , y, test_size = 0.3, random_state = 42, stratify = y)


# In[726]:


encoding_df.status.value_counts() / encoding_df.shape[0]


# In[727]:


y_balanced.value_counts() / len(y_balanced)


# In[728]:


y_test_balanced.value_counts() / len(y_test_balanced)


# In[729]:


X_balanced.shape


# In[730]:


y_balanced.shape


# In[731]:


X_test_balanced.shape


# In[732]:


y_test_balanced.shape


# In[733]:


X_balanced.head()


# In[734]:


y_balanced.head()


# In[735]:


y_balanced.value_counts()


# In[736]:


X_test_balanced.head()


# In[737]:


y_test_balanced.head()


# In[738]:


y_test_balanced.value_counts()


# Insights:-
# * By using the stratify method in the train_test split, we will maintain the same imbalance ratio of Eligible and Non-Eligible Candidates.
# * We will now implement different models to see which one performs the best.

# #### Apply the models

# In[739]:


# Import the model libraries

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# In[740]:


classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighbors" : KNeighborsClassifier(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier(),
    "XGBoost" : XGBClassifier(),
    "GradientBoostingClassifier" : GradientBoostingClassifier()
}


# In[741]:


scores_dict = {}

for key, classifier in classifiers.items():
    classifier.fit(X_balanced, y_balanced)
    
    train_score = classifier.score(X_balanced, y_balanced)
   
    test_score = classifier.score(X_test_balanced, y_test_balanced)
    
    scores_dict[key] = {"Train Score" : train_score, "Test Score" : test_score}

for key, value in scores_dict.items():
    print("\n{} :".format(key))
    for key1, value1 in value.items():
        print("\t{}\t : {}".format(key1, value1))


# #### Insights:-
# 
# * We see that among the models above, XGBoost model is performing best on the train set as well as test set with the accuracies of 89.87% and 88.69%.
# * Also the the variation between the train and test of XGBoost is also very minimal.
# * Therefore, we will use the XGBoost model to predict our values.

# ### Predict using the best Model as per the Test Score - XGBoost

# In[742]:


plt.rcParams.update({'figure.figsize': (12.0, 8.0)})


# In[743]:


#xgb = XGBClassifier()

xgb = XGBClassifier()

#model = xgb.fit(X_balanced, y_balanced)

xgb.fit(X_balanced, y_balanced)


# In[744]:


# Check the probability of the Eligible applicants

xgb.predict_proba(X_test_balanced)


# In[745]:


# Predict the eligibility of the applicants

xgb_pred = xgb.predict(X_test_balanced)


# In[746]:


#print(prediction)

xgb_pred


# In[747]:


xgb_pred.shape


# #### Model Evaluation

# In[748]:


from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score


# In[749]:


print(classification_report(y_test_balanced, xgb_pred))


# In[750]:


Accuracy = metrics.accuracy_score(y_test_balanced, xgb_pred)
Precision = metrics.precision_score(y_test_balanced, xgb_pred)
Sensitivity_recall = metrics.recall_score(y_test_balanced, xgb_pred)
Specificity = metrics.recall_score(y_test_balanced, xgb_pred, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, xgb_pred)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[751]:


# RMSE Computation

rmse = np.sqrt(MSE(y_test_balanced, xgb_pred))
print("RMSE : % f" %(rmse))


# In[752]:


# Accuracy Score

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test_balanced, xgb_pred)))


# In[753]:


# Confusion Matrix Chart

confusion_matrix = metrics.confusion_matrix(y_test_balanced, xgb_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(XGBoostClassifier)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[754]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced,  xgb_pred)

auc = metrics.roc_auc_score(y_test_balanced, xgb_pred)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(XGBoostClassifier)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# ### Feature Importance - XGBoost

# In[755]:


# Find the feature importance

importances = pd.DataFrame(data = {
    'Attribute': X_test_balanced.columns,
    'Importance': xgb.feature_importances_
})

importances = importances.sort_values(by = 'Importance', ascending = False)


# In[756]:


importances


# In[757]:


# Visually plot the feature importances

plt.bar(x = importances['Attribute'], height = importances['Importance'], color = '#087E8B')
plt.title('Feature Importances obtained from coefficients - XGBoostClassifier', size = 20)
plt.xticks(rotation = 'vertical')
plt.grid(False)
plt.show()


# ## Check other Models One-by-One

# ### Logistic Regression

# In[758]:


classifierLR = LogisticRegression(random_state = 42)
classifierLR.fit(X_balanced, y_balanced)


# In[759]:


classifierLR.classes_


# In[760]:


classifierLR.intercept_


# In[761]:


classifierLR.coef_


# In[762]:


pred_prob = classifierLR.predict_proba(X_test_balanced)
pred_prob


# In the matrix above, each row corresponds to a single observation. The first column is the probability of the predicted output being zero, that is 1 - 𝑝(𝑥). The second column is the probability that the output is one, or 𝑝(𝑥).

# In[763]:


y_predLR = classifierLR.predict(X_test_balanced)
y_predLR


# In[764]:


y_predLR.shape


# In[765]:


print('Accuracy Score is {:.5}'.format(accuracy_score(y_test_balanced, y_predLR)))


# In[766]:


# Confusion Matrix Chart

confusion_matrix = metrics.confusion_matrix(y_test_balanced, y_predLR)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(LogisticRegression)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# #### Results Explained
# 
# The Confusion Matrix created has four different quadrants:
# 
# * True Negative (Top-Left Quadrant)
# * False Positive (Top-Right Quadrant)
# * False Negative (Bottom-Left Quadrant)
# * True Positive (Bottom-Right Quadrant)
# 
# True means that the values were accurately predicted, False means that there was an error or wrong prediction.

# In[767]:


print(classification_report(y_test_balanced, y_predLR))


# In[768]:


Accuracy = metrics.accuracy_score(y_test_balanced, y_predLR)
Precision = metrics.precision_score(y_test_balanced, y_predLR)
Sensitivity_recall = metrics.recall_score(y_test_balanced, y_predLR)
Specificity = metrics.recall_score(y_test_balanced, y_predLR, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, y_predLR)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[769]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced,  y_predLR)

auc = metrics.roc_auc_score(y_test_balanced, y_predLR)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(LogisticRegression)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# In[770]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, y_predLR))
print("RMSE : % f" %(rmse))


# ### KNeighborsClassifier

# In[771]:


# Find the value of k


# In[772]:


import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[773]:


cost =[]
for i in range(1, 11):
    KM = KMeans(n_clusters = i, max_iter = 500)
    KM.fit(X_test_balanced)
     
    # calculates squared error
    # for the clustered points
    cost.append(KM.inertia_)    
 
# plot the cost against K values
plt.plot(range(1, 11), cost, color ='g', linewidth ='3')
plt.xlabel("Value of K")
plt.ylabel("Squared Error (Cost)")
plt.show()


# In this case the optimal value for k would be 4. (the last elbow bend)

# In[774]:


# Apply the k value in the model


# In[775]:


knn = KNeighborsClassifier(n_neighbors = 4)
  
knn.fit(X_balanced, y_balanced)


# In[776]:


knn_pred_prob = knn.predict_proba(X_test_balanced)
knn_pred_prob


# In[777]:


knn_pred = knn.predict(X_test_balanced)
knn_pred


# In[778]:


knn_pred.shape


# In[779]:


print('Accuracy Score is {:.5}'.format(accuracy_score(y_test_balanced, knn_pred)))


# In[780]:


# Confusion Matrix Chart

confusion_matrix = metrics.confusion_matrix(y_test_balanced, knn_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(KNeighborsClassifier)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[781]:


print(classification_report(y_test_balanced, knn_pred))


# In[782]:


Accuracy = metrics.accuracy_score(y_test_balanced, knn_pred)
Precision = metrics.precision_score(y_test_balanced, knn_pred)
Sensitivity_recall = metrics.recall_score(y_test_balanced, knn_pred)
Specificity = metrics.recall_score(y_test_balanced, knn_pred, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, knn_pred)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[783]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced,  knn_pred)

auc = metrics.roc_auc_score(y_test_balanced, knn_pred)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(KNeighborsClassifier)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# In[784]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, knn_pred))
print("RMSE : % f" %(rmse))


# ### Decision Tree Classifier

# In[785]:


# Create Decision Tree classifer object

clf_dt = DecisionTreeClassifier()


# In[786]:


# Train Decision Tree Classifer

clf_dt = clf_dt.fit(X_balanced, y_balanced)


# In[787]:


#Predict probabilities for test dataset

clf_dt.predict_proba(X_test_balanced)


# In[788]:


#Predict the response for test dataset

y_pred_dt = clf_dt.predict(X_test_balanced)


# In[789]:


y_pred_dt


# In[790]:


# Model Accuracy, how often is the classifier correct?

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test_balanced, y_pred_dt)))


# In[791]:


# Confusion Matrix Chart

confusion_matrix = metrics.confusion_matrix(y_test_balanced, y_pred_dt)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(DecisionTreeClassifier)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[792]:


print(classification_report(y_test_balanced, y_pred_dt))


# In[793]:


Accuracy = metrics.accuracy_score(y_test_balanced, y_pred_dt)
Precision = metrics.precision_score(y_test_balanced, y_pred_dt)
Sensitivity_recall = metrics.recall_score(y_test_balanced, y_pred_dt)
Specificity = metrics.recall_score(y_test_balanced, y_pred_dt, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, y_pred_dt)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[794]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced, y_pred_dt)

auc = metrics.roc_auc_score(y_test_balanced, y_pred_dt)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(DecisionTreeClassifier)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# In[795]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, y_pred_dt))
print("RMSE : % f" %(rmse))


# In[796]:


# Decision Tree - Gini


# In[797]:


clf_dt_gini = DecisionTreeClassifier(criterion = "gini", max_depth = 3, min_samples_leaf = 5, random_state = 42)


# In[798]:


clf_dt_gini = clf_dt_gini.fit(X_balanced, y_balanced)


# In[799]:


clf_dt_gini.predict_proba(X_test_balanced)


# In[800]:


y_pred_dt_gini = clf_dt.predict(X_test_balanced)
y_pred_dt_gini


# In[801]:


# Model Accuracy, how often is the classifier correct?

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test_balanced, y_pred_dt_gini)))


# In[802]:


# Confusion Matrix Chart

confusion_matrix = metrics.confusion_matrix(y_test_balanced, y_pred_dt_gini)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(DecisionTreeClassifier - Gini)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[803]:


print(classification_report(y_test_balanced, y_pred_dt_gini))


# In[804]:


Accuracy = metrics.accuracy_score(y_test_balanced, y_pred_dt_gini)
Precision = metrics.precision_score(y_test_balanced, y_pred_dt_gini)
Sensitivity_recall = metrics.recall_score(y_test_balanced, y_pred_dt_gini)
Specificity = metrics.recall_score(y_test_balanced, y_pred_dt_gini, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, y_pred_dt_gini)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[805]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced, y_pred_dt_gini)

auc = metrics.roc_auc_score(y_test_balanced, y_pred_dt_gini)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(DecisionTreeClassifier - Gini)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# In[806]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, y_pred_dt_gini))
print("RMSE : % f" %(rmse))


# In[807]:


from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus


# In[808]:


dot_data = StringIO()
export_graphviz(clf_dt_gini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X_test_balanced.columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree_gini.png')
Image(graph.create_png())


# In[809]:


# Decision Tree - Entropy


# In[810]:


clf_dt_ent = DecisionTreeClassifier(criterion = "entropy", max_depth = 3, min_samples_leaf = 5, random_state = 42)


# In[811]:


clf_dt_ent = clf_dt_ent.fit(X_balanced, y_balanced)


# In[812]:


clf_dt_ent.predict_proba(X_test_balanced)


# In[813]:


y_pred_dt_ent = clf_dt.predict(X_test_balanced)
y_pred_dt_ent


# In[814]:


# Model Accuracy, how often is the classifier correct?

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test_balanced, y_pred_dt_ent)))


# In[815]:


# Confusion Matrix Chart

confusion_matrix = metrics.confusion_matrix(y_test_balanced, y_pred_dt_ent)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(DecisionTreeClassifier - Entropy)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[816]:


print(classification_report(y_test_balanced, y_pred_dt_ent))


# In[817]:


Accuracy = metrics.accuracy_score(y_test_balanced, y_pred_dt_ent)
Precision = metrics.precision_score(y_test_balanced, y_pred_dt_ent)
Sensitivity_recall = metrics.recall_score(y_test_balanced, y_pred_dt_ent)
Specificity = metrics.recall_score(y_test_balanced, y_pred_dt_ent, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, y_pred_dt_ent)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[818]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced, y_pred_dt_ent)

auc = metrics.roc_auc_score(y_test_balanced, y_pred_dt_ent)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(DecisionTreeClassifier - Entropy)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# In[819]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, y_pred_dt_ent))
print("RMSE : % f" %(rmse))


# In[820]:


dot_data = StringIO()
export_graphviz(clf_dt_ent, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X_test_balanced.columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree_entropy.png')
Image(graph.create_png())


# ### Random Forest Classifier

# In[821]:


from sklearn.ensemble import RandomForestClassifier

# create regressor object
# clf_rf = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 42)
clf_rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

  
# fit the regressor with x and y data
clf_rf.fit(X_balanced, y_balanced) 


# In[822]:


y_pred_rf = clf_rf.predict(X_test_balanced)
y_pred_rf


# In[823]:


y_pred_rf.shape


# In[824]:


# Calculate the accuracy of the model
print(clf_rf.score(X_test_balanced, y_test_balanced))


# In[825]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, y_pred_rf))
print("RMSE : % f" %(rmse))


# In[826]:


# Confusion Matrix Chart

from sklearn.metrics import confusion_matrix

confusion_matrix = metrics.confusion_matrix(y_test_balanced, y_pred_rf)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(RandomForestClassifier)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[827]:


print(classification_report(y_test_balanced, y_pred_rf))


# In[828]:


Accuracy = metrics.accuracy_score(y_test_balanced, y_pred_rf)
Precision = metrics.precision_score(y_test_balanced, y_pred_rf)
Sensitivity_recall = metrics.recall_score(y_test_balanced, y_pred_rf)
Specificity = metrics.recall_score(y_test_balanced, y_pred_rf, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, y_pred_rf)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[829]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced, y_pred_rf)

auc = metrics.roc_auc_score(y_test_balanced, y_pred_rf)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(RandomForestClassifier)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# In[830]:


# Find best number of estimators using RandomForestClassifier


# In[831]:


for w in range(10, 300, 20):
    mod_rf = RandomForestClassifier(n_estimators = w, oob_score = True, n_jobs = 1, random_state = 42)
    mod_rf.fit(X_balanced,y_balanced)
    
    oob = mod_rf.oob_score_
    print('For n_estimators = '+str(w))
    print('oob score is '+str(oob))
    print('*****************')


# In[832]:


for w in range(300, 600, 20):
    mod_rf2 = RandomForestClassifier(n_estimators = w, oob_score = True, n_jobs = 1, random_state = 42)
    mod_rf2.fit(X_balanced, y_balanced)
    
    oob = mod_rf2.oob_score_
    print('For n_estimators = '+str(w))
    print('oob score is '+str(oob))
    print('*****************')


# In[833]:


# Taking 380 as the correct number of estimators becasue it has the highest oob score as 0.884321


# In[834]:


from sklearn.ensemble import RandomForestClassifier

# create regressor object
clf_rf4 = RandomForestClassifier(n_estimators = 380, random_state = 42)
  
# fit the regressor with x and y data
clf_rf4.fit(X_balanced, y_balanced) 


# In[835]:


clf_rf4.predict_proba(X_test_balanced)


# In[836]:


Y_pred_rf4 = clf_rf4.predict(X_test_balanced)
Y_pred_rf4


# In[837]:


Y_pred_rf4.shape


# In[838]:


# RMSE Computation

rmse = np.sqrt(MSE(y_test_balanced, Y_pred_rf4))
print("RMSE : % f" %(rmse))


# In[839]:


# Calculate the accuracy of the model

print(clf_rf4.score(X_test_balanced, y_test_balanced))


# In[840]:


# Confusion Matrix Chart

from sklearn.metrics import confusion_matrix

confusion_matrix = metrics.confusion_matrix(y_test_balanced, Y_pred_rf4)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(RandomForestClassifier - 380 estimators)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[841]:


print(classification_report(y_test_balanced, Y_pred_rf4))


# In[842]:


Accuracy = metrics.accuracy_score(y_test_balanced, Y_pred_rf4)
Precision = metrics.precision_score(y_test_balanced, Y_pred_rf4)
Sensitivity_recall = metrics.recall_score(y_test_balanced, Y_pred_rf4)
Specificity = metrics.recall_score(y_test_balanced, Y_pred_rf4, pos_label = 0)
F1_score = metrics.f1_score(y_test_balanced, Y_pred_rf4)

print({"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
       "Specificity": Specificity, "F1_score": F1_score})


# In[843]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced, Y_pred_rf4)

auc = metrics.roc_auc_score(y_test_balanced, Y_pred_rf4)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(RandomForestClassifier - 380 estimators)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# In[844]:


# Accuracy

clf_rf4.score(X_test_balanced, y_test_balanced)


# ### Feature Importance - RandomForestClassifier

# In[845]:


from sklearn.inspection import permutation_importance


# In[846]:


clf_rf.feature_importances_


# In[847]:


plt.rcParams.update({'figure.figsize': (12.0, 8.0)})

plt.barh(X_test_balanced.columns, clf_rf.feature_importances_)
plt.title('Feature Importances obtained from coefficients - RandomForestClassifier\n')
plt.grid(False)


# In[848]:


sorted_idx = clf_rf.feature_importances_.argsort()
plt.barh(X_test_balanced.columns[sorted_idx], clf_rf.feature_importances_[sorted_idx])
#plt.xlabel("Random Forest Feature Importance")
plt.title('Feature Importances obtained from coefficients - RandomForestClassifier\n')
plt.grid(False)


# ### Build the Random Forest model on selected features

# * Now, I will drop the least important feature flag_email from the model, rebuild the model and check its effect on accuracy.

# In[849]:


# drop the least important feature from X_balanced and X_test_balanced


# In[850]:


X_balanced_fi = X_balanced.copy()


# In[851]:


X_balanced_fi.head()


# In[852]:


X_balanced_fi = X_balanced_fi.drop(['flag_email'], axis=1)


# In[853]:


X_balanced_fi.shape


# In[854]:


X_balanced_fi.head()


# In[855]:


X_test_balanced_fi = X_test_balanced.copy()


# In[856]:


X_test_balanced_fi.head()


# In[857]:


X_test_balanced_fi = X_test_balanced_fi.drop(['flag_email'], axis=1)


# In[858]:


X_test_balanced_fi.shape


# In[859]:


X_test_balanced_fi.head()


# In[860]:


# Now, I will build the random forest model again and check accuracy


# In[861]:


clf_rf_fi = RandomForestClassifier(n_estimators = 100, random_state = 42)

  
# fit the regressor with x and y data
clf_rf_fi.fit(X_balanced_fi, y_balanced) 


# In[862]:


clf_rf_fi.predict_proba(X_test_balanced_fi)


# In[863]:


y_pred_rf_fi = clf_rf_fi.predict(X_test_balanced_fi)
y_pred_rf_fi


# In[864]:


# Calculate the accuracy of the model
print(clf_rf_fi.score(X_test_balanced_fi, y_test_balanced))


# In[865]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, y_pred_rf_fi))
print("RMSE : % f" %(rmse))


# In[866]:


# Confusion Matrix Chart

from sklearn.metrics import confusion_matrix

confusion_matrix = metrics.confusion_matrix(y_test_balanced, y_pred_rf_fi)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(RandomForestClassifier - Feature Importance 1)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# * There was not any significant change in the accuracy and RMSE score.
# * Both of them dipped slightly further after removing the least significant variable flag_email

# In[867]:


# Drop another least significant variable


# * Now, I will drop the least important feature flag_own_realty from the model, rebuild the model and check its effect on accuracy.

# In[868]:


X_balanced_fi = X_balanced_fi.drop(['flag_own_realty'], axis=1)


# In[869]:


X_balanced_fi.shape


# In[870]:


X_balanced_fi.head()


# In[871]:


X_test_balanced_fi = X_test_balanced_fi.drop(['flag_own_realty'], axis=1)


# In[872]:


X_test_balanced_fi.shape


# In[873]:


X_test_balanced_fi.head()


# In[874]:


# Now, I will build the random forest model again and check accuracy


# In[875]:


clf_rf_fi = RandomForestClassifier(n_estimators = 100, random_state = 42)

  
# fit the regressor with x and y data
clf_rf_fi.fit(X_balanced_fi, y_balanced) 


# In[876]:


clf_rf_fi.predict_proba(X_test_balanced_fi)


# In[877]:


y_pred_rf_fi = clf_rf_fi.predict(X_test_balanced_fi)
y_pred_rf_fi


# In[878]:


# Calculate the accuracy of the model
print(clf_rf_fi.score(X_test_balanced_fi, y_test_balanced))


# In[879]:


# RMSE Computation
rmse = np.sqrt(MSE(y_test_balanced, y_pred_rf_fi))
print("RMSE : % f" %(rmse))


# In[880]:


# Confusion Matrix Chart

from sklearn.metrics import confusion_matrix

confusion_matrix = metrics.confusion_matrix(y_test_balanced, y_pred_rf_fi)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(RandomForestClassifier - Feature Importance 2)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# #### Observation:-
# * The model gave the better accurace and RMSE score without dropping flag_email and flag_own_realty.

# ### Gradient Boosting Classifier

# In[881]:


from sklearn.ensemble import GradientBoostingClassifier

gboost = GradientBoostingClassifier(n_estimators = 5000, max_depth = 3, random_state = 42)


# In[882]:


gboost.fit(X_balanced, y_balanced)


# In[883]:


gboost.predict_proba(X_test_balanced)


# In[884]:


preds_gb = gboost.predict(X_test_balanced)
preds_gb


# In[885]:


preds_gb.shape


# In[886]:


# Calculate the accuracy of the model

print(gboost.score(X_test_balanced, y_test_balanced))


# In[887]:


# RMSE Computation

rmse = np.sqrt(MSE(y_test_balanced, preds_gb))
print("RMSE : % f" %(rmse))


# In[888]:


# Confusion Matrix Chart

from sklearn.metrics import confusion_matrix

confusion_matrix = metrics.confusion_matrix(y_test_balanced, preds_gb)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Eligible: 0', 'Not Eligible: 1'])
cm_display.plot(cmap = 'viridis', colorbar = False, xticks_rotation='horizontal')
cm_display.ax_.set_title("CONFUSION MATRIX\n" + "(GradientBoostingClassifier)\n")
plt.yticks(rotation = 90)
plt.grid(False)
plt.show()


# In[889]:


print(classification_report(y_test_balanced, preds_gb))


# In[890]:


# AUC - ROC

fpr, tpr, _ = metrics.roc_curve(y_test_balanced, preds_gb)

auc = metrics.roc_auc_score(y_test_balanced, preds_gb)

# ax = plt.axes()
plt.plot(fpr, tpr, label = 'AUC Score = %.4f'%auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC CURVE' + "\n(GradientBoostingClassifier)\n")
plt.legend(loc = 4)
# ax.set_facecolor("grey")
plt.grid(False)
plt.show()


# ### Save the Final Output

# In[891]:


predictions = pd.DataFrame(y_pred_rf)
predictions


# In[892]:


# Save the Predicted Values to a .csv file

predictions.to_csv('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\predictions.csv')


# In[893]:


# Add prediction column to the test dataframe X_test

predictions.rename(columns = {0:'predicted_status'}, inplace = True)
predictions.head()


# In[894]:


tested_df = X_test_balanced.copy()
tested_df.head()


# In[895]:


tested_df['predicted_status'] = predictions['predicted_status']
tested_df.head()


# In[896]:


tested_df.predicted_status.value_counts()


# In[897]:


# Save the final DataFrame to .csv

final_df = tested_df.copy()
final_df.head()


# In[898]:


final_df.to_csv('D:\\D - Drive\\IPBA\\BYOP\\Capstone Project\\Final - Credit Card Approval Model\\final_df.csv', index = False)


# ### Making a Prediction

# In[899]:


model_df.info()


# In[900]:


X_balanced.head()


# In[901]:


X_balanced.shape


# In[902]:


X_balanced = np.array([['F', 'N', 'Y', 27000, "Working", "Higher education", "Civil marriage", "House / apartment", 'Y', 'N', 'N', "Managers",
                     'Two', 40, 13]])
X_balanced


# In[903]:


X_balanced[:, 0] = le_code_gender.transform(X_balanced[:, 0])
X_balanced[:, 1] = le_flag_own_car.transform(X_balanced[:, 1])
X_balanced[:, 2] = le_flag_own_realty.transform(X_balanced[:, 2])
X_balanced[:, 4] = le_name_income_type.transform(X_balanced[:, 4])
X_balanced[:, 5] = le_name_education_type.transform(X_balanced[:, 5])
X_balanced[:, 6] = le_name_family_status.transform(X_balanced[:, 6])
X_balanced[:, 7] = le_name_housing_type.transform(X_balanced[:, 7])
X_balanced[:, 8] = le_flag_work_phone.transform(X_balanced[:, 8])
X_balanced[:, 9] = le_flag_phone.transform(X_balanced[:, 9])
X_balanced[:, 10] = le_flag_email.transform(X_balanced[:, 10])
X_balanced[:, 11] = le_occupation_type.transform(X_balanced[:, 11])
X_balanced[:, 12] = le_cnt_fam_members_bucket.transform(X_balanced[:, 12])

X_balanced = X_balanced.astype(int)

X_balanced


# In[904]:


y_pred_rf = clf_rf.predict(X_balanced)
print(y_pred_rf)


# In[905]:


if (y_pred_rf[0] == 0):
    print('Congratulations! You are ELIGIBLE for the Credit Card!')
else:
    print('Sorry! You are NOT ELIGIBLE for the Credit Card!')


# ### Create a Pickle file

# In[906]:


import pickle


# In[907]:


data = {"model" : clf_rf, "le_code_gender" : le_code_gender, "le_flag_own_car" : le_flag_own_car,
        "le_flag_own_realty" : le_flag_own_realty, "le_name_income_type" : le_name_income_type,
        "le_name_education_type" : le_name_education_type, "le_name_family_status" : le_name_family_status,
        "le_name_housing_type" : le_name_housing_type, "le_flag_work_phone" : le_flag_work_phone,
        "le_flag_phone" : le_flag_phone, "le_flag_email" : le_flag_email, "le_occupation_type" : le_occupation_type,
        "le_cnt_fam_members_bucket" : le_cnt_fam_members_bucket}

with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[908]:


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)
    
clf_rf_loaded = data["model"]
le_code_gender = data["le_code_gender"]
le_flag_own_car = data["le_flag_own_car"]
le_flag_own_realty = data["le_flag_own_realty"]
le_name_income_type = data["le_name_income_type"]
le_name_education_type = data["le_name_education_type"]
le_name_family_status = data["le_name_family_status"]
le_name_housing_type = data["le_name_housing_type"]
le_flag_work_phone = data["le_flag_work_phone"]
le_flag_phone = data["le_flag_phone"]
le_flag_email = data["le_flag_email"]
le_occupation_type = data["le_occupation_type"]
le_cnt_fam_members_bucket = data["le_cnt_fam_members_bucket"]


# In[909]:


round(clf_rf.predict_proba(X_balanced)[:, 0][0] * 100, 2)


# In[910]:


type(clf_rf.predict_proba(X_balanced)[:, 0][0])


# In[911]:


y_pred_rf = clf_rf_loaded.predict(X_balanced)
y_pred_rf


# In[912]:


if (y_pred_rf[0] == 0):
    print('Congratulations! You are ELIGIBLE for the Credit Card!')
else:
    print('Sorry! You are NOT ELIGIBLE for the Credit Card!')


# # THE END!!!
