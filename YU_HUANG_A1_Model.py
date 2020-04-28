
# ################################################################################
# Import Packages
#
# ################################################################################

# importing libraries
import pandas                  as pd # data science essentials
import random                  as rand
import matplotlib.pyplot       as plt # data visualization
import seaborn                 as sns # enhanced data visualization
import statsmodels.formula.api as smf # linear regression (statsmodels)
from sklearn.model_selection import train_test_split # train/test split
from  sklearn.linear_model import LinearRegression # linear regression (scikit-learn)
import sklearn.linear_model # linear models

# new libraries
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# #################################################################################
# Load Data
# ################################################################################

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying file name
file = 'Apprentice_Chef_Dataset.xlsx'

# reading the file into Python
original_df = pd.read_excel(file)

chef = original_df.copy()
#chef.info()
# Insight from the Info: There are 29 variables , 1946 customers were recorded. There is missing value on FAMILY_NAME.

#chef.isnull().sum()



#################################################################################
##Feature Engineering and (optional) Dataset Standardization##

################################################################################

# flag the missing value
chef['m_FAMILY_NAME'] = chef['FAMILY_NAME'].isnull().astype(int)
chef.shape

# As far, I found that the name didn't include the family name, only include the job in the bracket. So I flagged the missing family name as m_FAMILY_NAME.

def split_name(col, df, sep=' ',new_col_name = 'number_of_names'):
    """
    Split the value of a string in series(column of DataFrame), and sum of the number of result items.
    Automatically append summed column into original DataFrame.
    Parameter:
    --------------
    col: The column need to be splited
    df:  DataFrame where column is located
    sep: split by , default ' '
    new_col_name: the name of new column after summing split, default 'number of names'
    --------------
    """
    df[new_col_name] = 0
    for index,val in df.iterrows():
        df.loc[index, new_col_name] = len(df.loc[index, col].split())



split_name(col='NAME',df=chef)
chef['number_of_names'].value_counts()


chef['number_of_names_out'] = 0
condition = chef.loc[0:,'number_of_names_out'][(chef['number_of_names'] == 5) |
                                               (chef['number_of_names'] == 6) |
                                               (chef['number_of_names'] == 4)]
chef['number_of_names_out'].replace(to_replace = condition,
                                    value      = 1,
                                    inplace    = True)



# <h4>1: Group email to Categorical Data</h4><br>
# 1.Create an empty list (placeholder_lst)<br>
# 2.Loop over each email in placeholder_lst and SPLIT each personal email based on a common attribute ('@' and use it to SPLIT email and domain).<br>
# 3.APPEND placeholder_lst with split_email.<br>
# 4.Convert placeholder_lst into a DataFrame.<br>
# 5.Display the DataFrame and check your results.<br>


placeholder_lst=[]
for index,email in chef.iterrows():
    split_email = chef.loc[index,'EMAIL'].split(sep = '@')
    placeholder_lst.append(split_email)
email_df = pd.DataFrame(placeholder_lst)
email_df


# Concatenate the email domains as a new column in the email_df DataFrame. Name this column email_domain. Then, print the value counts for each domain.

email_df.columns = ['not_use','email_domain']
email_df


# Now that email domains have been extracted, let's go one step further to aggregate domains into higher-level categories.  this helps address issues when some categories have small sample sizes.Let's set emails to different group.

# create new dataframe chef_m include column 'domain_group'
chef_m = chef.copy()
# email domain types
personal_email_domains = ['@gmail.com','@yahoo.com','@protonmail.com']
professional_email_domains  = ['@mmm.com','@amex.com','@apple.com','@boeing.com','@caterpillar.com',
                               '@chevron.com','@cisco.com','@cocacola.com','@disney.com','@dupont.com'
                               '@exxon.com','@ge.org','@goldmansacs.com','@ibm.com','@intel.com','@jnj.com'
                               '@jpmorgan.com','@mcdonalds.com','@merck.com','@microsoft.com','@nike.com'
                               '@pfizer.com','@pg.com','@travelers.com','@unitedtech.com','@unitedhealth.com'
                               '@verizon.com','@visa.com','@walmart.com']
junk_email_domain = ['@me.com','@aol.com','@hotmail.com','@live.com','@msn.com','@passport.com']

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for domain in email_df['email_domain']:
        if '@' + domain in personal_email_domains:
            placeholder_lst.append('personal_email')

        elif '@' + domain in professional_email_domains:
            placeholder_lst.append('professional_email')
        elif '@' + domain in junk_email_domain:
            placeholder_lst.append('junk_email')
        else:
            placeholder_lst.append('unknown_email')


# concatenating with email_df
email_df['domain_group'] = pd.Series(placeholder_lst)  #if dataframe only one column, that is series
# # concatenating domain_group with chef DataFrame
chef_m = pd.concat([chef_m, email_df['domain_group']],axis=1)
chef_m.shape



# <h4>2: Encoding Categorical Data- dummy email group</h4><br>

# create gummies base on domain group
one_email = pd.get_dummies(chef_m['domain_group'])

one_email.sum()


# In[32]:


# create new dataframe chef_n include different email types' columns
chef_n = chef.join([one_email])
chef_n.shape


chef = chef_n.drop(['NAME', 'EMAIL', 'FIRST_NAME', 'FAMILY_NAME','unknown_email'], axis=1)

chef.shape

chef.columns



# create dummies base on median rate
one_MEDIAN_MEAL_RATING = pd.get_dummies(chef['MEDIAN_MEAL_RATING'])
one_MEDIAN_MEAL_RATING.columns =  ['MEDIAN_MEAL_RATING_1',
                                   'MEDIAN_MEAL_RATING_2',
                                   'MEDIAN_MEAL_RATING_3',
                                   'MEDIAN_MEAL_RATING_4',
                                   'MEDIAN_MEAL_RATING_5']
one_MEDIAN_MEAL_RATING



chef = chef.join([one_MEDIAN_MEAL_RATING])

#chef.shape


# I see the CONTACTS_W_CUSTOMER_SERVICE with additional outlier peak between[10,11,12], I want to flag this little peak
chef['out_CONTACTS_W_CUSTOMER_SERVICE_outpeak'] = 0
condition = chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE_outpeak'][(chef['CONTACTS_W_CUSTOMER_SERVICE'] == 10) |
                                                                   (chef['CONTACTS_W_CUSTOMER_SERVICE'] == 11) |
                                                                   (chef['CONTACTS_W_CUSTOMER_SERVICE'] == 12)]
chef['out_CONTACTS_W_CUSTOMER_SERVICE_outpeak'].replace(to_replace = condition,
                                                        value      = 1,
                                                        inplace    = True)



# I see the MOBILE_LOGINS with obvious peak of [1,2], I want to flag this little peak
chef['flag_MOBILE_LOGINS'] = 0
condition = chef.loc[0:,'flag_MOBILE_LOGINS'][(chef['MOBILE_LOGINS'] == 1) |
                                              (chef['MOBILE_LOGINS'] == 2) ]
chef['flag_MOBILE_LOGINS'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)



# I see the MOBILE_LOGINS with obvious peak of [1,2], I want to flag this little peak
chef['flag_PC_LOGINS'] = 0
condition = chef.loc[0:,'flag_PC_LOGINS'][(chef['PC_LOGINS'] == 5) |
                                          (chef['PC_LOGINS'] == 6) ]
chef['flag_PC_LOGINS'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)




# I see the EARLY_DELIVERIES with obvious peak at 0, I want to flag this little peak
chef['flag_EARLY_DELIVERIES'] = 0
condition = chef.loc[0:,'flag_EARLY_DELIVERIES'][(chef['EARLY_DELIVERIES'] == 0)]
chef['flag_EARLY_DELIVERIES'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)




# I see the WEEKLY_PLAN with obvious peak at 0, I want to flag this little peak
chef['flag_WEEKLY_PLAN'] = 0
condition = chef.loc[0:,'flag_WEEKLY_PLAN'][(chef['WEEKLY_PLAN'] == 0)]
chef['flag_WEEKLY_PLAN'].replace(to_replace = condition,
                                 value      = 1,
                                 inplace    = True)




# I see the CANCELLATIONS_AFTER_NOON with obvious peak at 0, I want to flag this little peak
chef['flag_CANCELLATIONS_AFTER_NOON'] = 0
condition = chef.loc[0:,'flag_CANCELLATIONS_AFTER_NOON'][(chef['CANCELLATIONS_AFTER_NOON'] == 0)]
chef['flag_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition,
                                              value      = 1,
                                              inplace    = True)




chef.shape


# we are going to assume outlier thresholds base on above plots. Run the following code to generate the thresholds and create outlier flag columns.

# In[52]:


# setting outlier thresholds
TOTAL_MEALS_ORDERED_hi          = 150
UNIQUE_MEALS_PURCH_hi           = 9
CONTACTS_W_CUSTOMER_SERVICE_lo  = 3
CONTACTS_W_CUSTOMER_SERVICE_hi  = 9
AVG_TIME_PER_SITE_VISIT_hi      = 250
CANCELLATIONS_BEFORE_NOON_hi    = 8
CANCELLATIONS_AFTER_NOON_hi     = 2
WEEKLY_PLAN_hi                  = 15
LATE_DELIVERIES_hi              = 10
AVG_PREP_VID_TIME_hi            = 300
LARGEST_ORDER_SIZE_hi           = 8
MASTER_CLASSES_ATTENDED_hi      = 2
AVG_CLICKS_PER_VISIT_lo         = 8
AVG_CLICKS_PER_VISIT_hi         = 18
TOTAL_PHOTOS_VIEWED_hi          = 500


##############################################################################
## Feature Engineering 2 (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers

# TOTAL_MEALS_ORDERED
chef['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = chef.loc[0:,'out_TOTAL_MEALS_ORDERED'][chef['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]
chef['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)


# UNIQUE_MEALS_PURCH
chef['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = chef.loc[0:,'out_UNIQUE_MEALS_PURCH'][chef['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]
chef['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                       value      = 1,
                                       inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE_hi
chef['out_CONTACTS_W_CUSTOMER_SERVICE_hi'] = 0
condition_hi = chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE_hi'][chef['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]
chef['out_CONTACTS_W_CUSTOMER_SERVICE_hi'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)


# CONTACTS_W_CUSTOMER_SERVICE_lo
chef['out_CONTACTS_W_CUSTOMER_SERVICE_lo'] = 0
condition_lo = chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE_lo'][chef['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]
chef['out_CONTACTS_W_CUSTOMER_SERVICE_lo'].replace(to_replace = condition_lo,
                                                   value      = 1,
                                                   inplace    = True)


# AVG_TIME_PER_SITE_VISIT
chef['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = chef.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][chef['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]
chef['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)

# CANCELLATIONS_BEFORE_NOON
chef['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = chef.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][chef['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]
chef['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)

# CANCELLATIONS_AFTER_NOON
chef['out_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = chef.loc[0:,'out_CANCELLATIONS_AFTER_NOON'][chef['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_hi]
chef['out_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                             value      = 1,
                                             inplace    = True)

#WEEKLY_PLAN
chef['out_WEEKLY_PLAN'] = 0
condition_hi = chef.loc[0:,'out_WEEKLY_PLAN'][chef['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]
chef['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# LATE_DELIVERIES
chef['out_LATE_DELIVERIES'] = 0
condition_hi = chef.loc[0:,'out_LATE_DELIVERIES'][chef['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]
chef['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# AVG_PREP_VID_TIME
chef['out_AVG_PREP_VID_TIME'] = 0
condition_hi = chef.loc[0:,'out_AVG_PREP_VID_TIME'][chef['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]
chef['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)

# LARGEST_ORDER_SIZE
chef['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = chef.loc[0:,'out_LARGEST_ORDER_SIZE'][chef['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
chef['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                       value      = 1,
                                       inplace    = True)

# MASTER_CLASSES_ATTENDED
chef['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = chef.loc[0:,'out_MASTER_CLASSES_ATTENDED'][chef['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]
chef['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)

# AVG_CLICKS_PER_VISIT_hi
chef['out_AVG_CLICKS_PER_VISIT_hi'] = 0
condition_hi = chef.loc[0:,'out_AVG_CLICKS_PER_VISIT_hi'][chef['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
chef['out_AVG_CLICKS_PER_VISIT_hi'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)


# AVG_CLICKS_PER_VISIT_lo
chef['out_AVG_CLICKS_PER_VISIT_lo'] = 0
condition_lo = chef.loc[0:,'out_AVG_CLICKS_PER_VISIT_lo'][chef['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]
chef['out_AVG_CLICKS_PER_VISIT_lo'].replace(to_replace = condition_lo,
                                            value      = 1,
                                            inplace    = True)


# TOTAL_PHOTOS_VIEWED
chef['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = chef.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][chef['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]
chef['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)





# setting trend-based thresholds
# CONTACTS_W_CUSTOMER_SERVICE_change_hi     = 10  #trend changes above this point
LARGEST_ORDER_SIZE_change_hi              = 5   # trend changes above this point
# LATE_DELIVERIES_change_hi                 =15   # trend changes above this point
AVG_CLICKS_PER_VISIT_change_hi            =10   # trend changes above this point
TOTAL_PHOTOS_VIEWED_change_hi             =800  # data scatters above this point
TOTAL_MEALS_ORDERED_change_lo             = 20

#MOBILE_LOGINS_change_at                   = 1   # only different at 0
# MEDIAN_MEAL_RATING_change_at              = 4
TOTAL_PHOTOS_VIEWED_change_at             = 0   # zero inflated
# PC_LOGINS_change_at                       = 4   # only different at 4
MASTER_CLASSES_ATTENDED_change_at         = 1   # only different at 3


# In[58]:


chef.shape

##############################################################################
## Feature Engineering 3 (trend changes)                                      ##
##############################################################################



# LARGEST_ORDER_SIZE
chef['change_LARGEST_ORDER_SIZE'] = 0
condition = chef.loc[0:,'change_LARGEST_ORDER_SIZE'][chef['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_change_hi]

chef['change_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                          value      = 1,
                                          inplace    = True)



# TOTAL_MEALS_ORDERED
chef['change_TOTAL_MEALS_ORDERED'] = 0
condition = chef.loc[0:,'change_TOTAL_MEALS_ORDERED'][chef['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_change_lo]

chef['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                           value      = 1,
                                           inplace    = True)

########################################
## change at threshold                ##
########################################

# TOTAL_PHOTOS_VIEWED
chef['change_TOTAL_PHOTOS_VIEWED'] = 0
condition = chef.loc[0:,'change_TOTAL_PHOTOS_VIEWED'][chef['TOTAL_PHOTOS_VIEWED'] == TOTAL_PHOTOS_VIEWED_change_at]

chef['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                           value      = 1,
                                           inplace    = True)

# MASTER_CLASSES_ATTENDED
chef['change_MASTER_CLASSES_ATTENDED'] = 0
condition = chef.loc[0:,'change_MASTER_CLASSES_ATTENDED'][chef['MASTER_CLASSES_ATTENDED'] == MASTER_CLASSES_ATTENDED_change_at]

chef['change_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition,
                                               value      = 1,
                                               inplace    = True)



# preparing x-variables
x_variables =['CROSS_SELL_SUCCESS', 'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE', 'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES', 'PC_LOGINS', 'MOBILE_LOGINS', 'WEEKLY_PLAN', 'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER', 'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED', 'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED', 'm_FAMILY_NAME', 'number_of_names', 'number_of_names_out', 'junk_email', 'personal_email', 'professional_email', 'MEDIAN_MEAL_RATING_1', 'MEDIAN_MEAL_RATING_2', 'MEDIAN_MEAL_RATING_3', 'MEDIAN_MEAL_RATING_4', 'MEDIAN_MEAL_RATING_5', 'out_CONTACTS_W_CUSTOMER_SERVICE_outpeak', 'flag_MOBILE_LOGINS', 'flag_PC_LOGINS', 'flag_EARLY_DELIVERIES', 'flag_WEEKLY_PLAN', 'flag_CANCELLATIONS_AFTER_NOON', 'out_TOTAL_MEALS_ORDERED',
       'out_UNIQUE_MEALS_PURCH', 'out_CONTACTS_W_CUSTOMER_SERVICE_hi', 'out_CONTACTS_W_CUSTOMER_SERVICE_lo', 'out_AVG_TIME_PER_SITE_VISIT', 'out_CANCELLATIONS_BEFORE_NOON', 'out_CANCELLATIONS_AFTER_NOON', 'out_WEEKLY_PLAN', 'out_LATE_DELIVERIES', 'out_AVG_PREP_VID_TIME', 'out_LARGEST_ORDER_SIZE', 'out_MASTER_CLASSES_ATTENDED', 'out_AVG_CLICKS_PER_VISIT_hi', 'out_AVG_CLICKS_PER_VISIT_lo', 'out_TOTAL_PHOTOS_VIEWED', 'change_LARGEST_ORDER_SIZE', 'change_TOTAL_MEALS_ORDERED', 'change_TOTAL_PHOTOS_VIEWED', 'change_MASTER_CLASSES_ATTENDED']



# preparing explanatory variable data
chef_explanatory = chef.loc[:, x_variables]

# preparing response variable data
chef_target =chef.loc[:, 'REVENUE']


chef_explanatory.shape


chef_target =chef.loc[:, 'REVENUE'].astype(int)

type(chef_target[2])


# ***
# ***
#
# <h3>Part III - Distance Standardization </h3> <br>
#
# Transform the explanatory variables of a dataset so that they are standardized, or put into a form where each feature's variance is measured on the same scale. In general, distance-based algorithms (i.e. KNN) perform much better after standardization.

# Standard Scaler:<br>
# Instantiate<br>
# Fit<br>
# Transform<br>
# Convert<br>

# In[71]:


# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with housing_data
scaler.fit(chef_explanatory)


# TRANSFORMING our data after fit
X_scaled = scaler.transform(chef_explanatory)


# converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)


# checking the results
X_scaled_df.describe().round(2)


# Notice that we've lost the column labels. Let's add them back and analyze the variance before and after scaling.

# In[72]:


# adding labels to the scaled DataFrame
X_scaled_df.columns = chef_explanatory.columns


# <h3>Part IV - Training and Testing Sets </h3> <br>
# Purpose: we need to set aside a portion of our data before training our model (known as a testing or validation set). After training, we can use this set to see how our algorithm performs on new data.<br>
#

# In[73]:


# Develop training and testing sets using the standardized dataset.
X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df,
            chef_target,
            test_size = 0.25,
            random_state = 222)
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


# ***
# ***
#
# <h3>Part V - Model Building </h3> <br>
# <h4>1. KNN Model</h4><br>
#

# #################################################################################
# Final Model (instantiate, fit, and predict)
# ################################################################################

# Use a loop and visually inspect the optimal value for k

# In[74]:


# creating lists for training set accuracy and test set accuracy
training_accuracy = []
test_accuracy = []


# building a visualization of 1 to 50 neighbors
neighbors_settings = range(1, 15)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)

    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))

    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

#
# # plotting the visualization
# fig, ax = plt.subplots(figsize=(12,8))
# plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
# plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()
# plt.show()


# finding the optimal number of neighbors
opt_neighbors = test_accuracy.index(max(test_accuracy)) + 1           # +1因为index start 0
print(f"""The optimal number of neighbors is {opt_neighbors}""")



# INSTANTIATING a model with the optimal number of neighbors
knn_opt = KNeighborsRegressor(algorithm ='auto',
                n_neighbors = opt_neighbors)



# FITTING the model based on the training data
knn_opt.fit(X_train, y_train)



# PREDITCING on new data
knn_opt_pred = knn_opt.predict(X_test)



# SCORING the results
print('KNN Training Score:', knn_opt.score(X_train, y_train).round(4))
print('KNN Testing Score:',  knn_opt.score(X_test, y_test).round(4))


# saving scoring data for future use
knn_opt_score_train = knn_opt.score(X_train, y_train).round(4)
knn_opt_score_test  = knn_opt.score(X_test, y_test).round(4)


# <br><h4>2. Ordinary Least Squares Regression</h4>
# Purpose: In order to work with statsmodels, we need to concatenate our training data on the 'x' side (X_train) and our training data on the 'y' side (y_train). Then, we can begin building models and analyze their results.<hr>

# <br><h5> Apply OLS regression model in scikit-learn</h5> <br>
# INSTANTIATE a LinearRegression( ) object<br>
# FIT the training data to the model object<br>
# PREDICT using the testing data<br>
# SCORE your results, rounding to four decimal places<br>


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
# creating a hyperparameter grid
param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False],
    'n_jobs': [None, -1]
}

# INSTANTIATING the model object without hyperparameters
lnreg = LinearRegression()

# GridSearchCV object
lnreg_cv = GridSearchCV(lnreg,
                         param_grid,
                         cv=10)

# FITTING to the data set (due to cross-validation)
lnreg_cv.fit(X_train, y_train)

lnreg_cv



# printing the optimal parameters and best score
print("Tuned Parameters  :", lnreg_cv.best_params_)
print("Tuned CV AUC      :", lnreg_cv.best_score_.round(2))





# applying modelin scikit-learn

# INSTANTIATING a model object
lr = LinearRegression(**lnreg_cv.best_params_)              #INSTANTIATE


# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)


# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)


# SCORING the results
print('lr Training Score:', lr.score(X_train, y_train).round(4))
print('lr Testing Score:',  lr.score(X_test,y_test).round(4))
lr_train_score = lr.score(X_train, y_train)
lr_test_score = lr.score(X_test,y_test)


# <br><h4>3. Ridge Regression Model</h4>




import sklearn.linear_model # linear models
# INSTANTIATING a model object
ridge_model = sklearn.linear_model.Ridge()

# FITTING the training data
ridge_fit  = ridge_model.fit(X_train, y_train)


# PREDICTING on new data
ridge_pred = ridge_fit.predict(X_test)

# SCORING the results
print('Ridge Training Score:', ridge_model.score(X_train, y_train).round(4))
print('Ridge Testing Score:',  ridge_model.score(X_test, y_test).round(4))


# saving scoring data for future use
ridge_train_score = ridge_model.score(X_train, y_train).round(4)
ridge_test_score  = ridge_model.score(X_test, y_test).round(4)


# <br><h4>4.  Lasso Regression Model</h4>



# INSTANTIATING a model object
lasso_model = sklearn.linear_model.Lasso()

# FITTING the training data
lasso_fit = lasso_model.fit(X_train, y_train)


# PREDICTING on new data
lasso_pred = lasso_fit.predict(X_test)

print('Lasso Training Score:', lasso_model.score(X_train, y_train).round(4))
print('Lasso Testing Score:',  lasso_model.score(X_test,y_test).round(4))

# saving scoring data for future use
lasso_train_score = lasso_model.score(X_train, y_train).round(4)
lasso_test_score  = lasso_model.score(X_test,y_test).round(4)


# <br><h4>5.  Bayesian ARD Regression Model</h4>





# <br><h4>6. Random Forest Regression Model</h4>

# In[85]:



rfr = RandomForestRegressor(n_estimators=50,
                           warm_start=True,
                           bootstrap=True,
                           criterion='mse')
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)
print('Rfr Training Score:', rfr.score(X_train, y_train).round(4))
print('Rfr Testing Score:',  rfr.score(X_test,y_test).round(4))
rfr_train_score = rfr.score(X_train, y_train)
rfr_test_score = rfr.score(X_test,y_test)


# <br><h4>7. GBDT Regression Model</h4>

# In[86]:


gbdt = GradientBoostingRegressor(max_depth=2,
                                subsample=0.9,
                                min_samples_leaf=0.009,
                                max_features=0.9,
                                n_estimators=65,
                                random_state=222)
gbdt.fit(X_train, y_train)
gbdt_pred = gbdt.predict(X_test)
# SCORING the results
print('GBDT Training Score:', gbdt.score(X_train, y_train).round(4))
print('GBDT Testing Score:',  gbdt.score(X_test,y_test).round(4))
gbdt_train_score = gbdt.score(X_train, y_train)
gbdt_test_score = gbdt.score(X_test,y_test)


# ***
# ***
#
# <h2>STEP IV: Compare Results</h2> <br>
#
# Let's compare the results of each model. In the interest of time, this code has already been written.

# In[87]:





# creating a dictionary for model results
model_performance = {'Model'    : ['knn','OLS', 'Ridge', 'Lasso','Forest Regression','GBDT'],  # 'ARD'

                     'Training' : [knn_opt_score_train,
                                   lr_train_score,
                                   ridge_train_score,
                                   lasso_train_score,
                                   rfr_train_score,
                                   gbdt_train_score],  # ard_train_score

                     'Testing'  : [knn_opt_score_test,
                                   lr_test_score,
                                   ridge_test_score,
                                   lasso_test_score,
                                   rfr_test_score,
                                   gbdt_test_score]} # ard_test_score


# converting model_performance into a DataFrame
model_performance = pd.DataFrame(model_performance)


# sending model results to Excel
model_performance.to_excel('regression_model_performance.xlsx',
                           index = False)

# model_performance['Total'] = model_performance['Training'] + model_performance['Testing']
# model_performance['Difference'] = abs(model_performance['Training'] - model_performance['Testing'])

model_performance.round(3)


# #################################################################################
# Final Model Score (score)
# ################################################################################

# In[88]:


test_score = gbdt.score(X_test,y_test).round(4)
test_score
