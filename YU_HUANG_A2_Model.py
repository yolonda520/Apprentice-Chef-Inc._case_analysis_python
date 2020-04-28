# %%timeit
# Student Name : YU HUANG (Yolonda)
# Cohort       : 5
#
# # A2: Regression-Based Analysis (Individual)

# Hult International Business School
#
# ***
# ***
# ***
#################################################################################
# Import Packages
#################################################################################


# importing libraries
import pandas                  as pd # data science essentials
import random                  as rand
import matplotlib.pyplot       as plt # data visualization
import seaborn                 as sns # enhanced data visualization
import statsmodels.formula.api as smf # linear regression (statsmodels)
from sklearn.model_selection import train_test_split # train/test split
from  sklearn.linear_model import LinearRegression # linear regression (scikit-learn)
from sklearn.linear_model import LogisticRegression  # logistic regression
import sklearn.linear_model # linear models

# new libraries
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.neighbors import KNeighborsClassifier   # KNN for classification
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.model_selection import GridSearchCV     # hyperparameter tuning
from sklearn.metrics import make_scorer              # customizable scorer
from sklearn.ensemble import RandomForestClassifier     # random forest
from sklearn.ensemble import GradientBoostingClassifier # gbm
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.svm import SVC


#################################################################################
# Load Data
#################################################################################

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying file name
file = 'Apprentice_Chef_Dataset.xlsx'


# reading the file into Python
original_df = pd.read_excel(file)
chef = original_df.copy()



#################################################################################
#  Feature Engineering and (optional) Dataset Standardization
#################################################################################

# flag the missing value
chef['m_FAMILY_NAME'] = chef['FAMILY_NAME'].isnull().astype(int)
chef.shape


# <strong> Insights from name: </strong>As far, I found that the name didn't include the family name, only include the job in the bracket. And some of the jobs in bracket is fake. So I flagged the missing family name as m_FAMILY_NAME.



########################################
# optimal_neighbors
########################################
def optimal_neighbors(X_data,
                      y_data,
                      standardize = True,
                      pct_test=0.25,
                      seed=802,
                      response_type='class',
                      max_neighbors=100,
                      show_viz=True):
    """
Exhaustively compute training and testing results for KNN across
[1, max_neighbors]. Outputs the maximum test score and (by default) a
visualization of the results.
PARAMETERS
----------
X_data        : explanatory variable data
y_data        : response variable
standardize   : whether or not to standardize the X data, default True
pct_test      : test size for training and validation from (0,1), default 0.25
seed          : random seed to be used in algorithm, default 802
response_type : type of neighbors algorithm to use, default 'class'
    Use 'reg' for regression (KNeighborsRegressor)
    Use 'class' for classification (KNeighborsClassifier)
max_neighbors : maximum number of neighbors in exhaustive search, default 100
show_viz      : display or surpress k-neigbors visualization, default True
"""


    if standardize == True:
        # optionally standardizing X_data
        scaler             = StandardScaler()
        scaler.fit(X_data)
        X_scaled           = scaler.transform(X_data)
        X_scaled_df        = pd.DataFrame(X_scaled)
        X_data             = X_scaled_df



    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size = pct_test,
                                                        random_state = seed)


    # creating lists for training set accuracy and test set accuracy
    training_accuracy = []
    test_accuracy = []


    # setting neighbor range
    neighbors_settings = range(1, max_neighbors + 1)


    for n_neighbors in neighbors_settings:
        # building the model based on response variable type
        if response_type == 'reg':
            clf = KNeighborsRegressor(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)

        elif response_type == 'class':
            clf = KNeighborsClassifier(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)

        else:
            print("Error: response_type must be 'reg' or 'class'")


        # recording the training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))

        # recording the generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))


    # optionally displaying visualization
    if show_viz == True:
        # plotting the visualization
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()


    # returning optimal number of neighbors
    print(f"The optimal number of neighbors is: {test_accuracy.index(max(test_accuracy))+1}")
    return test_accuracy.index(max(test_accuracy))+1


########################################
# visual_cm
########################################
def visual_cm(true_y, pred_y, labels = None):
    """
Creates a visualization of a confusion matrix.

PARAMETERS
----------
true_y : true values for the response variable
pred_y : predicted values for the response variable
labels : , default None
    """
    # visualizing the confusion matrix

    # setting labels
    lbls = labels


    # declaring a confusion matrix object
    cm = confusion_matrix(y_true = true_y,
                          y_pred = pred_y)


    # heatmap
    sns.heatmap(cm,
                annot       = True,
                xticklabels = lbls,
                yticklabels = lbls,
                cmap        = 'Blues',
                fmt         = 'g')


    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of the Classifier')
    plt.show()



########################################
# display_tree
########################################
def display_tree(tree, feature_df, height = 500, width = 800, export = False):
    """
    PARAMETERS
    ----------
    tree       : fitted tree model object
        fitted CART model to visualized
    feature_df : DataFrame
        DataFrame of explanatory features (used to generate labels)
    height     : int, default 500
        height in pixels to which to constrain image in html
    width      : int, default 800
        width in pixels to which to constrain image in html
    export     : bool, defalut False
        whether or not to export the tree as a .png file
    """

    # visualizing the tree
    dot_data = StringIO()


    # exporting tree to graphviz
    export_graphviz(decision_tree      = tree,
                    out_file           = dot_data,
                    filled             = True,
                    rounded            = True,
                    special_characters = True,
                    feature_names      = feature_df.columns)


    # declaring a graph object
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


    # creating image
    img = Image(graph.create_png(),
                height = height,
                width  = width,
                unconfined = True)


    return img


########################################
# split names
########################################

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


# <strong>Insights from length of names:</strong> As we see, some of the names length are longer than usual,I will flag the out usual length.

# In[17]:


chef['number_of_names_out'] = 0
condition = chef.loc[0:,'number_of_names_out'][(chef['number_of_names'] == 5) |
                                               (chef['number_of_names'] == 6) |
                                               (chef['number_of_names'] == 4)]
chef['number_of_names_out'].replace(to_replace = condition,
                                    value      = 1,
                                    inplace    = True)


# <h3>Part II: Working with Email Addresses</h3><br>
#
#
#

# <h4>1: Group email to Categorical Data</h4><br>
# 1.Create an empty list (placeholder_lst)<br>
# 2.Loop over each email in placeholder_lst and SPLIT each personal email based on a common attribute ('@' and use it to SPLIT email and domain).<br>
# 3.APPEND placeholder_lst with split_email.<br>
# 4.Convert placeholder_lst into a DataFrame.<br>
# 5.Display the DataFrame and check your results.<br>
#

# In[18]:


placeholder_lst=[]
for index,email in chef.iterrows():
    split_email = chef.loc[index,'EMAIL'].split(sep = '@')
    placeholder_lst.append(split_email)
email_df = pd.DataFrame(placeholder_lst)
email_df


# Concatenate the email domains as a new column in the email_df DataFrame. Name this column email_domain. Then, print the value counts for each domain.

# In[19]:


email_df.columns = ['not_use','email_domain']
email_df


# Now that email domains have been extracted, let's go one step further to aggregate domains into higher-level categories.  this helps address issues when some categories have small sample sizes.Let's set emails to different group.

# In[20]:


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


# ***
# ***
#
# <h4>2: Conduct one-hot encoding on email group where this has been deemed appropriate.</h4><br>
#

# In[21]:


# create gummies base on domain group
one_email = pd.get_dummies(chef_m['domain_group'])
one_email


# In[22]:


one_email.sum()


# In[23]:


# create new dataframe chef_n include different email types' columns
chef_n = chef.join([one_email])
chef_n.shape


# In[24]:


chef = chef_n.drop(['NAME', 'EMAIL', 'FIRST_NAME', 'FAMILY_NAME'], axis=1)


# In[25]:


chef.shape



# create dummies base on median rate
one_MEDIAN_MEAL_RATING = pd.get_dummies(chef['MEDIAN_MEAL_RATING'])

one_MEDIAN_MEAL_RATING.columns =  ['MEDIAN_MEAL_RATING_1',
                                   'MEDIAN_MEAL_RATING_2',
                                   'MEDIAN_MEAL_RATING_3',
                                   'MEDIAN_MEAL_RATING_4',
                                   'MEDIAN_MEAL_RATING_5']
one_MEDIAN_MEAL_RATING


# In[31]:


chef = chef.join([one_MEDIAN_MEAL_RATING])


# In[32]:


chef.shape


#
#
# <br>
# <h3>Part IV: Outlier Analysis</h3><br>
# let's engineer some new features in the hopes of outperforming our current predictive results.<br><br>
# Plot all the features to see the trend.<br>
# To save running time, I just comment out the visualization.
#






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


# It looks that we delivered BEFORE the alloted delivery time is not performing well. I will flag the EARLY_DELIVERIES as 1167 is 0.

# In[38]:


# I see the EARLY_DELIVERIES with obvious peak at 0, I want to flag this little peak
chef['flag_EARLY_DELIVERIES'] = 0
condition = chef.loc[0:,'flag_EARLY_DELIVERIES'][(chef['EARLY_DELIVERIES'] == 0)]
chef['flag_EARLY_DELIVERIES'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)


# In[39]:


# I see the WEEKLY_PLAN with obvious peak at 0, I want to flag this little peak
chef['flag_WEEKLY_PLAN'] = 0
condition = chef.loc[0:,'flag_WEEKLY_PLAN'][(chef['WEEKLY_PLAN'] == 0)]
chef['flag_WEEKLY_PLAN'].replace(to_replace = condition,
                                 value      = 1,
                                 inplace    = True)


# In[40]:


# I see the CANCELLATIONS_AFTER_NOON with obvious peak at 0, I want to flag this little peak
chef['flag_CANCELLATIONS_AFTER_NOON'] = 0
condition = chef.loc[0:,'flag_CANCELLATIONS_AFTER_NOON'][(chef['CANCELLATIONS_AFTER_NOON'] == 0)]
chef['flag_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition,
                                              value      = 1,
                                              inplace    = True)


# In[41]:


chef.shape


# we are going to assume outlier thresholds base on above plots. Run the following code to generate the thresholds and create outlier flag columns.

# In[42]:


# setting outlier thresholds
TOTAL_MEALS_ORDERED_hi          = 200
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
REVENUE_hi                      = 5000

##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
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


# REVENUE
chef['out_REVENUE'] = 0
condition_hi = chef.loc[0:,'out_REVENUE'][chef['REVENUE'] > REVENUE_hi]
chef['out_REVENUE'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)


# In[44]:


chef.shape


#
# ***
# ***
#
# <h2>STEP III: Preparing Building Predictive Models</h2>
#
# ***
# ***
#
# <h3>Part I: Develop a (Pearson) correlation</h3>
# Now that we've finished developing features (although we could have developed many, many more), we are ready to start the model building process. Before model step, let's see how the explanatory variables correlate to the response variable.<br>
# <br>
# <br>
# <strong>Purpose: Find the relationship between CROSS_SELL_SUCCESS and all of features, so we can figure out which are most important feature to affect CROSS_SELL_SUCCESS.</strong><br><br>
#
# Write a code to develop a (Pearson) correlation matrix with data rounded to two decimal places. Then, print the results rounded to two decimal places.
#

# In[45]:

#
# # creating a (Pearson) correlation matrix
# df_corr = chef.corr().round(2)
#
#
# # printing (Pearson) correlations with SalePrice
# print(df_corr.loc[:,'CROSS_SELL_SUCCESS'].sort_values(ascending = False))


# <strong>Insights from Pearson correlation:</strong> <br>
# 1.The most positive factor is 'FOLLOWED_RECOMMENDATIONS_PCT', which is make sense here. If customers prefer to follow the  recommendations of the service, there is a high probability that they will follow the recommendation to subscribe to new service "Halfway".
#
# 2.The second positive factor is 'PROFESSIONAL_EMAIL' , People have to view professional mail in weekdays, which means that they have high chance to glance the news of Halfway subscribe information than that letters come to personal or junk mail.
#
# 3.The third positive factor is 'NUMBER_OF_NAMES' - This is an interesting insight: if the length of the customers’ name reflects their status, then it is possible that people of a higher status prefer this romantic promotion to receive
# a half bottle of wine from a local California vineyard every Wednesday.
#
# 4.The forth positive factor is 'CANCELLATIONS_BEFORE_NOON'. Customers who canceled the order before noon can receive a full refund, this means that they had more satisfied experience on this application, so perhaps they will glad to subscribe more new service.
#
# 5.The fifth positive factor is 'MOBILE_NUMBER'. Perhaps the application has message news for mobile numbers to increase the buyer's chance to get the new service subscribers.
#
# 6.The most negative factor is 'junk_email'. Which is make sense, since this e-mail is fake and never be read by users and it is possible that they missed the newsletter about the new service.
#

# Analyze the dataset one more time. Drop features where it has been deemed appropriate to do so.

# In[46]:


chef = chef.drop(['unknown_email','MEDIAN_MEAL_RATING','MEDIAN_MEAL_RATING_3'],axis=1)


# In[47]:


chef.shape


# ***
# ***
#
# <h3>Part II - Distance Standardization </h3> <br>
#
# Transform the explanatory variables of a dataset so that they are standardized, or put into a form where each feature's variance is measured on the same scale. In general, distance-based algorithms perform much better after standardization.

# In[48]:


# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with housing_data
scaler.fit(chef)


# TRANSFORMING our data after fit
chef_scaled = scaler.transform(chef)


# converting scaled data into a DataFrame
chef_scaled_df = pd.DataFrame(chef_scaled)

# adding labels to the scaled DataFrame
chef_scaled_df.columns = chef.columns

# checking the results
chef_scaled_df.describe().round(2)


# In[49]:


chef_scaled_df.shape


# ***
# ***
# <h3>Part III: Stratification on explanatory variable and response variable</h3>
# <br>
# When working with classification problems, it is vitally important to understand the balance of the response variable is balanced.
# Declare the explanatory variables in full variables as <strong>logit_full</strong> and significant variables as <strong>logit_sig</strong>, declare the response variable as <strong>chef_target</strong>.

# In[50]:


########################################
# explanatory variable sets
########################################
candidate_dict = {

 # full model
 'logit_full'   : ['REVENUE','TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH',
                   'CONTACTS_W_CUSTOMER_SERVICE', 'PRODUCT_CATEGORIES_VIEWED', 'AVG_TIME_PER_SITE_VISIT',
                   'MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES',
                   'PC_LOGINS', 'MOBILE_LOGINS', 'WEEKLY_PLAN', 'EARLY_DELIVERIES', 'LATE_DELIVERIES', 'PACKAGE_LOCKER',
                   'REFRIGERATED_LOCKER', 'FOLLOWED_RECOMMENDATIONS_PCT', 'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE',
                   'MASTER_CLASSES_ATTENDED', 'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED', 'm_FAMILY_NAME',
                   'number_of_names', 'number_of_names_out', 'junk_email', 'personal_email', 'professional_email',
                   'MEDIAN_MEAL_RATING_1', 'MEDIAN_MEAL_RATING_2', 'MEDIAN_MEAL_RATING_4', 'MEDIAN_MEAL_RATING_5',
                   'flag_MOBILE_LOGINS', 'flag_PC_LOGINS', 'flag_EARLY_DELIVERIES', 'flag_WEEKLY_PLAN',
                   'flag_CANCELLATIONS_AFTER_NOON', 'out_TOTAL_MEALS_ORDERED', 'out_UNIQUE_MEALS_PURCH',
                   'out_CONTACTS_W_CUSTOMER_SERVICE_hi','out_CONTACTS_W_CUSTOMER_SERVICE_lo',
                   'out_AVG_TIME_PER_SITE_VISIT', 'out_CANCELLATIONS_BEFORE_NOON', 'out_CANCELLATIONS_AFTER_NOON',
                   'out_WEEKLY_PLAN', 'out_LATE_DELIVERIES', 'out_AVG_PREP_VID_TIME', 'out_LARGEST_ORDER_SIZE',
                   'out_MASTER_CLASSES_ATTENDED', 'out_AVG_CLICKS_PER_VISIT_hi', 'out_AVG_CLICKS_PER_VISIT_lo',
                   'out_TOTAL_PHOTOS_VIEWED','out_REVENUE'],

# significant variables only
 'logit_sig'    : ['FOLLOWED_RECOMMENDATIONS_PCT', 'professional_email', 'CANCELLATIONS_BEFORE_NOON',
                   'number_of_names','MOBILE_NUMBER', 'junk_email']

}


chef_target =  chef.loc[ : , 'CROSS_SELL_SUCCESS']
chef_full   =  chef_scaled_df.loc[ : , candidate_dict['logit_full']]
chef_sig    =  chef_scaled_df.loc[ : , candidate_dict['logit_sig']]


# ***
# ***
#
# <h3>Part IV - Set up train-test split for both full variables and significant variables </h3>
#
# ***
# ***
#
#

# In[51]:


# train/test split with the logit_full variables
chef_target =  chef.loc[ : , 'CROSS_SELL_SUCCESS']
chef_full   =  chef_scaled_df.loc[ : , candidate_dict['logit_full']]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
            chef_full,
            chef_target,
            random_state = 222,
            test_size    = 0.25,
            stratify     = chef_target)



#################################################################################
# Train/Test Split
#################################################################################

# train/test split with the logit_sig variables
chef_target =  chef.loc[ : , 'CROSS_SELL_SUCCESS']
chef_sig    =  chef_scaled_df.loc[ : , candidate_dict['logit_sig']]

# train/test split
X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(
            chef_sig,
            chef_target,
            random_state = 222,
            test_size    = 0.25,
            stratify     = chef_target)


# ***
# ***
#
#
#

#################################################################################
# Final Model (instantiate, fit, and predict)
#################################################################################


# creating an empty DataFrame
model_performance = pd.DataFrame(columns=['Model',
                                          'Training Accuracy',
                                          'Testing Accuracy',
                                          'AUC Value'
                                         ])
model_performance


#
# ***
# ***
#
# <h2>STEP IV:  Models Building</h2>

# ***
# ***
#

# <br>
# <h3>Part IV: Gradient Boosted Machines Classification on Hyperparameter Tuning with GridSearchCV</h3><br>
#

# perform hyperparameter optimization on a GBM model using the full dataset.

# In[68]:


# ########################################
# # GridSearchCV
# ########################################

# # declaring a hyperparameter space
# learn_space     = pd.np.arange(0.1, 1.1, 0.1)
# estimator_space = pd.np.arange(50, 150, 50)
# depth_space     = pd.np.arange(1, 10)
# subsample_space = pd.np.arange(0.2,1,0.1)

# # creating a hyperparameter grid
# param_grid = {'learning_rate' : learn_space,
#               'max_depth'     : depth_space,
#               'n_estimators'  : estimator_space,
#               'subsample'     : subsample_space}


# # INSTANTIATING the model object without hyperparameters
# full_gbm_grid = GradientBoostingClassifier(random_state = 222)


# # GridSearchCV object
# full_gbm_cv = GridSearchCV(estimator  = full_gbm_grid,
#                            param_grid = param_grid,
#                            cv         = 3,
#                            scoring    = make_scorer(roc_auc_score,
#                                         needs_threshold = False))


# # FITTING to the FULL DATASET (due to cross-validation)
# full_gbm_cv.fit(chef_full, chef_target)


# # PREDICT step is not needed


# # printing the optimal parameters and best score
# print("full Tuned Parameters  :", full_gbm_cv.best_params_)
# print("full Tuned Training AUC:", full_gbm_cv.best_score_.round(4))


# In[69]:


# Record the result of LogisticRegression Model best parameters
print("""
/--------------------------\\
|Gradient Boosted Machines Model |
\\--------------------------/

Full explanatory variables best parameters and best AUC score:
-----------
chef_full Tuned Parameters  :
'learning_rate': 0.2, 'max_depth': 1, 'n_estimators': 50, 'subsample': 0.5

chef_full Tuned AUC      :
0.655

""")


# Manually input the optimal set of hyperparameters when instantiating the model.

# In[70]:


# INSTANTIATING the model object without hyperparameters
full_gbm_tuned = GradientBoostingClassifier(learning_rate = 0.2,
                                            max_depth     = 1,
                                            n_estimators  = 50,
                                            subsample     = 0.5,
                                            random_state  = 222)


# FIT step is needed as we are not using .best_estimator
full_gbm_tuned_fit = full_gbm_tuned.fit(X_train, y_train)


# PREDICTING based on the testing set
full_gbm_tuned_pred = full_gbm_tuned_fit.predict(X_test)


# SCORING the results
print('full Training ACCURACY:', full_gbm_tuned_fit.score(X_train, y_train).round(4))
print('full Testing  ACCURACY:', full_gbm_tuned_fit.score(X_test, y_test).round(4))
print('full AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = full_gbm_tuned_pred).round(4))


# In[71]:


# declaring model performance objects
gbm_train_acc = full_gbm_tuned_fit.score(X_train, y_train).round(4)
gbm_test_acc  = full_gbm_tuned_fit.score(X_test, y_test).round(4)
gbm_auc       = roc_auc_score(y_true  = y_test,
                              y_score = full_gbm_tuned_pred).round(4)


# appending to model_performance
model_performance = model_performance.append(
                          {'Model'             : 'Tuned GBM',
                          'Training Accuracy'  : gbm_train_acc,
                          'Testing Accuracy'   : gbm_test_acc,
                          'AUC Value'          : gbm_auc},
                          ignore_index = True)




# Final model results

model_performance.round(3)


#################################################################################
# Final Model Score (score)
#################################################################################

final_auc_score = roc_auc_score(y_true  = y_test,
                              y_score = full_gbm_tuned_pred).round(4)

print(f"The final AUC score is : {final_auc_score}")
