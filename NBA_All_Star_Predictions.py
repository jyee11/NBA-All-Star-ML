
# coding: utf-8

# #Predicting NBA All Stars
# ###Jefferson Yee
# ###September 26th, 2018

# ##Background

# Every year in the NBA, 12 players from the Western and Eastern Conference (24 players in total) are selected to be a part of by fans and coaches to be an NBA All-Star. For many of the players selected, being an All-Star means realizing a lifelong dream as well as being honored by your peers, coaches, and fans as one of the NBA's top players. 
# 
# However, every year, players who many believe are deserving of an All-Star selection are "snubbed" every year. Due to this reason, it can be difficult to predict all stars year in and year out. There have also been calls to reform the selection process which the NBA have done over the years.  
# 
# This project serves to apply Machine Learning algorithms to see how well we can predict which players will be nominated as All-Stars based on the data provided. 

# ##Selection Process

# The All-Star Selection process has changed over the years to better represent the player who deserve to be selected. NBA All-Stars have seperated into two categories: starters and non-starters. Prior to 2017, NBA All-Star starters were voted in purely based on a fan voting system where the top fan voted players in their respective position were selected as All-Satr starters based on their conference. This changed in 2017 when Zaza Pachulia was nearly voted in as a starter when he was clearly not deserving of an All-Star selection. In 2017, the NBA changed the voting criteria for NBA All-Star starter selection to the following:
# 
# - 50% Fan Votes
# - 25% Player Votes
# - 25% Media Votes
# 
# Prior to 2013, NBA All-Stars starters were seperated into the following per conference: 
# 
# - 2 Guards
# - 2 Forwards
# - 1 Center 
# 
# However, in 2013, the league decided that All-Stars should be seperated based on the following:  
# 
# - 2 Backcourt Players
# - 3 Froncourt Players. 
# 
# This new distinction was made to better reflect a changing game of basketball that was predicated on "pace and space."

# ##Cleaning the Data

# Before I imported the data, I loaded the numpy, and pandas library so that I could properly clean and prep the data.

# In[1]:

import numpy as np
import pandas as pd


# For my data, I imported the past 10 seasons (2008 to 2018) of player statistics from https://www.basketball-reference.com/ and appended them into a list to allow for quicker, automated data clean up. I also added a Year variable for each dataset based on which year each NBA season was. 

# In[2]:

read_in_data = []
file_name = 'Data/nbadata{}.csv'

for pos, i in enumerate(range(2008,2019)):
    read_in_data.append(pd.read_csv(file_name.format(i)).drop('Rk', axis = 1))
    read_in_data[pos]['Year'] = i


# Before I started cleaning any data, I first did some data exploration to see what needed to be cleaned. I examined the 2008 dataset as a baseline.

# In[3]:

read_in_data[0].head(10)


# In[4]:

read_in_data[0].info()


# In the for loop below, I filled all NaN values with zeros as missing stats for a player can and should be properly represented as zeros. The Points Per Game variable was changed from "PS/G" to "PPG" and the cleaned data was inputted into another list. 

# In[5]:

clean_data = []

for data in read_in_data:
    na_fix = data.fillna(0)
    ppg_fix = na_fix.rename(index = str, columns ={"PS/G" : "PPG"})

    clean_data.append(ppg_fix)


# Because an All-Star designation was not included in the dataset I imported, I had to manually create the All-Star variable. 

# In[6]:

all_stars2008 = ["Allen Iverson", "Kobe Bryant", "Carmelo Anthony", "Tim Duncan", "Yao Ming", 
                 "Carlos Boozer", "Steve Nash", "Dirk Nowitzki", "Chris Paul", "Brandon Roy", 
                 "Amar'e Stoudemire", "David West", "Jason Kidd", "Dwyane Wade", "LeBron James", 
                 "Kevin Garnett", "Dwight Howard", "Ray Allen", "Chauncey Billups", "Caron Butler", 
                 "Chris Bosh", "Richard Hamilton", "Antawn Jamison", "Joe Johnson", "Paul Pierce", 
                 "Rasheed Wallace"]

all_stars2009 = ["Allen Iverson", "Dwyane Wade", "LeBron James", "Kevin Garnett", "Dwight Howard", 
                 "Ray Allen", "Devin Harris", "Joe Johnson", "Jameer Nelson", "Mo Williams", 
                 "Danny Granger", "Rashard Lewis", "Paul Pierce", "Chris Bosh", "Chris Paul", 
                 "Kobe Bryant", "Amar'e Stoudemire", "Tim Duncan","Yao Ming", "Chauncey Billups", 
                 "Tony Parker", "Brandon Roy", "Pau Gasol", "Dirk Nowitzki", 
              "David West", "Shaquille O'Neal"]

all_stars2010 = ["Allen Iverson", "Dwyane Wade", "LeBron James", "Kevin Garnett", "Dwight Howard", 
                 "Joe Johnson", "Rajon Rondo", "Derrick Rose", "Paul Pierce", "Gerald Wallace", 
                 "Chris Bosh", "Al Horford", "David Lee", "Steve Nash", "Kobe Bryant", 
                 "Carmelo Anthony", "Tim Duncan", "Amar'e Stoudemire", "Chauncey Billups", 
                 "Jason Kidd", "Chris Paul", "Brandon Roy", "Deron Williams", "Kevin Durant",
                 "Dirk Nowitzki", "Zach Randolph", "Pau Gasol", "Chris Kaman"]

all_stars2011 = ["Derrick Rose", "Dwyane Wade", "LeBron James", "Amar'e Stoudemire", 
                 "Dwight Howard", "Ray Allen", "Chris Bosh", "Kevn Garnett", "Al Horford", 
                 "Joe Johnson", "Paul Pierce", "Rajon Rondo", "Chris Paul", "Kobe Bryant", 
                 "Kevin Durant", "Carmelo Anthony", "Yao Ming", "Tim Duncan", "Pau Gasol", 
                 "Manu Ginobili", "Blake Griffin", "Kevin Love", "Dirk Nowitzki", 
                 "Russell Westbrook", "Deron Williams"]

all_stars2012 = ["Chris Paul", "Kobe Bryant", "Kevin Durant", "Blake Griffin", "Andrew Bynum", 
                 "LaMarcus Aldridge", "Marc Gasol", "Kevin Love", "Steve Nash", "Dirk Nowitzki", 
                 "Tony Parker", "Russell Westbrook", "Derrick Rose", "Dwyane Wade", "LeBron James", 
                 "Carmelo Anthony", "Dwight Howard", "Chris Bosh", "Luol Deng", "Roy Hibbert", 
                 "Andre Iguodala", "Joe Johnson", "Paul Pierce", "Rajon Rondo", "Deron Williams"]

all_stars2013 = ["Rajon Rondo", "Dwyane Wade", "LeBron James", "Carmelo Anthony", "Kevin Garnett", 
                 "Chris Bosh", "Tyson Chandler", "Luol Deng", "Paul George", "Jrue Holiday", 
                 "Kyrie Irving", "Joakim Noah", "Brook Lopez", "Chris Paul", "Kobe Bryant", 
                 "Kevin Durant", "Blake Griffin", "Dwight Howard", "LaMarcus Aldridge", "Tim Duncan", 
                 "James Harden", "David Lee", "Tony Parker", "Zach Randolph", "Russell Westbrook"]

all_stars2014 = ["Dwyane Wade", "Kyrie Irving", "LeBron James", "Paul George", "Carmelo Anthony", 
                 "Joakim Noah", "Roy Hibbert", "Chris Bosh", "Paul Millsap", "John Wall", 
                 "Joe Johnson", "DeMar DeRozan", "Stephen Curry", "Kobe Bryant", "Kevin Durant", 
                 "Blake Griffin", "Kevin Love", "Dwight Howard", "LaMarcus Aldridge", "Dirk Nowitzki", 
                 "Chris Paul", "James Harden", "Tony Parker", "Damian Lillard", "Anthony Davis"]

all_stars2015 = ["John Wall", "Kyle Lowry", "LeBron James", "Pau Gasol", "Carmelo Anthony", 
                 "Al Horford", "Chris Bosh","Paul Millsap", "Jimmy Butler", "Dwyane Wade", 
                 "Jeff Teague", "Kyrie Irving", "Kyle Korver", "Stephen Curry", "Kobe Bryant", 
                 "Anthony Davis", "Marc Gasol", "Blake Griffin", "LaMarcus Aldridge", 
                 "Tim Duncan", "Kevin Durant", "Klay Thompson", "Russell Westbrook", 
                 "James Harden", "Chris Paul","DeMarcus Cousins", "Damian Lillard", 
                 "Dirk Nowitzki"]

all_stars2016 = ["LaMarcus Aldridge", "Kobe Bryant", "DeMarcus Cousins", "Stephen Curry", 
                 "Anthony Davis", "Kevin Durant", "Draymond Green", "James Harden", 
                 "Kawhi Leonard", "Chris Paul", "Klay Thompson", "Russell Westbrook",
                 "Carmelo Anthony", "Chris Bosh", "Jimmy Butler", "DeMar DeRozan", 
                 "Andre Drummond", "Pau Gasol", "Paul George","Al Horford", "LeBron James", 
                 "Kyle Lowry", "Paul Millsap", "Isaiah Thomas", "Dwyane Wade", "John Wall"]

all_stars2017 = ["Giannis Antetokounmpo", "Carmelo Anthony", "Jimmy Butler", "DeMar DeRozan", 
                 "Paul George", "Kyrie Irving", "LeBron James", "Kevin Love", "Kyle Lowry", 
                 "Paul Millsap", "Isaiah Thomas", "Kemba Walker", "John Wall", "DeMarcus Cousins", 
                 "Stephen Curry", "Anthony Davis", "Kevin Durant", "Marc Gasol", "Draymond Green", 
                 "James Harden", "Gordon Hayward", "DeAndre Jordan", "Kawhi Leonard", "Klay Thompson", 
                 "Russell Westbrook"]
                
all_stars2018 = ["DeMarcus Cousins", "Anthony Davis", "Kevin Durant", "Kyrie Irving", "LeBron James", 
                 "LaMarcus Aldridge", "Bradley Beal", "Goran Dragic", "Andre Drummond", "Paul George", 
                 "Kevin Love", "Victor Oladipo", "Kristaps Porzingis", "Kemba Walker", "John Wall", 
                 "Russell Westbrook", "Giannis Antetokounmpo", "Stephen Curry", "DeMar DeRozan", 
                 "Joel Embiid", "James Harden", "Jimmy Butler", "Draymond Green", "Al Horford", 
                 "Damian Lillard", "Kyle Lowry", "Klay Thompson", "Karl-Anthony Towns"]

all_stars = [all_stars2008, all_stars2009, all_stars2010, all_stars2011, all_stars2012, all_stars2013, 
             all_stars2014, all_stars2015, all_stars2016, all_stars2017, all_stars2018, ]


# The code below adds the All-Star variable to each of the 10 datasets for the 10 NBA seasons.

# In[7]:

def add_all_stars(data, as_list):
    if data.Player in as_list:
        return 1
    else:
        return 0

for data, allstar in zip(clean_data, all_stars):
    data['All_Star'] = data.apply(add_all_stars, axis = 1, as_list = allstar)


# Before proceeding with cleaning the rest of the data, I first combined all the seasons into one large dataset.

# In[8]:

final_data = pd.concat(clean_data).reset_index().drop('index', axis = 1)


# The code below cleans up the Position variable as there are only 5 positions in the NBA: PG, SG, SF, PF, C. For any position listed as "POS-POS" the first position was used as that position designates that a player spends the most time playing that position.

# In[9]:

final_data.Pos = final_data.Pos.astype('category')
final_data.Pos.cat.categories


# In[10]:

def position_fix(data):
    if data.Pos == 'C-PF' or data.Pos == 'C-SF':
        return 'C'
    elif data.Pos == 'PF-C' or data.Pos == 'PF-SF':
        return 'PF'
    elif data.Pos == 'PG-SG':
        return 'PG'
    elif data.Pos == 'SF-PF' or data.Pos == 'SF-SG':
        return 'SF'
    elif data.Pos == 'SG-PF' or data.Pos == 'SG-PG' or data.Pos == 'SG-SF':
        return 'SG'
    else:
        return data.Pos

final_data.Pos = final_data.apply(position_fix, axis = 1)    
final_data.Pos = final_data.Pos.astype('category')


# In my final step in data cleaning, I set and sorted the indexes of the data by Year and Player while also coding the Position variable. 

# In[11]:

final_data.set_index(['Year', 'Player'], inplace = True)
final_data.sort_index(level = ('Year', 'Player'), inplace = True)


# In[12]:

non_num_fix = {"Pos" : {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C":5}}
final_data.replace(non_num_fix, inplace = True)


# Now that all of the data is clean, the data is separated into the input matrix containing the variables and the labeled output. I also wrote out the data so that it can be viewed separately.

# In[13]:

final_data.to_excel("Data/nba_all_star_data.xlsx")


# In[14]:

final_data.head(10)


# In[15]:

final_data.info()


# In[16]:

x = final_data.drop(['Tm','All_Star'], axis = 1).as_matrix()
y = np.array(final_data.All_Star)


# In[17]:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 123)


# ##Model Selection

# For the machine learning models, I decided to focus on Support Vector Clustering (SVC), Binomial Naive Bayes Classifiers, and Random Forests.

# ###Support Vector Clustering

# Support Vector Clustering (SVC) is a type of supervised learning model that finds a hyperplane that can best divde the dataset into two classes. The hyperplane is designated by the support vectors which are the points that are closest to the divide. When using SVC, we want our data points to be as far away from the hyperplane as possible as the farther away from the hyperplane a point is, the more certain we can be of its classification.

# For SVC, I created a function that tunes the algorithm, finding the best parameters for for achieving the best error rate. The best paramteres are reported below as well as the training and testing error rate. 

# In[18]:

#SVM
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[19]:

def svc_param_search(x, y, n):
    Cs = [0.001, 0.01, 0.1, 10]
    gammas = [0.001, 0.01, 0.1, 10]
    param_grid = {'C': Cs, 'gamma': gammas}
    svc_grid_search = GridSearchCV(svm.SVC(), param_grid, cv = n)
    svc_grid_search.fit(x,y)
    svc_grid_search.best_params_
    #return svc_grid_search.cv_results_
    return svc_grid_search.best_params_


# In[20]:

svc_param_search(x_train, y_train, 3)


# In[21]:

svm_model = svm.SVC(C = 10, gamma = 0.001)
svm_model.fit(x_train, y_train)

svm_model.score(x_train,y_train)


# In[22]:

svm_model.score(x_test, y_test)


# ###Bernoulli Naive Bayes Classifier

# Bernoulli Naive Bayes Classifiers is a supervised learning algorithm that implements and applies Bayes Theorem with strong independence assumptions, meaning that it assumes that variables in the dataset are not related to other variables in the dataset. 

# I implemented the Bernoulli Naive Bayes Classifier method below and reported the training and testing error rate. 

# In[23]:

from sklearn.naive_bayes import BernoulliNB

nb_model = BernoulliNB()
nb_model.fit(x_train, y_train)
nb_model.score(x_train, y_train)


# In[24]:

nb_model.score(x_test, y_test)


# ###Random Forest

# Random Forest is a supervised learning algorithm that creates a specified number of decision trees that each get access to a random subset of the data and considers a random subset of the features(variables). This method helps create much more diveristy, resulting in more robust predicitons. For numerical predictions, an average of all the individual decision tree is taken while for classification, a "vote" is taken for the prediction.

# I created a function that tunes the RandomForest function, searching for the parameters that yield the best error rate. 

# In[25]:

from sklearn.ensemble import RandomForestClassifier


# In[48]:

def rf_param_search(x, y, n):
    Estimators = [10, 20, 30 ,40, 50]
    max_feat = ["auto", "sqrt", "log2", None]
    max_dep = [4, 6, 8, 10, None]
    
    param_grid = {'n_estimators': Estimators, 'max_features': max_feat, 'max_depth': max_dep}
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state = 123), param_grid, cv = n)
    rf_grid_search.fit(x,y)
    rf_grid_search.best_params_
    
    #return rf_grid_search.cv_results_
    return rf_grid_search.best_params_


# In[49]:

rf_param_search(x_train, y_train, 3)


# In[46]:

rf_model = RandomForestClassifier(n_estimators = 30, max_features = None, max_depth= None, random_state = 123)
rf_model.fit(x,y)

rf_model.score(x_train, y_train)


# In[47]:

rf_model.score(x_test, y_test)


# In[30]:

import pickle


# In[31]:

pickle.dump(rf_model, open("nba_random_forest.sav", 'wb'))


# ##Conclusion

# In the summary table above, we can see that Random Forest algorithm yields the best correct prediction score with 99.97% for training data and 99.92% testing data. This is not surprising at all considering that random forest creates many decision trees. For this project, I considered the baseline score as the "just say no" percentage, which means the score I would have gotten if I had just considered every single player to not be an All-Star. That score is 4927/5213 which is 0.9451. (There are 286 All-Stars in the dataset and 5213-286 is 4927.) SVC's testing score (96.40%) was able to beat this baseline score but the Naive Bayes classifier score (94.02% )was lower than the baseline score. 
# 
# Our training and testing score for Random Forest was nearly 100% for both the training and testing score at 99.97% and 99.92% respectively, but the fact that it was not exactly 100% but very close to it actually indicated that the Random Forest model is a very good model. In years before 2017, where All-Star starters were still voted in based purely on fan voting, some players were still always voted in despite failing to perform up to par. A few clear examples are Kobe Bryant in 2014, 2015, and 2016 when he was often injured and clearly not playing at an All-Star level, and Yao Ming in 2011 where he only played 5 games but was still voted by a wide margin due to massive influxes of voting coming from Ming's home country, China. In these cases, players were voted in as All-Star starters due to their past achievements and massive popularity rather than their merits. A less than 100% training and testing score shows that our model is not overfitting to fit these outliers, indicating that the random forest model excelled in making the correct prediction of whether or not an NBA player was an All-Star.
