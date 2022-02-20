# Columns in the dataset

## All columns and missing values
0   PassengerId  891 non-null    int64  
1   Survived     891 non-null    int64  
2   Pclass       891 non-null    int64  
3   Name         891 non-null    object 
4   Sex          891 non-null    object 
5   [] Age       714 non-null    float64
6   SibSp        891 non-null    int64  
7   Parch        891 non-null    int64  
8   Ticket       891 non-null    object 
9   Fare         891 non-null    float64
10  [] Cabin     204 non-null    object 
11  [] Embarked  889 non-null    object 

## Survived
[x] Is the dependent value, all visualizations have to use it to evaluate the predictability power.
[x] There are more not survived (549) than survived (342).

## Pclass
[x] Transform to a categorical value.
[] Categorical value. We need to encode it.

## Name
[x] String value, maybe we can associate some lastnames or extract information. 
[x] Create value if have parenthesis
[x] Create value if have quotes
[x] Create value for title (Mrs, Mr, Miss)
[x] Create value for number of words (names)
[x] Create value for numer of letters
[x] Average large of the names
[x] Large of the lastname
[x] Composed lastname
[] Categorical values. We need to encode it.

## Sex
[] Categorical value. We need to encode it.

## Age
[x] It's possible to apply feature engineering to simplify the age state & incorporate non linearity
[x] Some missing values, need to evaluate what to do.
[] Categorical value. We need to encode it.

## SibSp (# siblings / spouses aboard titanic)
[x] The alone people is a interesting case
[x] The people with more than 4 is a interesting case
[] Wee need to scale the values

## Parch (# parents / children aboard titanic)
[x] The alone people is a interesting case
[x] The people with more than 3 is a interesting case
[] More can correlate with more survive probability?

## Ticket
[x] String and number values. Maybe we can extract information from letters or numbers logic.
[x] Extract categories before the code
[x] Analyze the numeric code

## Fare
[x] Clean outliers
[x] Check no linearity
[x] PowerTransformer

## Cabin
[x] String and number values. Maybe we can extract information from letters or numbers logic.
[x] Some missing values, need to evaluate what to do.

## Embarked
[x] Categorical values. We need to encode it.
[x] Two missing values. Minimal, we can delete it.

# Preprocessing

## Scale variables

## One hot encoder

## PCA

# Models

## First Approach
[] Use only the numerical values to evaluate some classification models: KMeans, KNeighbors, XGboost, LightGB, CatBoost

Prediction in train set:
Prediction in test set:

## Second Approach
[] Evaluate models with some alternatives of using or imputing missing values.
[] Evaluate models with some feature engineering.

Prediction in train set:
Prediction in test set: